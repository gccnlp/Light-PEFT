import logging
import os
from typing import Dict, OrderedDict
import torch
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers import Trainer
from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import pdb
import time
logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

      

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}


            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

class SpeedTestTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

      

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}


            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        start_time=time.time()
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nForward Pass: {elapsed_time:.10f}")
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        start_time=time.time()
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nBackward Pass: {elapsed_time:.10f}")
        return loss.detach() / self.args.gradient_accumulation_steps


class MaskTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, 
                test_key = "accuracy",
                l1_loss_coef=1e-4,
                attn_mask_lr_coef=100,
                ffn_mask_lr_coef=50,
                attn_modules_class=None,
                ffn_modules_class=None,
                peft_modules_class=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        print("l1_loss_coef:{}, attn_mask_lr_coef:{}, ffn_mask_lr_coef:{}".format(l1_loss_coef, attn_mask_lr_coef, ffn_mask_lr_coef))
        self.l1_loss_coef = l1_loss_coef
        self.attn_mask_lr_coef = attn_mask_lr_coef
        self.ffn_mask_lr_coef = ffn_mask_lr_coef
        self.attn_modules_class = attn_modules_class
        self.peft_mask_lr_coef = 1
        self.ffn_modules_class = ffn_modules_class
        self.peft_modules_class = peft_modules_class

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}


            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            attn_mask_parameters = [name for name,p in self.model.named_parameters() if "early_attn" in name]
            ffn_mask_parameters = [name for name,p in self.model.named_parameters() if "early_ffn" in name]
            peft_mask_parameters = [name for name,p in self.model.named_parameters() if "peft_mask" in name]
            mask_parameters = attn_mask_parameters + ffn_mask_parameters + peft_mask_parameters
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in mask_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in attn_mask_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.attn_mask_lr_coef,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in ffn_mask_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.ffn_mask_lr_coef,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in peft_mask_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.peft_mask_lr_coef,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from fairscale.optim import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)


        for m in model.modules():
            if hasattr(m, "importance_scores"):
                if hasattr(m, "lora_A"):
                    down_module = m.lora_A
                    up_module = m.lora_B
                elif hasattr(m, "adapter_down"):
                    down_module = m.adapter_down
                    up_module = m.adapter_up
                if "p2" in m.peft_imp:
                    indices=(inputs["attention_mask"]==1).detach().cpu().nonzero(as_tuple=True)
                    m.pos_importance_scores[-1] = torch.mean(m.pos_importance_scores[-1][indices])
                if "first_order" in m.peft_imp:
                    w = down_module.weight
                    grad = down_module.weight.grad
                    imp_A = w * grad
                    w = up_module.weight
                    grad = up_module.weight.grad
                    imp_B = w * grad
                    m.importance_scores.append(torch.stack([imp_A.mean(1).abs(),imp_B.mean(0).abs()],dim=0).sum(dim=0).detach().cpu())
                else:
                    pass

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        l1_loss = 0.0
        for m in model.modules():
            if isinstance(m, self.attn_modules_class) and m.self.mask_training:
                l1_loss += m.self.early_attn_mask.abs().sum()
            if isinstance(m, self.ffn_modules_class) and m.mask_training:
                l1_loss += m.early_ffn_mask.abs().sum()
        loss = loss + l1_loss * self.l1_loss_coef

        return (loss, outputs) if return_outputs else loss