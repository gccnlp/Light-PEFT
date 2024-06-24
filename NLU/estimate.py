import logging
import os
import sys
import numpy as np
from typing import Dict
import torch
import pdb
import random
import pickle
import datasets
import transformers
from transformers import set_seed, Trainer, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from nlu_utils import record_mask_param, count_trainable_parameters, count_removed_peft_modules, get_target_module
from arguments import *
from nlu_dataset import task_to_keys as nlu_tasks
NLU_DATASETS = list(nlu_tasks.keys())
from transformers import TrainerCallback, TrainerState, TrainerControl
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
early_attn_mask_records = None
early_ffn_mask_records = None
need_to_record = True


class SuperGLUECallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        def init_mask_records():
            global early_attn_mask_records, early_ffn_mask_records
            early_attn_mask_records = [[] for _ in range(model.config.num_hidden_layers)]
            early_ffn_mask_records = [[] for _ in range(model.config.num_hidden_layers)]
        if args.local_rank == 0:
            init_mask_records()
            peft_mask_records = {}
            with open(os.path.join(args.output_dir, "init_peft_mask_records.pkl"), "wb") as f:
                for n,m in model.named_modules():
                    if hasattr(m, "peft_mask"):
                        #pdb.set_trace()
                        peft_mask_records[n]=m.peft_mask.data
                pickle.dump(peft_mask_records, f)
                f.close()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        def update_records():
            global early_attn_mask_records, early_ffn_mask_records, need_to_record
            if need_to_record:
                early_attn_mask_records, early_ffn_mask_records = record_mask_param(model, early_attn_mask_records, early_ffn_mask_records)
                print("Layer 0 Attn Masks Min={}, Max={}".format(early_attn_mask_records[0][-1].min(), early_attn_mask_records[0][-1].max()))
                print("Layer 0 FFN Masks Min={}, Max={}".format(early_ffn_mask_records[0][-1].min(), early_ffn_mask_records[0][-1].max()))
            else:
                pass

        if args.local_rank == 0:
            update_records()
            
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        def update_records():
            global early_attn_mask_records, early_ffn_mask_records, need_to_record
            if need_to_record:
                early_attn_mask_records, early_ffn_mask_records = record_mask_param(model, early_attn_mask_records, early_ffn_mask_records)
                print("Layer 0 Attn Masks Min={}, Max={}".format(early_attn_mask_records[0][-1].min(), early_attn_mask_records[0][-1].max()))
                print("Layer 0 FFN Masks Min={}, Max={}".format(early_ffn_mask_records[0][-1].min(), early_ffn_mask_records[0][-1].max()))
            else:
                pass
        def save_records():
            global early_attn_mask_records, early_ffn_mask_records, need_to_record
            if need_to_record:
                for i, early_attn_mask in enumerate(early_attn_mask_records):
                    early_attn_mask_records[i] = np.stack(early_attn_mask, axis=0)
                np.save(os.path.join(args.output_dir, 'early_attn_mask_records.npy'),
                        np.stack(early_attn_mask_records, axis=0))
                for i, early_ffn_mask in enumerate(early_ffn_mask_records):
                    early_ffn_mask_records[i] = np.stack(early_ffn_mask, axis=0)
                np.save(os.path.join(args.output_dir, 'early_ffn_mask_records.npy'),
                        np.stack(early_ffn_mask_records, axis=0))
            else:
                pass
        def process_peft_imp(imp):
            total_steps = len(imp)
            data_tensor = torch.tensor(np.stack(imp[total_steps//2:]))
            # data_tensor = torch.tensor(np.stack(m.importance_scores))
            # data_tensor = m.importance_scores[-1]
            data_tensor = torch.where(torch.isnan(data_tensor), torch.full_like(data_tensor, 0), data_tensor) # replace nan and inf
            data_tensor = torch.where(torch.isinf(data_tensor), torch.full_like(data_tensor, 0), data_tensor) # replace nan and inf
            return data_tensor
        def save_peft_imp():
            peft_importance_records = {}
            peft_pos_importance_records = {}
            with open(os.path.join(args.output_dir, "peft_module_records.pkl"), "wb") as f:
                for n,m in model.named_modules():
                    if isinstance(m, get_target_module(model_type="peft", module=args.adapter_name)):
                        if len(m.pos_importance_scores)>0:
                            process_imp = process_peft_imp(m.pos_importance_scores)
                            peft_pos_importance_records[n]=torch.mean(process_imp,dim=0)
                pickle.dump(peft_pos_importance_records, f)
                f.close()            
            with open(os.path.join(args.output_dir, "peft_mask_records.pkl"), "wb") as f:
                for n,m in model.named_modules():
                    if isinstance(m, get_target_module(model_type="peft", module=args.adapter_name)):
                        process_imp = process_peft_imp(m.importance_scores)
                        peft_importance_records[n]=torch.mean(process_imp,dim=0)
                        if hasattr(m, "peft_mask"):
                            peft_importance_records[n]=peft_importance_records[n]*torch.abs(m.peft_mask.data.detach().cpu())
                            
                pickle.dump(peft_importance_records, f)
                f.close()
        
        if args.local_rank == 0:
            update_records()
            save_records()
            if args.peft_imp:
                save_peft_imp()
            

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.add_callback(SuperGLUECallback)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # trainer.save_model()

    metrics = train_result.metrics
    metrics["total_params"], metrics["trainable_params"] = count_trainable_parameters(trainer.model)
    metrics["total_peft_modules"], metrics["removed_peft_modules"], metrics["removed_peft_proportion"] = count_removed_peft_modules(trainer.model)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    #trainer.log_best_metrics()

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

if __name__ == '__main__':
    setup_seed(42)
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, BottleneckArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        bottleneck_args
    ) = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if data_args.task_name.lower() == "superglue" or data_args.task_name.lower() == "glue":
        if data_args.task_name.lower() == "superglue":
            data_args.task_name="super_glue"
        assert data_args.dataset_name.lower() in NLU_DATASETS
        from get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(NLU_DATASETS)))

    set_seed(training_args.seed)
    need_to_record = model_args.use_mask_trainer
    trainer, model = get_trainer(model_args,
        data_args,
        training_args,
        lora_args,
        bottleneck_args)
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
        if training_args.local_rank == 0:
            model.save_pretrained(training_args.output_dir, state_dict=None)

    
    # if training_args.do_eval:
    #     evaluate(trainer)

    # if training_args.do_predict:
    #     predict(trainer, predict_dataset)

   