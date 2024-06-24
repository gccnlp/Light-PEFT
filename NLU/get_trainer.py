import logging
import os
import random
import sys
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from nlu_dataset import NLUDataset
from trainer_base import BaseTrainer, MaskTrainer, SpeedTestTrainer
from nlu_utils import (
    get_model,
    TaskType,
    get_peft_config,
    get_peft_model,
    set_mask,
    get_target_module,
    get_early_model,
    get_early_peft_model,
    SavePeftModelCallback,
    count_trainable_parameters
)

logger = logging.getLogger(__name__)

def get_trainer(model_args,
        data_args,
        training_args,
        lora_args,
        bottleneck_args):

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )
    dataset = NLUDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            cache_dir=training_args.cache_dir
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            cache_dir=training_args.cache_dir
        )

    if not dataset.multiple_choice:
        task_type = TaskType.SEQUENCE_CLASSIFICATION
    else:
        task_type = TaskType.MULTIPLE_CHOICE

    if model_args.early_search==True:
        if model_args.use_mask_trainer==True:
            print("Use Mask Trainer.")
            config.mask_training = True
            model = get_model(model_args, task_type, config)
            if model_args.use_peft==True:
                peft_config = get_peft_config(training_args, lora_args, bottleneck_args, task_type)
                model = get_peft_model(model, peft_config)
            else:
                print("Full Fine-tuning.")
            model = set_mask(model, init_mask=True)
        elif model_args.prune_model_before==True:
            config.mask_training = True
            model, original_config, early_attn_mask_records, early_ffn_mask_records = get_early_model(model_args, training_args, task_type, config)
            peft_config = get_peft_config(training_args, lora_args, bottleneck_args, task_type)
            model = get_peft_model(model, peft_config)
            model = set_mask(model, init_mask=False)
        else:
            assert("Please choose a pruning method.")
    elif model_args.draw_and_training==True:
            print("Draw and Train.")
            config.mask_training = True
            if model_args.use_peft==True:
                model, original_config, early_attn_mask_records, early_ffn_mask_records = get_early_model(model_args, training_args, task_type, config)
                model = get_early_peft_model(model, training_args, lora_args, bottleneck_args, task_type, original_config, early_attn_mask_records, early_ffn_mask_records)
            else:
                model_args.model_name_or_path = training_args.peft_dir
                model, original_config, early_attn_mask_records, early_ffn_mask_records = get_early_model(model_args, training_args, task_type, config)
                print("Full Fine-tuning.")
            model = set_mask(model, init_mask=False)
    else:
        print("Use Vanilla Model")
        if model_args.use_layer_drop==True:
            print("Use Layer Drop")
            config.layer_drop = model_args.layer_drop
        model = get_model(model_args, task_type, config)
        if model_args.use_peft==True:
            peft_config = get_peft_config(training_args, lora_args, bottleneck_args, task_type=task_type)
            model = get_peft_model(model, peft_config)
        else:
            print("Full Fine-tuning.")

    
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    model.config.use_cache = False

    if training_args.local_rank == 0:
        print(model)
        if model_args.use_peft==True:
            model.print_trainable_parameters()
        
    if model_args.use_mask_trainer==True or model_args.prune_model_before==True:
        print("Use Mask Trainer.")
        trainer = MaskTrainer(
            model=model,
            args=training_args,
            l1_loss_coef=training_args.l1_loss_coef,
            attn_mask_lr_coef=training_args.attn_mask_lr_coef,
            ffn_mask_lr_coef=training_args.ffn_mask_lr_coef,
            attn_modules_class=get_target_module(model_type=config.model_type, module="attn"),
            ffn_modules_class=get_target_module(model_type=config.model_type, module="layer"),
            peft_modules_class=get_target_module(model_type="peft", module=training_args.adapter_name),
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key,
            callbacks=[SavePeftModelCallback],
        )
    elif model_args.speed_test==True:
        print("Test Speed.")
        trainer = SpeedTestTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key
        )
    else:
        print("Use Base Trainer.")
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key
        )



    return trainer, model
