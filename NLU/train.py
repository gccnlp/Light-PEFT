import logging
import os
import sys
import numpy as np
from typing import Dict
import torch
import random
import datasets
import transformers
from transformers import set_seed, Trainer, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from nlu_utils import record_mask_param, count_trainable_parameters, count_removed_peft_modules
from arguments import *
from nlu_dataset import task_to_keys as nlu_tasks
NLU_DATASETS = list(nlu_tasks.keys())
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

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
    print("peak mem (G)", torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
    print("load mem (G)", torch.cuda.memory_allocated()/ (1024 * 1024 * 1024))
    print("activation mem (G)", (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / (1024 * 1024 * 1024))