from dataclasses import dataclass, field
from typing import List, Optional, Union
from transformers import TrainingArguments

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    bias: str = "none"
    reserved_ratio: float = 0.4

@dataclass
class BottleneckArguments:
     # bottleneck adapter hyperparams
    bottleneck_size: int = 64
    non_linearity: str = "tanh"
    adapter_dropout: float = 0.1
    use_parallel_adapter: bool = False
    use_adapterp: bool = False
    target_modules: List[str] = field(
        default_factory=lambda: ['fc1','fc2']
    )
    scaling: Union[float, str] = 1.0

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    early_search: bool = False
    pruned_model_path: Optional[str] = field(default="")
    use_llm_pruner: bool = False
    use_mask_trainer: bool = False
    draw_and_training: bool = False
    speed_test: bool = False
    use_peft: bool = True
    use_layer_drop: bool = False
    layer_drop: float = 0.4
    prune_model_before: bool = False


@dataclass
class DataArguments:
    task_name: str = field(
        default=None,
    )
    max_seq_length: int = field(
        default=128,
    )
    dataset_name: str = field(
        default=None, metadata={"help": "Data Name."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Data Name."}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    pad_to_max_length: bool = True
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 4
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
    )

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    adapter_name: str = field(default="lora",metadata={"choices": ["lora", "bottleneck"],})
    peft_dir: str = field(default=None)
    attn_pruning_ratio: float = 0.5
    ffn_pruning_ratio: float = 0.5
    l1_loss_coef: float = 1e-4
    attn_mask_lr_coef: int = 1
    ffn_mask_lr_coef: int = 1
    random_pruning: bool = False
    rank_pruning_ratio: float = 0.5
    random_peft: bool = False
    ffn_pruning_method: str = field(default="layerwise")
    peft_pruning_method: str = field(default="layerwise")
    peft_imp: Optional[str] = field(default=None)
    module_pruning_ratio: float = 0.5