import re
import os
import string
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
import pdb
from collections import defaultdict, Counter
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments
dir_path = os.path.dirname(os.path.realpath(__file__))  #load current file path
sys.path.insert(0, dir_path + "/..")
from models.hf_roberta.modeling_roberta import (
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaAttention,
    RobertaLayer,
    prune_linear_layer  
)
from enum import Enum
from peft import get_peft_model, LoraConfig, BottleneckConfig, set_peft_model_state_dict
from peft.tuners.lora import Linear as LoRALinear
from peft.tuners.bottleneck import Linear as AdapterLinear
import pickle
class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4
PEFT_MODELS={ 
    TaskType.SEQUENCE_CLASSIFICATION:"SEQ_CLS",
    TaskType.MULTIPLE_CHOICE: "MULTI_CHOICE",
}
AUTO_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaForMultipleChoice,
    },
    # "bert": {
    #     TaskType.SEQUENCE_CLASSIFICATION: BertForSequenceClassification,
    #     TaskType.MULTIPLE_CHOICE: BertForMultipleChoice,
    # },
}
AUTO_MODULES = {
    "roberta": {
        "attn": RobertaAttention,
        "layer": RobertaLayer,
    },
    "peft": {
        "lora": LoRALinear,
        "bottleneck": AdapterLinear,
    }
}


class SavePeftModelCallback(TrainerCallback):
    def on_save(self,training_args: TrainingArguments,state: TrainerState,control: TrainerControl,**kwargs,):
        checkpoint_folder = os.path.join(training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder,exclude_frozen_parameters=True)
        
        for fname in os.listdir(checkpoint_folder):
            if not fname.startswith("adapter"):
                try:
                    os.remove(os.path.join(checkpoint_folder, fname))
                except:
                    pass
        return control
    
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_model(model_args, task_type, config):
    model_class = AUTO_MODELS[config.model_type][task_type]
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                device_map=device_map
            )
    return model

def get_peft_config(training_args, lora_args=None, bottleneck_args=None, task_type=None):
    if training_args.adapter_name == "lora":
        config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.bias,
            task_type=PEFT_MODELS[task_type],
        )
    elif training_args.adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_args.bottleneck_size,
            non_linearity=bottleneck_args.non_linearity,
            adapter_dropout=bottleneck_args.adapter_dropout,
            use_parallel_adapter=bottleneck_args.use_parallel_adapter,
            use_adapterp=bottleneck_args.use_adapterp,
            target_modules=bottleneck_args.target_modules,
            scaling=bottleneck_args.scaling,
            bias="none",
            task_type=PEFT_MODELS[task_type],
        )
    config.peft_imp = training_args.peft_imp
    return config

def record_mask_param(model, early_attn_mask_records, early_ffn_mask_records):
    idx_layer = 0
    for m in model.modules():
        if isinstance(m, AUTO_MODULES[model.config.model_type]["attn"]) and m.self.mask_training:
            early_attn_mask_records[idx_layer].append(m.self.early_attn_mask.detach().cpu().numpy().reshape(-1))
            idx_layer += 1

    idx_layer = 0
    for m in model.modules():
        if isinstance(m, AUTO_MODULES[model.config.model_type]["layer"]) and m.mask_training:
            early_ffn_mask_records[idx_layer].append(m.early_ffn_mask.detach().cpu().numpy().reshape(-1))
            idx_layer += 1
    return early_attn_mask_records, early_ffn_mask_records

def set_mask(model, init_mask=False):
    #layers = model.model.roberta.encoder.layer if model.config.model_type == "roberta" else model.model.bert.encoder.layer
    for m in model.modules():
        if isinstance(m, AUTO_MODULES[model.config.model_type]["attn"]) and m.self.mask_training:
            init.constant_(m.self.early_attn_mask, 1.0)
            m.self.early_attn_mask.requires_grad = init_mask
            m.self.mask_training = init_mask
            m.self.early_attn_mask.to(torch.float32)
        if isinstance(m, AUTO_MODULES[model.config.model_type]["layer"]) and m.mask_training:
            init.constant_(m.early_ffn_mask, 1.0)
            m.early_ffn_mask.requires_grad = init_mask
            m.early_ffn_mask.requires_grad = init_mask
            m.mask_training = init_mask
            m.early_ffn_mask.to(torch.float32)
        if isinstance(m, AUTO_MODULES["peft"]["lora"]) and hasattr(m, "peft_mask"):
            m.peft_mask.requires_grad = init_mask
            m.peft_mask.to(torch.float32)
            
    return model

def prune_bert_attn(model, early_attn_mask_records, use_pruning_step, pruning_ratio=0.4, pruning_method="layerwise"):
    print("Pruning Attention Ratio:{}".format(pruning_ratio))
    attention_modules = []
    for m in model.modules():
        if isinstance(m, AUTO_MODULES[model.config.model_type]["attn"]):
            attention_modules.append(m)
    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_attention_heads)
    if use_pruning_step > 0:
        early_attn_mask_records = early_attn_mask_records[:, use_pruning_step-1, :]
    else:
        # Random pruning
        # Ret internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-use_pruning_step)
        early_attn_mask_records = np.random.rand(len(attention_modules), model.config.num_attention_heads)
        # Reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `early_attn_mask_records`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `early_attn_mask_records`.
    quantile_axis = -1 if pruning_method == 'layerwise' else None
    threshold = np.quantile(early_attn_mask_records, pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = early_attn_mask_records > threshold
    for layer_id, (m, mask) in enumerate(zip(attention_modules, layers_masks)):
        pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
        #logger.info(pruned_heads)
        m.prune_heads(pruned_heads, early_attn_mask_records[layer_id])
    return model

def prune_bert_ffn(model, early_ffn_mask_records, use_pruning_step, pruning_ratio=0.4, pruning_method="layerwise"):
    print("Pruning FFN Ratio:{}".format(pruning_ratio))
    layers = []
    for m in model.modules():
        if isinstance(m, AUTO_MODULES[model.config.model_type]["layer"]):
            layers.append(m)
    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_inter_neurons)
    if use_pruning_step > 0:
        early_ffn_mask_records = early_ffn_mask_records[:, use_pruning_step-1, :]
    else:
        # Random pruning
        # Get internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-use_pruning_step + 1)
        early_ffn_mask_records = np.random.rand(
            len(layers), model.config.intermediate_size)
        # Reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `early_ffn_mask_records`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `early_ffn_mask_records`.
    quantile_axis = -1 if pruning_method == 'layerwise' else None
    threshold = np.quantile(early_ffn_mask_records, pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = early_ffn_mask_records > threshold
    for layer_id, (m, mask) in enumerate(zip(layers, layers_masks)):
        pruned_inter_neurons = [i for i in range(len(mask)) if mask[i] == 0]
        #logger.info('{} neurons are pruned'.format(len(pruned_inter_neurons)))
        m.prune_inter_neurons(pruned_inter_neurons, early_ffn_mask_records[layer_id])
    return model

def prune_lora_attn(adapters_weights, early_attn_mask_records, use_pruning_step, pruning_ratio=0.4, pruning_method="layerwise", config=None):
    print("Pruning PEFT on Attention Ratio:{}".format(pruning_ratio))
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    def prune_lora_by_heads(heads, weight, dim=0):
        pruned_heads = set()
        if len(heads) == 0:
            return weight
        heads = set(heads) - pruned_heads  # Convert to set and remove already pruned heads
        prune_head_flags = torch.ones(num_heads, head_dim)    
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in pruned_heads)
            prune_head_flags[head] = 0
        prune_head_flags = prune_head_flags.view(-1).contiguous().eq(1)
        prune_head_index = torch.arange(len(prune_head_flags))[prune_head_flags].long()
        # Prune linear layers
        prune_head_index = prune_head_index.to(weight.device)
        W = weight.index_select(dim, prune_head_index).clone().detach()
    
        return W
        
    if use_pruning_step > 0:
        early_attn_mask_records = early_attn_mask_records[:, use_pruning_step-1, :]
    else:
        # Random pruning
        # Ret internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-use_pruning_step)
        early_attn_mask_records = np.random.rand(config.num_hidden_layers, config.num_attention_heads)
        # Reset internal state
        np.random.set_state(rand_state)
    quantile_axis = -1 if pruning_method == 'layerwise' else None
    threshold = np.quantile(early_attn_mask_records, pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = early_attn_mask_records > threshold
    for param_name in adapters_weights:
        if "lora_B" in param_name and "attention" in param_name and "output" not in param_name:
            name_list = param_name.split('.')
            layer_id = int(name_list[name_list.index("attention")-1])
            mask = layers_masks[layer_id]
            need_pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
            adapters_weights[param_name] = prune_lora_by_heads(need_pruned_heads, adapters_weights[param_name])
        if "lora_A" in param_name and "attention" in param_name and "output" in param_name:
            name_list = param_name.split('.')
            layer_id = int(name_list[name_list.index("attention")-1])
            mask = layers_masks[layer_id]
            need_pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
            adapters_weights[param_name] = prune_lora_by_heads(need_pruned_heads, adapters_weights[param_name], dim=1)
    return adapters_weights

def prune_lora_ffn(adapters_weights, early_ffn_mask_records, use_pruning_step, pruning_ratio=0.4, pruning_method="layerwise", config=None):
    print("Pruning PEFT on FFN Ratio:{}".format(pruning_ratio))
    fc1_out_features = config.intermediate_size
    def prune_lora_by_inter_neurons(neurons, weight, dim=0):
        pruned_inter_neurons = set()
        if len(neurons) == 0:
            return weight
        prune_ffn_flags = torch.ones(fc1_out_features)
        neurons = set(neurons) - pruned_inter_neurons  # Convert to set and remove already pruned neurons

        for neuron in neurons:
            # Compute how many pruned neurons are before the neuron and move the index accordingly
            neuron = neuron - sum(1 if n < neuron else 0 for n in pruned_inter_neurons)
            prune_ffn_flags[neuron] = 0
        prune_ffn_flags = prune_ffn_flags.view(-1).contiguous().eq(1)
        prune_ffn_index = torch.arange(len(prune_ffn_flags))[prune_ffn_flags].long()
        # Prune linear layers
        prune_ffn_index = prune_ffn_index.to(weight.device)
        W = weight.index_select(dim, prune_ffn_index).clone().detach()
    
        return W
        
    if use_pruning_step > 0:
        early_ffn_mask_records = early_ffn_mask_records[:, use_pruning_step-1, :]
    else:
        # Random pruning
        # Get internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-use_pruning_step + 1)
        early_ffn_mask_records = np.random.rand(config.num_hidden_layers, config.ffn_dim)
        # Reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `early_ffn_mask_records`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `early_ffn_mask_records`.
    quantile_axis = -1 if pruning_method == 'layerwise' else None
    threshold = np.quantile(early_ffn_mask_records, pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = early_ffn_mask_records > threshold
    for param_name in adapters_weights:
        if "lora_B" in param_name and "intermediate" in param_name:
            name_list = param_name.split('.')
            layer_id = int(name_list[name_list.index("intermediate")-1])
            mask = layers_masks[layer_id]
            need_pruned_inter_neurons = [i for i in range(len(mask)) if mask[i] == 0]
            adapters_weights[param_name] = prune_lora_by_inter_neurons(need_pruned_inter_neurons, adapters_weights[param_name])
        if "lora_A" in param_name and "output" in param_name and "attention" not in param_name:
            name_list = param_name.split('.')
            layer_id = int(name_list[name_list.index("output")-1])
            mask = layers_masks[layer_id]
            need_pruned_inter_neurons = [i for i in range(len(mask)) if mask[i] == 0]
            adapters_weights[param_name] = prune_lora_by_inter_neurons(need_pruned_inter_neurons, adapters_weights[param_name], dim=1)
    return adapters_weights

def prune_peft_rank(adapters_weights, early_peft_mask_records, use_pruning_step, pruning_ratio=0.4, pruning_method="layerwise", peft_config=None, model=None):
    print("Pruning PEFT on Rank Ratio:{}".format(pruning_ratio))
    if hasattr(peft_config,"r"):
        rank_features = peft_config.r
    else:
        rank_features = peft_config.bottleneck_size
    def prune_lora_by_inter_neurons(neurons, module, dim=0):
        pruned_inter_neurons = set()
        if len(neurons) == 0:
            return module
        prune_rank_flags = torch.ones(rank_features)
        neurons = set(neurons) - pruned_inter_neurons  # Convert to set and remove already pruned neurons

        for neuron in neurons:
            # Compute how many pruned neurons are before the neuron and move the index accordingly
            neuron = neuron - sum(1 if n < neuron else 0 for n in pruned_inter_neurons)
            prune_rank_flags[neuron] = 0
        prune_rank_flags = prune_rank_flags.view(-1).contiguous().eq(1)
        prune_rank_index = torch.arange(len(prune_rank_flags))[prune_rank_flags].long()
        # Prune linear layers
        module = prune_linear_layer(module, prune_rank_index, dim=dim)

        return module
        
    if use_pruning_step > 0:
        pass
    else:
        # Random pruning
        # Get internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-use_pruning_step + 1)
        num_peft_modules = sum('lora_A' in key for key in adapters_weights)
        early_peft_mask_records = np.random.rand(num_peft_modules, rank_features)
        # Reset internal state
        np.random.set_state(rand_state)
    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `early_peft_mask_records`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `early_peft_mask_records`.
    quantile_axis = -1 if pruning_method == 'layerwise' else None
    early_peft_mask_records = {key: early_peft_mask_records[key] for key in early_peft_mask_records if any(key in sub for sub in adapters_weights)}
    early_peft_mask_records_values=np.array(list(early_peft_mask_records.values()))
    threshold = np.quantile(early_peft_mask_records_values, pruning_ratio, axis=quantile_axis, keepdims=True)
    rank_masks = early_peft_mask_records_values > threshold
    early_peft_mask_records_masks = {key: rank_masks[index] for key, index in zip(early_peft_mask_records.keys(), range(len(early_peft_mask_records)))}
    total_modules = 0
    removed_peft_modules = 0
    for param_name,m in model.named_modules():
        if hasattr(m,"lora_A"):
            total_modules+=1
            mask = early_peft_mask_records_masks[param_name]
            need_pruned_inter_neurons = [i for i in range(len(mask)) if mask[i] == 0]
            if len(need_pruned_inter_neurons)>=rank_features:
                print(param_name)
                m.r=0
                m.lora_A=nn.Linear(0,0)
                m.lora_B=nn.Linear(0,0)
                m.lora_A.requires_grad=False
                m.lora_B.requires_grad=False
                removed_peft_modules+=1
            else:
                m.lora_A = prune_lora_by_inter_neurons(need_pruned_inter_neurons, m.lora_A)
                m.lora_B = prune_lora_by_inter_neurons(need_pruned_inter_neurons, m.lora_B, dim=1)
        elif hasattr(m,"adapter_down"):
            total_modules+=1
            mask = early_peft_mask_records_masks[param_name]
            need_pruned_inter_neurons = [i for i in range(len(mask)) if mask[i] == 0]
            if len(need_pruned_inter_neurons)>=rank_features:
                print(param_name)
                m.bottleneck_size=0
                m.adapter_down=nn.Linear(0,0)
                m.adapter_up=nn.Linear(0,0)
                m.adapter_down.requires_grad=False
                m.adapter_up.requires_grad=False
                m.disable_adapters=True
                removed_peft_modules+=1
            else:
                m.adapter_down = prune_lora_by_inter_neurons(need_pruned_inter_neurons, m.adapter_down)
                m.adapter_up = prune_lora_by_inter_neurons(need_pruned_inter_neurons, m.adapter_up, dim=1)
    print((removed_peft_modules/total_modules)*100,"percent modules has been removed.")
    return model

def prune_peft_modules(adapters_weights, pruning_ratio=0.4, imp_data=None, random_shuffle=False):
    print("Pruning PEFT on Module Ratio:{}".format(pruning_ratio))
    imp_data_values = np.array(list(imp_data.values()))
    threshold = np.quantile(imp_data_values, pruning_ratio, keepdims=True)
    modules_masks = imp_data_values > threshold
    imp_data_masks = {key: modules_masks[index] for key, index in zip(imp_data.keys(), range(len(imp_data)))}
    result_dict = {key: adapters_weights[key] for key in adapters_weights if any(sub in key for sub in imp_data_masks if imp_data_masks[sub]) or (key not in imp_data_masks and all(sub not in key for sub in imp_data_masks))}

    return list(result_dict.keys())

def get_early_model(model_args, training_args, task_type, config):
    model_class = AUTO_MODELS[config.model_type][task_type]
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            device_map=device_map
    )
    early_attn_mask_records = np.load(training_args.peft_dir+"/early_attn_mask_records.npy")
    early_ffn_mask_records = np.load(training_args.peft_dir+"/early_ffn_mask_records.npy")
    if training_args.random_pruning:
        print("Use random pruning.")
        use_pruning_step = -1
    else:
        print("Use mask pruning.")
        use_pruning_step = len(early_attn_mask_records[0])
    if training_args.attn_pruning_ratio!=0:
        model = prune_bert_attn(model, early_attn_mask_records, use_pruning_step=use_pruning_step, pruning_ratio=training_args.attn_pruning_ratio)
    if training_args.ffn_pruning_ratio!=0:
        model = prune_bert_ffn(model, early_ffn_mask_records, use_pruning_step=use_pruning_step, pruning_ratio=training_args.ffn_pruning_ratio, pruning_method=training_args.ffn_pruning_method)

    print("-------------------------------Model pruning is complete.-------------------------------")
    return model, config, early_attn_mask_records, early_ffn_mask_records

def get_early_peft_model(model, training_args, lora_args, bottleneck_args, task_type, original_config, early_attn_mask_records, early_ffn_mask_records):
    # Load adapter weights
    adapters_weights = torch.load(
            training_args.peft_dir+"/adapter_model.bin", map_location=model.device
        )
    # Random Pruning?
    if training_args.random_pruning:
        print("Use random pruning.")
        use_pruning_step = -1
    else:
        print("Use mask pruning.")
        use_pruning_step = len(early_attn_mask_records[0])
    modules_pruning = True
    ranks_pruning = True
    if training_args.module_pruning_ratio==0:
        modules_pruning=False
    if training_args.rank_pruning_ratio==0:
        ranks_pruning=False
    # Calculate reserved peft modules
    if modules_pruning:
        with open(training_args.peft_dir+"/peft_module_records.pkl", "rb") as f:
            imp_data = pickle.load(f)
        reserved_params = prune_peft_modules(adapters_weights, pruning_ratio=training_args.module_pruning_ratio, imp_data=imp_data, random_shuffle=training_args.random_peft)

    # Load peft config
    peft_config = get_peft_config(training_args, lora_args, bottleneck_args, task_type)
    # Prune LoRA features weights
    if training_args.attn_pruning_ratio!=0:
        adapters_weights = prune_lora_attn(adapters_weights=adapters_weights, 
                                            early_attn_mask_records=early_attn_mask_records, 
                                            use_pruning_step=use_pruning_step, 
                                            pruning_ratio=training_args.attn_pruning_ratio, 
                                            config=original_config)
    if training_args.ffn_pruning_ratio!=0:
        adapters_weights = prune_lora_ffn(adapters_weights=adapters_weights, 
                                            early_ffn_mask_records=early_ffn_mask_records, 
                                            use_pruning_step=use_pruning_step, 
                                            pruning_ratio=training_args.ffn_pruning_ratio,
                                            pruning_method=training_args.ffn_pruning_method,
                                            config=original_config)
    # Get reserved peft modules
    if modules_pruning:
        adapters_weights = {key: value for key, value in adapters_weights.items() if key in reserved_params}
        peft_config.reserved_modules = reserved_params
        peft_config.prune_module = True
    # Get peft model
    model = get_peft_model(model, peft_config)
    model = set_peft_model_state_dict(model, adapters_weights)
    # Prune peft rank weights
    if ranks_pruning:
        with open(training_args.peft_dir+"/peft_mask_records.pkl", "rb") as f:
            early_peft_mask_records = pickle.load(f)
        model =  prune_peft_rank(adapters_weights=adapters_weights, 
                                        early_peft_mask_records=early_peft_mask_records, 
                                        use_pruning_step=use_pruning_step,
                                        pruning_method=training_args.peft_pruning_method, 
                                        pruning_ratio=training_args.rank_pruning_ratio, 
                                        peft_config=peft_config,model=model)
    print("-------------------------------Peft Model pruning is complete.-------------------------------")
    return model

def get_target_module(model_type, module):
    return AUTO_MODULES[model_type][module]

def count_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    
    for param_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad and "classifier" not in param_name:
            trainable_params += num_params
            print(param_name,str(num_params))
        if hasattr(param,"lora_A"):
            total_peft_modules+=1
            if param.r==0:
                removed_peft_modules+=1
        elif hasattr(param,"adapter_up"):
            total_peft_modules+=1
            if param.bottleneck_size==0:
                removed_peft_modules+=1

    return all_param/(10**6),trainable_params/(10**6)

def count_removed_peft_modules(model):
    total_peft_modules = 0
    removed_peft_modules = 0
    for param_name,m in model.named_modules():
        if hasattr(m,"lora_A"):
            total_peft_modules+=1
            if m.r==0:
                removed_peft_modules+=1
        elif hasattr(m,"adapter_up"):
            total_peft_modules+=1
            if m.bottleneck_size==0:
                removed_peft_modules+=1
    removed_peft_proportion=removed_peft_modules/total_peft_modules
    print(removed_peft_proportion*100,"percent modules has been romoved.")
    return total_peft_modules, removed_peft_modules, removed_peft_proportion