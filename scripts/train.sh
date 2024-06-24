export CUDA_VISIBLE_DEVICES=2

TASK_NAME=glue
DATASET_NAME=sst2
model_name_or_path=roberta-large
bs=32
lr=3e-4
epoch=10
seq_len=128
adapter_name="lora"
rank=8
lora_alpha=16
peft_imp="p2-first_order"

attn_pruning_ratio=0.333333333
ffn_pruning_ratio=0.333333333
module_pruning_ratio=0.75
rank_pruning_ratio=0.5
peft_dir=results/${DATASET_NAME}/${model_name_or_path}-${adapter_name}-${peft_imp}
output_dir=${peft_dir}/second_train/ratio_${attn_pruning_ratio}-${ffn_pruning_ratio}-lr${lr}-${module_pruning_ratio}_${rank_pruning_ratio}
python NLU/train.py \
  --adapter_name  ${adapter_name} \
  --lora_r ${rank} --lora_alpha ${lora_alpha} --lora_dropout 0.05 --lora_target_modules "query" "value" "key" "output.dense" "intermediate.dense" \
  --bottleneck_size ${rank} --adapter_dropout 0.1 --non_linearity "relu" --target_modules "output.dense" \
  --attn_pruning_ratio ${attn_pruning_ratio} --ffn_pruning_ratio ${ffn_pruning_ratio} \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${TASK_NAME} \
  --dataset_name ${DATASET_NAME} \
  --do_train \
  --pad_to_max_length True \
  --do_eval \
  --overwrite_output_dir \
  --max_seq_length ${seq_len} \
  --per_device_train_batch_size ${bs} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --output_dir ${output_dir} \
  --seed 42 \
  --peft_dir ${peft_dir} \
  --save_strategy no \
  --evaluation_strategy epoch \
  --draw_and_training True --use_peft True \
  --ffn_pruning_method "global" \
  --module_pruning_ratio ${module_pruning_ratio} \
  --rank_pruning_ratio ${rank_pruning_ratio} --peft_pruning_method "global" 