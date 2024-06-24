export CUDA_VISIBLE_DEVICES=2

TASK_NAME=glue
DATASET_NAME=sst2
model_name_or_path=roberta-large
bs=32
lr=3e-4
seq_len=128
epoch=5
max_steps=800
adapter_name="lora" #or "bottleneck"
rank=8
lora_alpha=16
peft_imp="p2-first_order"
l1_loss_coef=1e-4

output_dir=results/${DATASET_NAME}/${model_name_or_path}-${adapter_name}-${peft_imp}
python NLU/estimate.py \
  --adapter_name  ${adapter_name} \
  --lora_r ${rank} --lora_alpha ${lora_alpha} --lora_dropout 0.05 --lora_target_modules "query" "value" "key" "output.dense" "intermediate.dense" \
  --bottleneck_size ${rank} --adapter_dropout 0.1 --non_linearity "relu" --target_modules "output.dense" \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${TASK_NAME} \
  --dataset_name ${DATASET_NAME} \
  --do_train \
  --do_eval \
  --pad_to_max_length True \
  --max_seq_length ${seq_len} \
  --per_device_train_batch_size ${bs} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --max_steps ${max_steps} \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --seed 42 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --early_search True --use_mask_trainer True --use_peft True \
  --peft_imp ${peft_imp} \
  --l1_loss_coef ${l1_loss_coef}


