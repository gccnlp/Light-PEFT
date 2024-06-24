export CUDA_VISIBLE_DEVICES=0

TASK_NAME=glue
DATASET_NAME=mnli
model_name_or_path=roberta-large
bs=32
lr=3e-4
seq_len=128
epoch=5
rank=8
adapter_name="lora"

output_dir=results/${DATASET_NAME}/${model_name_or_path}-${adapter_name}-original
python NLU/vanilla_train.py \
  --adapter_name  ${adapter_name} \
  --lora_r ${rank} --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules "query" "value" \
  --bottleneck_size ${rank} --adapter_dropout 0.1 --non_linearity "relu" --target_modules "output.dense" \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${TASK_NAME} \
  --dataset_name ${DATASET_NAME} \
  --do_train \
  --pad_to_max_length True \
  --do_eval \
  --max_seq_length ${seq_len} \
  --per_device_train_batch_size ${bs} \
  --learning_rate ${lr} \
  --num_train_epochs $epoch \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --seed 42 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --use_peft True
