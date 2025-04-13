USERNAME=""

MODEL_TYPE='llama2-7b'

DATASET=multi_role_train
MAXSAMPLE=1000

MODEL_PATH="/home/${USERNAME}/base-model/${MODEL_TYPE}"
SFT_CHECKPOINT="sft_checkpoint/${MODEL_TYPE}-${DATASET}-${MAXSAMPLE}"

# --evaluation_strategy steps \
# --val_size 0.05 \
# --per_device_eval_batch_size 1 \
# --max_samples 1000\

nohup deepspeed --include localhost:0,1,2,3 fairllm/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${DATASET} \
    --template default \
    --finetuning_type lora \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_rank 8 \
    --lora_target q_proj,v_proj \
    --output_dir ${SFT_CHECKPOINT} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_steps 200 \
    --max_length 1024 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --max_samples ${MAXSAMPLE} \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --report_to='wandb'&