USERNAME=""

DATASET=multi_role
MODEL_TYPE='llama2-7b'

LR=3e-4
WARMUP=200

MODEL_PATH="/home/${USERNAME}/base-model/${MODEL_TYPE}"
RM_CHECKPOINT="rm_checkpoint/${MODEL_TYPE}-${DATASET}-lr=${LR}-warmup=${WARMUP}"

nohup deepspeed --include localhost:0,1,2,3 fairllm/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --create_new_adapter \
    --dataset ${DATASET} \
    --val_size 0.05 \
    --template default \
    --finetuning_type lora \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_rank 8 \
    --lora_target q_proj,v_proj \
    --output_dir ${RM_CHECKPOINT} \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type linear \
    --warmup_steps ${WARMUP} \
    --max_length 512 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${LR} \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16 \
    --report_to='wandb'&