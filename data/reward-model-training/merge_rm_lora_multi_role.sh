USERNAME=""

MODEL_TYPE='llama2-7b'
MODEL_PATH="/home/${USERNAME}/base-model/${MODEL_TYPE}"
RM_CHECKPOINT="rm_checkpoint/${MODEL_TYPE}-multi_role-lr=3e-4-warmup=200"
EXPORT_MODEL="/home/${USERNAME}/export-model/${MODEL_TYPE}-rm-multi_role"

CUDA_VISIBLE_DEVICES=0 python fairllm/export_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${RM_CHECKPOINT} \
    --template default \
    --finetuning_type lora \
    --export_dir ${EXPORT_MODEL}\
    --export_size 2 \
    --export_legacy_format False