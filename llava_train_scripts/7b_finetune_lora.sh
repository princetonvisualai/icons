#!/bin/bash

# Prompt user for data_path
read -p "Enter the data_path (press Enter for default: /n/fs/visualai-scr/Data/llava/download/llava-v1.5-instruct/llava_v1_5_mix665k.json): " user_data_path
data_path=${user_data_path:-"/n/fs/visualai-scr/Data/llava/download/llava-v1.5-instruct/llava_v1_5_mix665k.json"}

# Prompt user for output_dir
read -p "Enter the output directory name (will be appended to the base path): " user_output_dir
read -p "Is the current directory full? (y/n): " is_full

if [ -z "$user_output_dir" ]; then
    output_dir="./checkpoints/defaults/"
else
    if [ "$is_full" = "y" ] || [ "$is_full" = "Y" ]; then
        output_dir="/n/fs/visualai-scr/Models/xindiw/checkpoints/important/llava_lora_${user_output_dir}"
    else
        output_dir="./checkpoints/important/llava_lora_${user_output_dir}"
    fi
fi

echo "Output directory set to: $output_dir"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./llava_train_scripts/zero3.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path "$data_path" \
    --image_folder /n/fs/visualai-scr/Data/llava/download/llava-v1.5-instruct \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$output_dir" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 50 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# Tips:
# - You can adjust --save_steps (currently 250) to control checkpoint frequency
# - You can adjust --save_total_limit (currently 50) to control max checkpoints saved
