#!/bin/bash


checkpoint_number=${1:-500}  

model=./checkpoints/llava_lora_full/checkpoint-$checkpoint_number
echo "Model path: $model"
dims='5120'
output_path=./output/train_gradient/${dims}_train_gradient_llava_665k_lora_checkpoint_$checkpoint_number 
gradient_type="sgd"
image_folder=./Data/llava/download/llava-v1.5-instruct
split_dir=./data/665k_split_200


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

task_id=$SLURM_ARRAY_TASK_ID

# Compute the file index based on the SLURM task ID
file_index=$(printf "%d" $((task_id + 1)))
train_file="chunk_${file_index}.json"

if [[ -f "$split_dir/$train_file" ]]; then
    echo "Processing file: $train_file"

    CUDA_VISIBLE_DEVICES=0 python3 icons/obtain_info.py \
        --train_file "$split_dir/$train_file" \
        --info_type grads \
        --model_path "$model" \
        --output_path "${output_path}/$(basename $train_file .json)" \
        --gradient_type $gradient_type \
        --image_folder $image_folder \
        --gradient_projection_dimension $dims 
else
    echo "File $train_file does not exist. Skipping..."
fi

