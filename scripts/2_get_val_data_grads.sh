#!/bin/bash


read -p "Which checkpoint do you want to use? (100, 200, ..., 1000): " checkpoint
model="./checkpoints/llava_lora_full/checkpoint-$checkpoint"


read -p "Is this the path you wanted? $model (y/n): " confirm_model
if [[ $confirm_model != "y" ]]; then
    read -p "Please enter the correct model path: " model
fi

dims='5120'
current_date=$(date +"%m%d")

declare -A task_configs=(
    ["llavabench_in_the_wild"]="./Data/llava/eval/llava-bench-in-the-wild/conversations/conversations.json|./Data/llava/eval/llava-bench-in-the-wild/images/"
    ["gqa"]="./Data/llava/eval/gqa/conversations.json|./Data/llava/eval/gqa/images"
    ["vqav2"]="./Data/llava/eval/vqav2/conversations.json|./Data/llava/eval/vqav2/images"
    ["vizwiz"]="./Data/llava/eval/vizwiz/conversations.json|./Data/llava/eval/vizwiz/images"
    ["textvqa"]="./Data/llava/eval/textvqa/conversations.json|./Data/llava/eval/textvqa/images"
    ["sqa"]="./Data/llava/eval/scienceqa/conversations.json|./Data/llava/eval/scienceqa/images"
    ["pope"]="./Data/llava/eval/pope/conversations.json|./Data/llava/eval/pope/images"
    ["mmbench"]="./Data/llava/eval/mmbench/conversations/mmbench_dev_en.json|./Data/llava/eval/mmbench/images"
    ["mmbench_cn"]="./Data/llava/eval/mmbench/conversations/mmbench_dev_cn.json|./Data/llava/eval/mmbench/images"
)

for task in "${!task_configs[@]}"; do
    IFS='|' read -r val_file image_folder <<< "${task_configs[$task]}"
    base_output_path="./output/val_gradient/${task}-${current_date}-ckpt${checkpoint}"
    
    echo "Processing task: $task"
    
    if [[ ! -d $base_output_path ]]; then
        mkdir -p $base_output_path
    fi

    CUDA_VISIBLE_DEVICES=0 python3 icons/get_info.py \
        --train_file "$val_file" \
        --task "$task" \
        --info_type grads \
        --model_path "$model" \
        --output_path "$base_output_path" \
        --gradient_projection_dimension "$dims" \
        --gradient_type sgd \
        --image_folder "$image_folder"
done

# Special handling for MME
echo "Processing MME tasks..."
base_output_path="./output/val_gradient/mme-${current_date}-ckpt${checkpoint}"
base_val_path="./Data/llava/eval/MME/target_task/val_set"
mme_image_folder="./Data/llava/eval/MME/MME_Benchmark_release_version"

# Array of MME validation files and corresponding tasks
declare -A mme_tasks=(
    ["landmark"]="landmark"
    ["celebrity"]="celebrity"
    ["posters"]="posters"
    ["scene"]="scene"
    ["code_reasoning"]="code_reasoning"
    ["color"]="color"
    ["commonsense_reasoning"]="commonsense_reasoning"
    ["count"]="count"
    ["existence"]="existence"
    ["OCR"]="OCR"
    ["numerical_calculation"]="numerical_calculation"
    ["position"]="position"
    ["text_translation"]="text_translation"
)

for val_file_name in "${!mme_tasks[@]}"; do
    task=${mme_tasks[$val_file_name]}
    val_file="${base_val_path}/${val_file_name}.json"
    output_path="${base_output_path}/${task}"

    echo "Processing MME subtask: $task"
    
    if [[ ! -d $output_path ]]; then
        mkdir -p $output_path
    fi

    python3 icons/get_info.py \
        --train_file "$val_file" \
        --task "$task" \
        --info_type grads \
        --model_path "$model" \
        --output_path "$output_path" \
        --gradient_type sgd \
        --image_folder "$mme_image_folder" \
        --gradient_projection_dimension "$dims"
done 