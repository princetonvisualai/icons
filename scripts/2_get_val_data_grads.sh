#!/bin/bash

read -p "Please enter the model path: " model
read -p "Please enter the data directory path [default: /n/fs/visualai-scr/Data/llava/eval]: " data_path
data_path=${data_path:-"/n/fs/visualai-scr/Data/llava/eval"}

dims='5120'
current_date=$(date +"%m%d")

declare -A task_configs=(
    ["llavabench_in_the_wild"]="${data_path}/llava-bench-in-the-wild/conversations/conversations.json|${data_path}/llava-bench-in-the-wild/images/"
    ["gqa"]="${data_path}/gqa/conversations.json|${data_path}/gqa/images"
    ["vqav2"]="${data_path}/vqav2/conversations.json|${data_path}/vqav2/images"
    ["vizwiz"]="${data_path}/vizwiz/conversations.json|${data_path}/vizwiz/images"
    ["textvqa"]="${data_path}/textvqa/conversations.json|${data_path}/textvqa/images"
    ["sqa"]="${data_path}/scienceqa/conversations.json|${data_path}/scienceqa/images"
    ["pope"]="${data_path}/pope/conversations.json|${data_path}/pope/images"
    ["mmbench"]="${data_path}/mmbench/conversations/mmbench_dev_en.json|${data_path}/mmbench/images"
    ["mmbench_cn"]="${data_path}/mmbench/conversations/mmbench_dev_cn.json|${data_path}/mmbench/images"
)

for task in "${!task_configs[@]}"; do
    IFS='|' read -r val_file image_folder <<< "${task_configs[$task]}"
    base_output_path="./output/val_gradient/${current_date}/${task}"
    
    echo "Processing task: $task"
    
    if [[ ! -d $base_output_path ]]; then
        mkdir -p $base_output_path
    fi

    CUDA_VISIBLE_DEVICES=0 python3 icons/obtain_info.py \
        --train_file "$val_file" \
        --task "$task" \
        --info_type grads \
        --model_path "$model" \
        --output_path "$base_output_path" \
        --gradient_projection_dimension "$dims" \
        --gradient_type sgd \
        --image_folder "$image_folder"
done


echo "Processing MME tasks..."
base_output_path="./output/val_gradient/${current_date}/mme"
base_val_path="${data_path}/MME/target_task/val_set"
mme_image_folder="${data_path}/MME/MME_Benchmark_release_version"


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

    python3 icons/obtain_info.py \
        --train_file "$val_file" \
        --task "$task" \
        --info_type grads \
        --model_path "$model" \
        --output_path "$output_path" \
        --gradient_type sgd \
        --image_folder "$mme_image_folder" \
        --gradient_projection_dimension "$dims"
done 