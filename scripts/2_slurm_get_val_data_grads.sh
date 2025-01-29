#!/bin/bash

#SBATCH --job-name=icons_val_grads
#SBATCH --output=./slurm/val_gradient/output_%A_%a.out
#SBATCH --error=./slurm/val_gradient/error_%A_%a.err
#SBATCH --array=0-21     # Total 22 tasks (9 regular + 13 MME tasks)
#SBATCH --gres=gpu:l40:1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBA

source xxxx/anaconda3/etc/profile.d/conda.sh 
conda activate icons


if [ -z "$1" ]; then
    echo "Please provide checkpoint number as argument"
    exit 1
fi

checkpoint=$1
model="./checkpoints/llava_lora_full/checkpoint-$checkpoint"
dims='5120'
current_date=$(date +"%m%d")


declare -a all_tasks=(
    # Regular tasks
    "llavabench_in_the_wild|./Data/llava/eval/llava-bench-in-the-wild/conversations/conversations.json|./Data/llava/eval/llava-bench-in-the-wild/images/"
    "gqa|./Data/llava/eval/gqa/conversations.json|./Data/llava/eval/gqa/images"
    "vqav2|./Data/llava/eval/vqav2/conversations.json|./Data/llava/eval/vqav2/images"
    "vizwiz|./Data/llava/eval/vizwiz/conversations.json|./Data/llava/eval/vizwiz/images"
    "textvqa|./Data/llava/eval/textvqa/conversations.json|./Data/llava/eval/textvqa/images"
    "sqa|./Data/llava/eval/scienceqa/conversations.json|./Data/llava/eval/scienceqa/images"
    "pope|./Data/llava/eval/pope/conversations.json|./Data/llava/eval/pope/images"
    "mmbench|./Data/llava/eval/mmbench/conversations/mmbench_dev_en.json|./Data/llava/eval/mmbench/images"
    "mmbench_cn|./Data/llava/eval/mmbench/conversations/mmbench_dev_cn.json|./Data/llava/eval/mmbench/images"
    # MME tasks
    "mme_landmark|./Data/llava/eval/MME/target_task/val_set/landmark.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_celebrity|./Data/llava/eval/MME/target_task/val_set/celebrity.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_posters|./Data/llava/eval/MME/target_task/val_set/posters.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_scene|./Data/llava/eval/MME/target_task/val_set/scene.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_code_reasoning|./Data/llava/eval/MME/target_task/val_set/code_reasoning.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_color|./Data/llava/eval/MME/target_task/val_set/color.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_commonsense_reasoning|./Data/llava/eval/MME/target_task/val_set/commonsense_reasoning.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_count|./Data/llava/eval/MME/target_task/val_set/count.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_existence|./Data/llava/eval/MME/target_task/val_set/existence.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_OCR|./Data/llava/eval/MME/target_task/val_set/OCR.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_numerical_calculation|./Data/llava/eval/MME/target_task/val_set/numerical_calculation.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_position|./Data/llava/eval/MME/target_task/val_set/position.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
    "mme_text_translation|./Data/llava/eval/MME/target_task/val_set/text_translation.json|./Data/llava/eval/MME/MME_Benchmark_release_version"
)


current_task="${all_tasks[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r task val_file image_folder <<< "$current_task"


if [[ $task == mme_* ]]; then
    # For MME tasks
    task_name=${task#mme_}  # Remove 'mme_' prefix
    base_output_path="./output/val_gradient/mme-${current_date}-ckpt${checkpoint}/${task_name}"
else
    base_output_path="./output/val_gradient/${task}-${current_date}-ckpt${checkpoint}"
fi


mkdir -p "$base_output_path"


CUDA_VISIBLE_DEVICES=0 python3 icons/obtain_info.py \
    --train_file "$val_file" \
    --task "$task" \
    --info_type grads \
    --model_path "$model" \
    --output_path "$base_output_path" \
    --gradient_projection_dimension "$dims" \
    --gradient_type sgd \
    --image_folder "$image_folder"