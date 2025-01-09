#!/bin/bash

declare -A TASK_CONFIGS=(
    ["gqa"]="gqa"
    ["llavabench_in_the_wild"]="llavabench_in_the_wild"
    ["mmbench_cn"]="mmbench_cn"
    ["mmbench"]="mmbench"
    ["pope"]="pope"
    ["sqa"]="sqa"
    ["textvqa"]="textvqa"
    ["vizwiz"]="vizwiz"
    ["vqav2"]="vqav2"
)

# MME subtasks
declare -a MME_TASKS=(
    "celebrity"
    "posters"
    "scene"
    "code_reasoning"
    "color"
    "commonsense_reasoning"
    "count"
    "existence"
    "OCR"
    "numerical_calculation"
    "position"
    "text_translation"
    "landmark"
)

# Prompt user to select tasks
echo "Available tasks:"
echo "1. All regular tasks"
echo "2. MME tasks only"
echo "3. Specific tasks"
read -p "Select option (1/2/3): " TASK_OPTION

if [[ $TASK_OPTION == "3" ]]; then
    echo "Available regular tasks:"
    for task in "${!TASK_CONFIGS[@]}"; do
        echo "- $task"
    done
    echo "- mme (for all MME tasks)"
    
    read -p "Enter task names separated by spaces: " -a SELECTED_TASKS
else
    if [[ $TASK_OPTION == "1" ]]; then
        SELECTED_TASKS=("${!TASK_CONFIGS[@]}")
    else
        SELECTED_TASKS=("mme")
    fi
fi


read -p "Enter checkpoint number: " ckpt_number


train_gradient_base="./output/train_gradient"
matching_train_path=$(find "$train_gradient_base" -maxdepth 1 -type d -name "*checkpoint_${ckpt_number}" | head -n 1)

if [ -z "$matching_train_path" ]; then
    echo "No matching train gradient path found for checkpoint ${ckpt_number}"
    exit 1
fi

echo "Found matching train gradient path: $matching_train_path"
read -p "Is this correct? (y/n): " confirm

if [[ $confirm != [Yy]* ]]; then
    echo "Aborted. Please run the script again with the correct checkpoint number."
    exit 1
fi

train_gradient_path="${matching_train_path}/everything_all_normalized.pt"
current_date=$(date +"%m%d")
val_gradient_base="./output/val_gradient"

# Process each selected task
for task in "${SELECTED_TASKS[@]}"; do
    if [[ $task == "mme" ]]; then
        # Handle MME tasks
        matching_val_paths=($(find "$val_gradient_base" -maxdepth 1 -type d -name "*ckpt${ckpt_number}*"))
        
        if [ ${#matching_val_paths[@]} -gt 0 ]; then
            if [ ${#matching_val_paths[@]} -eq 1 ]; then
                validation_gradient_base_path="${matching_val_paths[0]}/"
            else
                echo "Multiple matching validation gradient paths found for MME:"
                for i in "${!matching_val_paths[@]}"; do
                    echo "[$i] ${matching_val_paths[$i]}"
                done
                read -p "Enter the number of the correct path: " choice
                validation_gradient_base_path="${matching_val_paths[$choice]}/"
            fi

            influence_score_base="./output/influence_score/${current_date}/mme-${current_date}-ckpt-${ckpt_number}/"

            for sub_task in "${MME_TASKS[@]}"; do
                validation_gradient_path="${validation_gradient_base_path}${sub_task}/dim5120/all_unormalized.pt"
                influence_score="${influence_score_base}${sub_task}/"
                
                if [[ ! -d $influence_score ]]; then
                    mkdir -p $influence_score
                fi

                python3 ./icons/influence_matrix.py \
                    --train_gradient_path $train_gradient_path \
                    --validation_gradient_path $validation_gradient_path \
                    --influence_score $influence_score \
                    --train_file_name $sub_task
            done
        fi
    else
        # Handle regular tasks
        matching_val_paths=($(find "$val_gradient_base" -maxdepth 1 -type d -name "${task}-*ckpt${ckpt_number}*"))
        
        if [ ${#matching_val_paths[@]} -gt 0 ]; then
            if [ ${#matching_val_paths[@]} -eq 1 ]; then
                validation_gradient_path="${matching_val_paths[0]}/dim5120/all_normalized.pt"
            else
                echo "Multiple matching validation gradient paths found for ${task}:"
                for i in "${!matching_val_paths[@]}"; do
                    echo "[$i] ${matching_val_paths[$i]}/dim5120/all_normalized.pt"
                done
                read -p "Enter the number of the correct path: " choice
                validation_gradient_path="${matching_val_paths[$choice]}/dim5120/all_normalized.pt"
            fi

            influence_score_base="./output/influence_score/${current_date}/${task}-${current_date}-ckpt-${ckpt_number}/"

            if [[ ! -d $influence_score_base ]]; then
                mkdir -p $influence_score_base
            fi

            python3 ./icons/influence_matrix.py \
                --train_gradient_path $train_gradient_path \
                --validation_gradient_path $validation_gradient_path \
                --influence_score $influence_score_base \
                --train_file_name "${TASK_CONFIGS[$task]}"
        else
            echo "No matching validation gradient paths found for ${task}"
        fi
    fi
done 