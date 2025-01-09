#!/bin/bash

DATE=$(date +%m%d)
BASE_PREFIX="./output/influence_score"

read -p "Enter the top percentage to select (default is 20): " TOP_PERCENT
TOP_PERCENT=${TOP_PERCENT:-20}  # Use 20 as default if no input provided

read -p "Enter the path containing all influence matrices from all tasks: " ALL_INFLUENCE_PATH

read -p "Enter a custom name for this aggregation under the path of ./output/final_aggregation/${DATE}/: " CUSTOM_NAME

if [ -z "$CUSTOM_NAME" ]; then
    echo "Error: Custom name cannot be empty"
    exit 1
fi

OUTPUT_DIR="./output/final_aggregation/${DATE}/${CUSTOM_NAME}"
mkdir -p $OUTPUT_DIR
read -p "Output directory $OUTPUT_DIR will be created. Press Enter to continue..."

if [ -z "$ALL_INFLUENCE_PATH" ]; then
    echo "Error: Influence path cannot be empty"
    exit 1
fi

echo "Using influence path: $ALL_INFLUENCE_PATH"
read -p "Is this correct? (y/n): " confirm
if [[ $confirm != "y" ]]; then
    echo "Aborting. Please run the script again with correct path."
    exit 1
fi

python ./icons/generalist.py \
    --all_influence_path "$ALL_INFLUENCE_PATH" \
    --top_percent $TOP_PERCENT \
    --output_dir "$OUTPUT_DIR"
