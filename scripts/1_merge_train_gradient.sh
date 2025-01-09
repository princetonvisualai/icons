#!/bin/bash


process_folder() {
    local BASE_DIR="$1"
    
    echo "Checking for missing chunks in $BASE_DIR..."
    python icons/missing_chunks.py "$BASE_DIR"
    
    if [ -f "missing_files_chunks.txt" ] && [ -s "missing_files_chunks.txt" ] || \
       [ -f "missing_chunks.txt" ] && [ -s "missing_chunks.txt" ]; then
        echo "Missing chunks detected in $BASE_DIR. Skipping merge process."
        return 1
    fi
    
    if ls "$BASE_DIR"/everything* 1> /dev/null 2>&1; then
        echo "Files starting with 'everything' already exist in $BASE_DIR"
        echo "Skipping the merge process. You can find the existing file(s) at:"
        ls -1 "$BASE_DIR"/everything*
    else
        python ./icons/merge_chunks.py "$BASE_DIR"
    fi
}


read -p "Multiple folders or single folder? (m/s): " FOLDER_CHOICE

if [[ $FOLDER_CHOICE == "s" ]]; then
    read -p "Enter the BASE_DIR for merging the training gradient: " BASE_DIR
    process_folder "$BASE_DIR"
elif [[ $FOLDER_CHOICE == "m" ]]; then
    read -p "Enter the path to the parent directory containing all folders: " PARENT_DIR
    
    SUBDIRS=$(find "$PARENT_DIR" -maxdepth 1 -type d | tail -n +2)
    
    echo "Processing folders in parallel..."
    for DIR in $SUBDIRS; do
        process_folder "$DIR" &
    done
    
    wait
    echo "All folders have been processed."
else
    echo "Invalid choice. Please run the script again and choose 'm' for multiple or 's' for single."
    exit 1
fi
