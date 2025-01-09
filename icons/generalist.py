"""Generalist stage: This file is for aggregating influence scores and selecting top samples"""
import numpy as np
import os
from pathlib import Path
import torch
import argparse
import gc
import json
from tqdm import tqdm
from icons.write import write_selected_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_process_matrix(file_path):
    """
    Load and process a single influence matrix with memory efficiency
    """
    print(f"Processing {file_path}")
    matrix = torch.load(file_path)
    if torch.is_tensor(matrix):
        # Calculate mean scores immediately and free the original matrix
        row_means = matrix.mean(dim=1).cpu().numpy()
        del matrix
        torch.cuda.empty_cache()
        gc.collect()
        return row_means
    else:
        row_means = np.mean(matrix, axis=1)
        del matrix
        gc.collect()
        return row_means

def load_influence_matrices(base_paths):
    """
    Load all influence matrices from multiple base paths with memory efficiency
    """
    mean_scores = []
    task_names = []
    
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    
    for base_path in base_paths:
        print(f"\nProcessing base path: {base_path}")
        
        # First check if there's an MME directory and process it as one task
        mme_dir = None
        for d in os.listdir(base_path):
            if d.startswith('mme-') and os.path.isdir(os.path.join(base_path, d)):
                mme_dir = os.path.join(base_path, d)
                break
        
        if mme_dir:
            # Handle MME directory structure
            print(f"Found MME directory: {mme_dir}")
            mme_subtask_scores = []
            
            task_dirs = [d for d in os.listdir(mme_dir) 
                        if os.path.isdir(os.path.join(mme_dir, d))]
            
            print(f"Found {len(task_dirs)} MME subtask directories:")
            for task_dir in task_dirs:
                file_path = os.path.join(mme_dir, task_dir, f'{task_dir}_influence_score.pt')
                
                if os.path.exists(file_path):
                    print(f"Loading MME subtask {task_dir} influence matrix from {file_path}")
                    row_means = load_and_process_matrix(file_path)
                    mme_subtask_scores.append(row_means)
                else:
                    print(f"Warning: No influence score file found for MME subtask {task_dir}")
            
            if mme_subtask_scores:
                # Take maximum across all MME subtasks
                mme_scores = np.maximum.reduce(mme_subtask_scores)
                mean_scores.append(mme_scores)
                task_names.append('mme')
        
        # Process regular task directories
        task_dirs = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))
                    and not d.startswith('mme-')
                    and not d == 'selected_data']
        
        print(f"Found {len(task_dirs)} regular task directories:")
        for task_dir in task_dirs:
            task_name = task_dir.split('-')[0]
            file_path = os.path.join(base_path, task_dir, f'{task_name}_influence_score.pt')
            
            if os.path.exists(file_path):
                print(f"Loading {task_name} influence matrix from {file_path}")
                row_means = load_and_process_matrix(file_path)
                mean_scores.append(row_means)
                task_names.append(task_name)
            else:
                print(f"Warning: No influence score file found for {task_name}")
    '''
    The np.column_stack(mean_scores) creates a matrix where:
    Each row represents one training sample
    Each column represents a different task
    The value at position [i,j] is the mean influence score of training sample i for task j
    '''
    return np.column_stack(mean_scores), task_names

def process_influence_matrices(base_path, output_path, full_data_path, top_percent=20):
    """
    Process influence matrices using voting method:
    
    vote: Threshold-based voting system.
          Counts how many tasks consider the sample in their top K%.
          Good for finding samples consistently important across tasks.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print("Loading and processing influence matrices...")
    mean_scores_matrix, task_names = load_influence_matrices(base_path)
    
    n_samples = mean_scores_matrix.shape[0]
    k = int((top_percent/100) * n_samples)
    
    print("\nProcessing voting method...")
    
    top_masks = []
    for scores_col in mean_scores_matrix.T:
        threshold = np.percentile(scores_col, 100-top_percent)
        top_masks.append(scores_col >= threshold)
    scores = np.sum(np.column_stack(top_masks), axis=1)
    
    print(f"\nOverall statistics:")
    print(f"- Top {top_percent}% count min: {np.min(scores):.3f}")
    print(f"- Top {top_percent}% count max: {np.max(scores):.3f}")
    print(f"- Top {top_percent}% count mean: {np.mean(scores):.3f}")
    
    # Get top indices
    top_indices = np.argsort(scores)[-k:]
    
    # Save results
    shortname = 'vote'
    indices_txt_path = os.path.join(output_path, f'{shortname}_top_indices.txt')
    np.savetxt(indices_txt_path, top_indices, fmt='%d')
    print(f"Saved top indices to: {indices_txt_path}")
    
    output_json_path = os.path.join(output_path, f'{shortname}_selected_data.json')
    write_selected_data(full_data_path, indices_txt_path, output_json_path)
    print(f"Saved selected data to: {output_json_path}")
    
    print("\nAll results saved to directory:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_influence_path', type=str,
                       required=True,
                       help="Path containing all influence matrices")
    parser.add_argument('--top_percent', type=float, default=20,
                       help="Percentage of top samples to select")
    parser.add_argument('--output_dir', type=str,
                       required=True,
                       help="Output directory for aggregated results")
    parser.add_argument('--full_data_path', type=str, 
                       default='./Data/llava/download/llava-v1.5-instruct/llava_v1_5_mix665k.json',
                       help='Path to the llava dataset json file')
    
    args = parser.parse_args()
    
    process_influence_matrices(args.all_influence_path, args.output_dir, args.full_data_path, args.top_percent)