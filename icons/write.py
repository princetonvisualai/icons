"""This file is for writing selected data"""
import argparse
import os
import torch
import json
from tqdm import tqdm
import datetime

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')
    argparser.add_argument('--use_multiple_matrices', action='store_true',
                           help='Whether to use multiple matrices or not')
    argparser.add_argument('--multiple_matrices', type=str, 
                           choices=['none', 'aggregate', 'argmax'],
                           default='none',
                           help='How to handle multiple matrices: none (single matrix), aggregate (add), or argmax')
    argparser.add_argument('--checkpoint_numbers', type=str,
                           help='Comma-separated list of checkpoint numbers')

    args = argparser.parse_args()
    return args

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count

def write_selected_data(full_data_path, selected_index_path, output_json_path):
    with open(selected_index_path, 'r') as file:
        content = file.read().strip()
        if content.startswith('[') and content.endswith(']'):
            # The file contains a list-like structure
            import ast
            support_indices = ast.literal_eval(content)
        else:
            # The file contains individual integers on separate lines
            support_indices = [int(line.strip()) for line in content.split('\n') if line.strip()]
    print(f"Number of selected indices: {len(support_indices)}")

    with open(full_data_path, 'r') as file:
        full_data = json.load(file)
    print(f"Total number of items: {len(full_data)}")

    selected_data = [full_data[i] for i in tqdm(support_indices, desc="Extracting selected data")]
    print(f"Number of items extracted: {len(selected_data)}")

    with open(output_json_path, 'w') as output_file:
        json.dump(selected_data, output_file, indent=4)
    print(f"The selected data has been saved to {output_json_path}")

if __name__ == "__main__":
    args = parse_args()
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cpu")

    # Define a legend mapping task names to method names
    method_legend = {
        'gqa': 'Graph Question Answering',
        'llavabench_in_the_wild': 'LLAVA Benchmark in the Wild',
        'mmbench_cn_no_hints': 'MMBench Chinese No Hints',
        'mmbench_no_hints': 'MMBench No Hints',
        'mme': 'Multimodal Evaluation',
        'pope': 'Pope Task',
        'sqa_a': 'Structured Question Answering A',
        'textvqa_really_cleaner': 'TextVQA Really Cleaner',
        'vizwiz': 'VizWiz Task',
        'vqav2': 'Visual Question Answering V2'
    }

    all_scores = []
    checkpoint_numbers = args.checkpoint_numbers.split(',') if args.checkpoint_numbers else ['']

    for target_task in args.target_task_names:
        method_name = method_legend.get(target_task, 'Unknown Method')
        print(f"Processing task: {target_task} ({method_name})")

        # Check if this is MME benchmark
        is_mme = 'mme' in args.output_path.lower()
        
        if is_mme:
            score_path = os.path.join(args.output_path, target_task, f"{target_task}_influence_score.pt")
        else:
            score_path = os.path.join(args.output_path, f"{target_task}_influence_score.pt")
        
        if not os.path.exists(score_path):
            print(f"Score file not found: {score_path}")
            continue
        
        score = torch.load(score_path, map_location=device)
        print(f"Shape of {target_task}_influence_score.pt: {score.shape}")
        all_scores.append(score)
        
    if not all_scores:
        raise ValueError("No valid score files found. Please check the file paths and try again.")
    
    all_scores = torch.cat(all_scores, dim=1)

    avg_scores = all_scores.mean(dim=1)

    total_samples = avg_scores.shape[0]
    print(f"Total number of samples: {total_samples}")

    if args.percentage is not None:
        num_samples = int(args.percentage * total_samples)
    else:
        num_samples = args.max_samples

    print(f"Selecting top {num_samples} samples...")

    # Get the indices of the top samples
    scores, indices = torch.topk(avg_scores, num_samples, largest=True)


    unsorted_indices = indices.tolist()  # These are indices ordered by their scores
    sorted_indices = sorted(unsorted_indices)  # These are indices in ascending order


    ckpt_str = '_'.join(checkpoint_numbers) if checkpoint_numbers else ''
    

    os.makedirs(args.output_path, exist_ok=True)
    

    unsorted_indices_file = os.path.join(args.output_path, f"train_indices_unsorted_p{args.percentage}_ckpts_{ckpt_str}.txt")
    with open(unsorted_indices_file, 'w') as f:
        for index in unsorted_indices:
            f.write(f"{index}\n")
    

    sorted_indices_file = os.path.join(args.output_path, f"train_indices_sorted_p{args.percentage}_ckpts_{ckpt_str}.txt")
    with open(sorted_indices_file, 'w') as f:
        for index in sorted_indices:
            f.write(f"{index}\n")

    print(f"Saved {num_samples} unsorted indices to {unsorted_indices_file}")
    print(f"Saved {num_samples} sorted indices to {sorted_indices_file}")


    selected_data_output_path = os.path.join(os.path.dirname(args.output_path), "selected_data")
    os.makedirs(selected_data_output_path, exist_ok=True)
    
    current_date = datetime.datetime.now().strftime("%m%d")

    for indices_type in ['sorted', 'unsorted']:
        indices_file = sorted_indices_file if indices_type == 'sorted' else unsorted_indices_file
        selected_data_file = os.path.join(selected_data_output_path, 
            f"{current_date}_p{args.percentage}_ckpts_{ckpt_str}_{indices_type}_selected_{target_task}.json")
        write_selected_data(args.train_files[0], indices_file, selected_data_file)
        print(f"The {indices_type} selected data has been saved to {selected_data_file}")
