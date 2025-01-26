# This script is used to split the training data json file into smaller chunks for parallel gradient computation.

import json
import math
import os
from multiprocessing import Pool
from functools import partial
import ijson
from tqdm import tqdm
import argparse

def save_chunk(chunk_data, output_dir, chunk_num):
    output_file = os.path.join(output_dir, f'chunk_{chunk_num}.json')
    with open(output_file, 'w') as f:
        json.dump(chunk_data, f, indent=4)

def process_chunk_with_indices(args):
    # Unpack arguments
    chunk_indices, input_file, output_dir, chunk_num = args
    return process_chunk(chunk_indices, input_file, output_dir, chunk_num)

def process_chunk(chunk_indices, input_file, output_dir, chunk_num):
    start_idx, end_idx = chunk_indices
    chunk_data = []
    
    with open(input_file, 'rb') as file:
        parser = ijson.items(file, 'item')
        
        for _ in range(start_idx):
            next(parser, None)
        
        for idx, item in enumerate(parser):
            if idx >= (end_idx - start_idx):
                break
            chunk_data.append(item)
    
    save_chunk(chunk_data, output_dir, chunk_num)
    return chunk_num

def split_json_file(input_file, output_dir, num_splits=2000):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Counting total items...")
    total_items = 0
    with open(input_file, 'rb') as file:
        parser = ijson.items(file, 'item')
        for _ in tqdm(parser, desc="Counting items"):
            total_items += 1
    
    print(f"\nTotal items: {total_items}")
    
    items_per_split = math.ceil(total_items / num_splits)
    print(f"Items per split: {items_per_split}")
    
    chunk_indices = [
        (i * items_per_split, min((i + 1) * items_per_split, total_items))
        for i in range(num_splits)
    ]
    
    process_args = [
        (indices, input_file, output_dir, i+1)
        for i, indices in enumerate(chunk_indices)
    ]
    
    cpu_count = os.cpu_count() 
    # Use 90% of available CPUs to leave some headroom for system processes
    num_workers = max(1, int(cpu_count * 0.9)) 
    
    print(f"\nUsing {num_workers} workers out of {cpu_count} available CPUs")
    
    chunk_size = 10  # Adjust this based on performance
    
    print("\nProcessing chunks...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_chunk_with_indices, process_args, chunksize=chunk_size),
            total=num_splits,
            desc="Splitting files"
        ))
    
    print("\nSplitting complete!")
    print(f"Created {len(results)} files in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a JSON file into multiple chunks')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('output_dir', type=str, help='Directory to store the output chunks')
    parser.add_argument('--num-splits', type=int, default=2000, help='Number of splits to create')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    from time import time
    start_time = time()
    
    split_json_file(args.input_file, args.output_dir, args.num_splits)
    
    elapsed_time = time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")