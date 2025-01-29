"""This file is for merging chunk gradient"""
import gc
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

def load_tensor(file_path):
    try:
        return torch.load(file_path, map_location='cpu', mmap=True)
    except Exception as e:
        print(f"Error loading tensor from {file_path}: {e}")
        return None

def merge_tensors(base_dir, output_file_name, chunk_size=5, data_per_chunk=3326):
    output_file = os.path.join(base_dir, output_file_name)
    id_file = os.path.join(base_dir, f"{output_file_name}_ids.txt")
    temp_dir = os.path.join(base_dir, "temp_merge_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    subdirs = [os.path.join(base_dir, subdir) for subdir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subdir))]
    
    valid_subdirs = []
    for subdir in subdirs:
        try:
            int(os.path.basename(subdir).split('_')[-1])
            valid_subdirs.append(subdir)
        except ValueError:
            print(f"Skipping directory with unexpected name format: {subdir}")
    
    valid_subdirs.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))

    all_files = []
    chunk_numbers = []
    for subdir in valid_subdirs:
        chunk_number = int(os.path.basename(subdir).split('_')[-1])
        found_files = False
        for root, _, files in os.walk(subdir):
            files.sort()
            for file in files:
                if file == "all_normalized.pt":  
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
                    chunk_numbers.append(chunk_number)
                    found_files = True

        if not found_files:
            print(f"No 'all_normalized.pt' found in folder: {subdir}")

    print(f"Total files found: {len(all_files)}")

    merged_data = []
    chunk_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:  #
        futures = {executor.submit(load_tensor, file_path): file_path for file_path in all_files}
        with open(id_file, 'w') as idf:
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Merging files")):
                data = future.result()
                if data is None:
                    print(f"Skipping file {all_files[i]} due to loading error.")
                    continue

                merged_data.append(data.cpu())  

                chunk_number = chunk_numbers[i]
                for j in range(data.shape[0]):
                    idf.write(f"{(chunk_number - 1) * data_per_chunk + j + 1}\n")

                if len(merged_data) >= chunk_size:
                    merged_chunk = torch.cat(merged_data, dim=0)
                    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_count}.pt")
                    torch.save(merged_chunk, chunk_file)
                    chunk_count += 1
                    merged_data = []
                    del merged_chunk
                    gc.collect()
                    torch.cuda.empty_cache()  
    if merged_data:
        merged_chunk = torch.cat(merged_data, dim=0)
        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_count}.pt")
        torch.save(merged_chunk, chunk_file)
        chunk_count += 1

    print(f"Saved IDs to {id_file}")

    save_dir = os.path.join(base_dir, output_file_name.replace('.pt', '_chunks'))
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = 0
    for i in range(chunk_count):
        chunk_file = os.path.join(temp_dir, f"chunk_{i}.pt")
        chunk = torch.load(chunk_file, map_location='cpu')
        total_samples += len(chunk)
        
        final_chunk_file = os.path.join(save_dir, f"chunk_{i}.pt")
        torch.save(chunk, final_chunk_file)
        
        os.remove(chunk_file)
        del chunk
        gc.collect()

    metadata = {
        'num_chunks': chunk_count,
        'total_samples': total_samples
    }
    torch.save(metadata, os.path.join(base_dir, output_file_name))
    
    output_file_txt = os.path.join(base_dir, f"{output_file_name}.txt")
    with open(output_file_txt, 'w') as f:
        f.write(f"Merged data files into chunks at {save_dir}\n")
        f.write(f"Total number of chunks: {chunk_count}\n")
        f.write(f"Total number of samples: {total_samples}\n")
    print(f"Saved tensor details to {output_file_txt}")

    os.rmdir(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tensor chunks")
    parser.add_argument('base_dir', type=str, help='Base directory containing chunk folders')
    args = parser.parse_args()
    
    merge_tensors(args.base_dir, "everything_all_normalized.pt")
    merge_tensors(args.base_dir, "everything_all_unormalized.pt")