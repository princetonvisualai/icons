import gc
import os
import torch
import psutil
import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTracker:
    def __init__(self, threshold_gb=0.85):
        self.threshold_gb = threshold_gb
        self.total_memory = psutil.virtual_memory().total / (1024**3)
        
    def get_memory_usage(self):
        """Returns current memory usage in GB"""
        return psutil.Process().memory_info().rss / (1024**3)
    
    def get_available_memory(self):
        """Returns available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    def is_memory_critical(self):
        """Returns True if memory usage is above threshold"""
        available_memory = self.get_available_memory()
        return available_memory < (self.total_memory * (1 - self.threshold_gb))
    
    def log_memory_status(self):
        """Logs current memory status"""
        used = self.get_memory_usage()
        available = self.get_available_memory()
        logger.info(f"Memory Usage: {used:.2f}GB | Available: {available:.2f}GB")

class ChunkManager:
    def __init__(self, base_dir, output_name, max_chunk_size_gb=10):
        self.base_dir = Path(base_dir)
        self.output_name = output_name
        self.max_chunk_size_gb = max_chunk_size_gb
        self.temp_dir = self.base_dir / "temp_merge_chunks"
        self.final_dir = self.base_dir / f"{output_name}_chunks"
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        self.temp_dir.mkdir(exist_ok=True)
        self.final_dir.mkdir(exist_ok=True)
        
    def estimate_tensor_size_gb(self, tensor):
        """Estimate tensor size in GB"""
        return tensor.element_size() * tensor.nelement() / (1024**3)
    
    def should_create_new_chunk(self, current_tensors):
        """Determine if a new chunk should be created based on size"""
        if not current_tensors:
            return False
        total_size = sum(self.estimate_tensor_size_gb(t) for t in current_tensors)
        return total_size >= self.max_chunk_size_gb

def load_tensor_safe(file_path, memory_tracker):
    """Load tensor with memory safety checks"""
    try:
        if memory_tracker.is_memory_critical():
            logger.warning("Memory usage critical, forcing garbage collection")
            gc.collect()
            
        # Explicitly disable CUDA memory caching
        torch.cuda.empty_cache()
        # Force CPU loading and disable grad
        with torch.no_grad():
            tensor = torch.load(file_path, map_location='cpu')
            tensor = tensor.cpu()  
            tensor.requires_grad_(False)  # Disable gradient tracking
        return tensor
    except Exception as e:
        logger.error(f"Error loading tensor from {file_path}: {e}")
        return None

def merge_tensors_cambrian(base_dir, output_file_name, batch_size=20, data_per_chunk=3326, num_workers=None):
    """
    Merge tensors with improved memory management and error handling
    
    Args:
        base_dir (str): Base directory containing data
        output_file_name (str): Name of output file
        batch_size (int): Number of files to process in each batch
        data_per_chunk (int): Number of data points per original chunk
        num_workers (int): Number of worker threads to use. If None, will be calculated based on CPU count
    """
    torch.cuda.is_available = lambda: False
    
    if num_workers is None:
        cpu_count = os.cpu_count()
        num_workers = max(1, min(cpu_count // 2, 8))  
    
    logger.info(f"Using {num_workers} workers based on CPU count {os.cpu_count()}")
    
    start_time = datetime.now()
    logger.info(f"Starting merge process at {start_time}")
    
    memory_tracker = MemoryTracker(threshold_gb=0.85)
    chunk_manager = ChunkManager(base_dir, output_file_name)
    
    base_dir = Path(base_dir)
    
    logger.info(f"Checking base directory: {base_dir}")
    logger.info(f"Looking for {'normalized' if 'normalized' in output_file_name else 'unormalized'} files")
    
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != "temp_merge_chunks"]
    logger.info(f"Found subdirectories: {[d.name for d in subdirs]}")
    valid_subdirs = []
    
    for subdir in subdirs:
        try:
            int(subdir.name.split('_')[-1])
            valid_subdirs.append(subdir)
        except ValueError:
            logger.warning(f"Skipping invalid directory: {subdir}")
    
    valid_subdirs.sort(key=lambda x: int(x.name.split('_')[-1]))
    logger.info(f"Valid subdirectories: {[d.name for d in valid_subdirs]}")
    
    all_files = []
    chunk_numbers = []
    
    for subdir in valid_subdirs:
        chunk_number = int(subdir.name.split('_')[-1])
        file_suffix = "all_normalized.pt" if "normalized" in output_file_name else "all_unormalized.pt"
        target_file = subdir / "dim5120" / file_suffix
        
        logger.info(f"Checking for {file_suffix} in {subdir}/dim5120")
        if target_file.exists():
            logger.info(f"Found file: {target_file}")
            all_files.append(target_file)
            chunk_numbers.append(chunk_number)
        else:
            logger.warning(f"Missing file {file_suffix} in {subdir}/dim5120")
    
    logger.info(f"Found {len(all_files)} files to process")
    
    # explicitly sorts the chunks before processing:
    sorted_pairs = sorted(zip(all_files, chunk_numbers), key=lambda x: x[1])
    all_files, chunk_numbers = zip(*sorted_pairs)
    
    logger.info(f"Starting processing of {len(all_files)} files in strict numerical order")
    logger.info(f"Chunk number range: {min(chunk_numbers)} to {max(chunk_numbers)}")

    merged_data = []
    chunk_count = 0
    total_samples = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, len(all_files), batch_size):
            batch_end = min(batch_start + batch_size, len(all_files))
            batch_files = all_files[batch_start:batch_end]
            batch_numbers = chunk_numbers[batch_start:batch_end]
            
            batch_num = batch_start//batch_size + 1
            total_batches = len(all_files)//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (chunks {batch_numbers[0]} to {batch_numbers[-1]})")
            memory_tracker.log_memory_status()
            
            # Process files sequentially within batch
            batch_results = []
            for file_path, chunk_number in zip(batch_files, batch_numbers):
                try:
                    logger.info(f"Processing chunk {chunk_number}")
                    data = load_tensor_safe(str(file_path), memory_tracker)
                    if data is not None:
                        batch_results.append(data.cpu())
                        total_samples += len(data)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_number} from {file_path}: {e}")
                    continue

            # When chunks are merged, they maintain their order through torch.cat() which concatenates tensors in the order they're provided
            if batch_results:
                try:
                    merged_chunk = torch.cat(batch_results, dim=0)
            
                    chunk_file = chunk_manager.temp_dir / f"chunk_{batch_numbers[0]}_{batch_numbers[-1]}.pt"
                    torch.save(merged_chunk, chunk_file)
                    
                 
                    final_chunk_file = chunk_manager.final_dir / f"chunk_{batch_numbers[0]}_{batch_numbers[-1]}.pt"
                    chunk_file.rename(final_chunk_file)
                    
                    logger.info(f"Saved chunks {batch_numbers[0]} to {batch_numbers[-1]}")
                    chunk_count += 1
                    del merged_chunk
                except Exception as e:
                    logger.error(f"Error merging batch for chunks {batch_numbers[0]} to {batch_numbers[-1]}: {e}")

            del batch_results
            gc.collect()
            torch.cuda.empty_cache()

    if merged_data:
        merged_chunk = torch.cat(merged_data, dim=0)
        final_chunk_file = chunk_manager.final_dir / f"chunk_{chunk_count}.pt"
        torch.save(merged_chunk, final_chunk_file)
        chunk_count += 1
    
    metadata = {
        'num_chunks': chunk_count,
        'total_samples': total_samples,
        'creation_date': datetime.now().isoformat(),
        'chunk_locations': str(chunk_manager.final_dir)
    }
    torch.save(metadata, base_dir / output_file_name)
    
    summary_file = base_dir / f"{output_file_name}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Merge completed at: {datetime.now()}\n")
        f.write(f"Total processing time: {datetime.now() - start_time}\n")
        f.write(f"Total chunks created: {chunk_count}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Chunks directory: {chunk_manager.final_dir}\n")
    
    logger.info(f"Merge completed. Summary saved to {summary_file}")
    chunk_manager.temp_dir.rmdir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge tensor chunks with advanced memory management")
    parser.add_argument('base_dir', type=str, help='Base directory containing chunk folders')
    parser.add_argument('--batch-size', type=int, default=20, help='Number of files to process in each batch')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker threads to use')
    args = parser.parse_args()
    
    merge_tensors_cambrian(args.base_dir, "everything_all_normalized.pt", 
                          batch_size=args.batch_size, 
                          num_workers=args.num_workers)
    merge_tensors_cambrian(args.base_dir, "everything_all_unormalized.pt", 
                          batch_size=args.batch_size,
                          num_workers=args.num_workers) 