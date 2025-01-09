"""This file is for finding missing chunks"""
import os
import sys

def check_missing_chunks(base_dir):
    missing_files_chunks = []
    missing_chunks = []

    for chunk_id in range(1, 2001):
        chunk_dir = os.path.join(base_dir, f"chunk_{chunk_id}/dim5120")
        if os.path.isdir(chunk_dir):
            file_path = os.path.join(chunk_dir, "all_normalized.pt")
            if not os.path.exists(file_path):
                missing_files_chunks.append(chunk_id)
        else:
            missing_chunks.append(chunk_id)

    output_file = "./cambrian_missing_chunks_report.txt"
    with open(output_file, "w") as f:
        if missing_files_chunks:
            f.write("Chunks without 'all_normalized.pt':\n")
            for chunk_id in missing_files_chunks:
                f.write(f"chunk_{chunk_id}\n")
        else:
            f.write("All existing chunks contain 'all_normalized.pt'.\n")

        if missing_chunks:
            f.write("\nMissing chunk directories:\n")
            for chunk_id in missing_chunks:
                f.write(f"chunk_{chunk_id}\n")

    if missing_files_chunks or missing_chunks:
        print(f"\nResults have been saved to {output_file}.")
    
    return missing_files_chunks, missing_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 0_missing_chunks.py <base_directory>")
        sys.exit(1)
    base_dir = sys.argv[1]
    check_missing_chunks(base_dir)
