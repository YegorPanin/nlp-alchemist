#!/usr/bin/env python3
from huggingface_hub import hf_hub_download
import os

def download_dataset_files():
    """Download required files from Hugging Face."""
    print("Downloading dataset files from Hugging Face...")
    
    # Define the files to download
    files = [
        "word_embeddings.faiss",  # Correct filename from HF repo
        "words.list"
    ]
    
    for filename in files:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id="YegorPanin/popular_word_embeddings_ru",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                raise
        else:
            print(f"{filename} already exists, skipping download")

if __name__ == "__main__":
    download_dataset_files()