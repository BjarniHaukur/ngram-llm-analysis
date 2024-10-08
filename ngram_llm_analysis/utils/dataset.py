from functools import lru_cache
from pathlib import Path
import tempfile

try: from .tokenizer import load_tokenizer
except ImportError: from tokenizer import load_tokenizer

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = Path("../data/")


class MemmapDataset(Dataset):
    """Reads tokens from a memmap file."""
    def __init__(self, dataset_file:str, tokenizer_name:str, num_tokens:int = 2048):
        self.memmap = np.memmap(
            DATA_PATH / dataset_file.removesuffix(".txt") / (tokenizer_name + ".dat"),
            dtype="uint16", mode="r"
        )
        self.num_tokens = num_tokens

    def __getitem__(self, idx: int):
        if idx < 0: idx += len(self)
        tokens = self.memmap[idx * self.num_tokens: (idx + 1) * self.num_tokens]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.memmap) // self.num_tokens
    
    @staticmethod
    def build_memmap(dataset_file:str, tokenizer_name:str, batch_size:int=10000):
        tokenizer = load_tokenizer(tokenizer_name)
        memmap_dir = DATA_PATH / dataset_file.removesuffix(".txt")
        memmap_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = DATA_PATH / (dataset_file if dataset_file.endswith(".txt") else dataset_file + ".txt")

        # First pass: Tokenize and write tokens to a temporary file
        total_tokens = 0
        temp_file = tempfile.TemporaryFile()
        with open(dataset_path, 'r', encoding='utf-8') as file:
            file.readline()  # Skip the first line
            batch_lines = []
            pbar = tqdm(desc="Tokenizing and writing to tempfile...", unit=" lines")
            for line in file:
                batch_lines.append(line)
                if len(batch_lines) >= batch_size:
                    # Tokenize batch
                    encoded_batch = tokenizer.encode_batch(batch_lines)
                    # Write tokens to temp_file
                    for enc in encoded_batch:
                        tokens = enc.ids
                        temp_file.write(np.array(tokens, dtype='uint16').tobytes())
                        total_tokens += len(tokens)
                    batch_lines = []
                    pbar.update(batch_size)
            # Process any remaining lines
            if batch_lines:
                encoded_batch = tokenizer.encode_batch(batch_lines)
                for enc in encoded_batch:
                    tokens = enc.ids
                    temp_file.write(np.array(tokens, dtype='uint16').tobytes())
                    total_tokens += len(tokens)
                pbar.update(len(batch_lines))
            pbar.close()

        # Create memmap with the determined shape
        memmap_path = memmap_dir / (tokenizer_name + ".dat")
        memmap = np.memmap(memmap_path, mode="w+", dtype='uint16', shape=(total_tokens,))

        # Second pass: Read from temp file and write to memmap
        temp_file.seek(0)
        bytes_per_token = np.dtype('uint16').itemsize
        buffer_size = 1024 * 1024  # 1MB buffer
        offset = 0
        with tqdm(total=total_tokens, desc="Writing tokens to memmap", unit=" tokens") as pbar:
            while True:
                chunk = temp_file.read(buffer_size)
                if not chunk:
                    break
                num_tokens = len(chunk) // bytes_per_token
                tokens = np.frombuffer(chunk, dtype='uint16', count=num_tokens)
                memmap[offset:offset + num_tokens] = tokens
                offset += num_tokens
                pbar.update(num_tokens)

        memmap.flush()
        temp_file.close()
        print(f"Memmap created with {total_tokens} tokens.")

if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Creating a memmap from the TinyStories dataset and a given tokenizer")
    parser.add_argument("file_name", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("--batch_size", type=int, default=10000)
    args = parser.parse_args()
    
    start = time.time()
    MemmapDataset.build_memmap(args.file_name, args.tokenizer_name, batch_size=args.batch_size)
    end = time.time()
    print(f"Time taken to build memmap: {end - start:.2f} seconds")

