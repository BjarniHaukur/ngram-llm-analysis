from functools import lru_cache
from pathlib import Path

try: from .tokenizer import load_tokenizer
except ImportError: from tokenizer import load_tokenizer

import numpy as np
import torch
from torch.utils.data import Dataset

from datetime import datetime

DATA_PATH = Path("../data/")


class MemmapDataset(Dataset):
    """Reads tokens from a memmap file."""
    def __init__(self, memmap_name:str, tokenizer_name:str, num_tokens:int = 4096):
        self.memmap = np.memmap(
            DATA_PATH / "memmaps" / memmap_name.removesuffix(".dat") / (tokenizer_name + ".dat"),
            dtype="uint16", mode="r"
        )
        self.num_tokens = num_tokens

    @lru_cache()
    def __getitem__(self, idx: int):
        if idx < 0: idx += len(self)
        tokens = self.memmap[idx * self.num_tokens: (idx + 1) * self.num_tokens]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.memmap) // self.num_tokens
    

    def build_memmap(dataset_file:str, tokenizer_name:str, memmap_name:str, verbose:bool = False):
        tokenizer = load_tokenizer(tokenizer_name)
        with open(DATA_PATH / (dataset_file if dataset_file.endswith(".txt") else dataset_file + ".txt"), "r", encoding="utf-8") as file:
            text = file.read()
        tokens = tokenizer.encode(text).ids
        
        memmap_dir = DATA_PATH / "memmaps" / memmap_name.removesuffix(".dat")
        memmap_dir.mkdir(parents=True, exist_ok=True)
        
        memmap = np.memmap((memmap_dir / (tokenizer_name + ".dat")), mode="w+", dtype='uint16', shape=(len(tokens),))
        memmap[:] = tokens
        memmap.flush()
        if verbose:
            print(f"Memmap created with {len(tokens)} tokens.")
    
    def exists(memmap_name:str, tokenizer_name:str) -> bool:
        path = DATA_PATH / "memmaps" / memmap_name.removesuffix(".dat") / (tokenizer_name + ".dat")
        return Path(path).exists()

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Creating a memmap from the TinyStories dataset and a given tokenizer")
    parser.add_argument("file_name", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("memmap_name", type=str)
    args = parser.parse_args()
    
    MemmapDataset.build_memmap(args.file_name, args.tokenizer_name, args.memmap_name, verbose=True)

