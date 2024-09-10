from typing import Literal
from functools import lru_cache
from pathlib import Path

try: from .tokenizer import BPETokenizer # super lazy
except ImportError: from tokenizer import BPETokenizer

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

DATA_PATH = Path("../data/")


class MemmapDataset(Dataset):
    """ Reads tokens from a memmap file. """
    def __init__(self, memmap_name:str, num_tokens: int = 4096):
        self.memmap = np.memmap(
            DATA_PATH / (memmap_name if memmap_name.endswith(".dat") else memmap_name + ".dat"),
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Creating a memmap from the TinyStories dataset and a given tokenizer")
    parser.add_argument("file_name", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("memmap_name", type=str)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer_name)

    print("Reading and tokenizing data...")
    with open(DATA_PATH / (args.file_name if args.file_name.endswith(".txt") else args.file_name + ".txt"), "r", encoding="iso-8859-1") as file: # does this need to be iso-8859-1?
        text = file.read()
        
    tokens = tokenizer.tokenize(text)

    # Create memmap for the entire dataset
    print("Creating memmap...")
    memmap = np.memmap(DATA_PATH / (args.memmap_name + ".dat"), mode="w+", dtype='uint16', shape=(len(tokens),))
    memmap[:] = tokens
    memmap.flush()

    print(f"Memmap created with {len(tokens)} tokens.")
