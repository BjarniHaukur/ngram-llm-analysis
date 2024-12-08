from functools import lru_cache
from pathlib import Path
import tempfile

try: from .tokenizer import load_tokenizer, BOS, EOS
except ImportError: from tokenizer import load_tokenizer, BOS, EOS

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = Path(__file__).parent.parent.parent / "data"


class MemmapDataset(Dataset):
    """Reads tokens from a memmap file."""
    def __init__(self, dataset_file:str, tokenizer_name:str, num_tokens:int = 2048):
        self.memmap = np.memmap(
            DATA_PATH / (dataset_file.removesuffix(".txt") + "_delineated") / (tokenizer_name + ".dat"),
            dtype="uint16", mode="r"
        )
        self.num_tokens = num_tokens

    def __getitem__(self, idx: int):
        if idx < 0: idx += len(self)
        tokens = self.memmap[idx * self.num_tokens: (idx + 1) * self.num_tokens]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.memmap) // self.num_tokens
    
def build_memmap(dataset_file:str, tokenizer_name:str, batch_size:int=8192):
    tokenizer = load_tokenizer(tokenizer_name)
    memmap_dir = DATA_PATH / dataset_file.removesuffix(".txt")
    memmap_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_PATH / (dataset_file if dataset_file.endswith(".txt") else dataset_file + ".txt")

    print(f"Building memmap for {dataset_file} with tokenizer {tokenizer_name}...")

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

def delineate_tinystories(file_name:str):
    """Delineates the TinyStories dataset into stories wrapped in <bos> and <eos> tokens (instead of <bos> and <eos> tokens at the beginning and end of the sequence)."""
    file_name = file_name + ".txt" if not file_name.endswith(".txt") else file_name
    with open(DATA_PATH / file_name, 'r', encoding='utf-8') as f:
        f.readline() #
        lines = f.readlines()

    stories = []
    current_story = []
    prev_line_empty = True  # Start with True to handle the first line
        
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # If the line is not empty after stripping whitespace
            if not prev_line_empty:
                # If the previous line was not empty, and the current line is not empty,
                # this indicates the start of a new story.
                if current_story:
                    # Save the current story before starting a new one
                    stories.append(''.join(current_story).strip())
                    current_story = []
            current_story.append(line)
            prev_line_empty = False
        else:
            # If the line is empty, add it to the current story (to maintain paragraph spacing)
            current_story.append(line)
            prev_line_empty = True

    # After the loop, don't forget to add the last story if there is one
    if current_story:
        stories.append(''.join(current_story).strip())

    # Prepend <bos> and append <eos> to each story
    stories = [BOS + story + EOS for story in stories]

    # Save to file
    with open(DATA_PATH / file_name.replace(".txt", "_delineated.txt"), "w", encoding="utf-8") as f:
        for story in stories:
            f.write(story + "\n")


if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Creating a memmap from the TinyStories dataset and a given tokenizer")
    parser.add_argument("file_name", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--delineate", action="store_true")
    args = parser.parse_args()

    args.file_name = args.file_name + ".txt" if not args.file_name.endswith(".txt") else args.file_name
    
    start = time.time()

    if args.delineate:
        if (file_name := DATA_PATH / args.file_name.replace(".txt", "_delineated.txt")).exists():
            print(f"Found delineated file: {file_name}")
            print(file_name)
        else:
            print(f"Delineating {args.file_name}...")
            delineate_tinystories(args.file_name)

        args.file_name = str(file_name.stem)

    build_memmap(args.file_name, args.tokenizer_name, batch_size=args.batch_size)

    end = time.time()
    print(f"Time taken to build memmap: {end - start:.2f} seconds")

