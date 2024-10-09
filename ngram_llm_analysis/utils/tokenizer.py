import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CHECKPOINT_PATH = Path("../checkpoints/tokenizer/")

BOS, BOS_ID = "<bos>", 0
EOS, EOS_ID = "<eos>", 1
PAD, PAD_ID = "<pad>", 2
UNK, UNK_ID = "<unk>", 3

def fit_tokenizer(text_iterator, vocab_size:int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        special_tokens=[UNK, PAD, BOS, EOS],
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=True,
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer

def path(filename:str) -> Path:
    return CHECKPOINT_PATH / (filename if filename.endswith(".json") else filename + ".json")

def save_tokenizer(tokenizer:Tokenizer, filename:str):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    filepath = path(filename)
    tokenizer.save(str(filepath))

def load_tokenizer(filename:str)->Tokenizer:
    filepath = path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Tokenizer file {filepath} not found. Please build the tokenizer first.")
    return Tokenizer.from_file(str(filepath))
 

RESET_BG = '\x1b[0m'
COLORS = [
    (194, 224, 255),
    (255, 218, 194),
    (194, 255, 208),
    (255, 194, 224),
    (218, 255, 194),
]

def ansi_color(rgb:Tuple[int,int,int]) -> str:
    return f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def html_color(rgb:Tuple[int,int,int]) -> str:
    return f'style="background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"'

def color_text_ansi(tokenizer:Tokenizer, text:str) -> str:
    encoding = tokenizer.encode(text)
    colored = ""
    for i, (token_id, word_id) in enumerate(zip(encoding.ids, encoding.word_ids)):
        token = tokenizer.decode([token_id])
        if word_id is not None:
            color = COLORS[word_id % len(COLORS)]
            colored += ansi_color(color) + token + RESET_BG
        else:
            colored += token
    return colored

def color_text_html(tokenizer:Tokenizer, text:str) -> str:
    encoding = tokenizer.encode(text)
    colored = ""
    for i, (token_id, word_id) in enumerate(zip(encoding.ids, encoding.word_ids)):
        token = tokenizer.decode([token_id])
        if word_id is not None:
            color = COLORS[word_id % len(COLORS)]
            token_text = token.replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            colored += f'<span {html_color(color)}>{token_text}</span>'
        else:
            colored += token
    return colored

def build_tokenizer(data_name:str, name:str = "tokenizer", vocab_size:int = 8096, chunk_size:int = 1024**2):
    """
    Builds a tokenizer from a 'data_name' file in the data folder.
    The tokenizer is saved to a file named 'name' in tokenizer checkpoints.
    """
    data_file = "../data/" + (data_name if data_name.endswith(".txt") else data_name + ".txt")
    total_lines = 0

    # First pass to count total lines
    with open(data_file, "r") as f:
        f.readline()  # Skip the first line
        for _ in f:
            total_lines += 1

    num_lines_to_read = chunk_size

    def data_iterator():
        with open(data_file, "r") as f:
            f.readline()  # Skip the first line
            for i, line in enumerate(f):
                if i >= num_lines_to_read:
                    break
                yield line.strip()

        # Ensure all characters are included
        yield ''.join([chr(i) for i in range(32, 127)])

    tokenizer = fit_tokenizer(data_iterator(), vocab_size)
    save_tokenizer(tokenizer, name)

    return tokenizer

if __name__ == "__main__":
    import random
    import argparse

    random.seed(1337)

    parser = argparse.ArgumentParser(description="Fit a Byte-Pair Encoding tokenizer")
    parser.add_argument("data_name", type=str)
    parser.add_argument("--name", type=str, default="tokenizer")
    parser.add_argument("--chunk_size", type=int, default=1024**2, help="Number of lines to process at a time")
    parser.add_argument("--vocab_size", type=int, default=8192)
    args = parser.parse_args()

    build_tokenizer(args.data_name, args.name, args.vocab_size, args.chunk_size)


