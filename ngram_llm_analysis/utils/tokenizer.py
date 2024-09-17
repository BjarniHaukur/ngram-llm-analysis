import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CHECKPOINT_PATH = Path("../checkpoints/tokenizer/")

BOS, BOS_ID = "<bos>", 0
EOS, EOS_ID = "<eos>", 1
PAD, PAD_ID = "<pad>", 2
UNK, UNK_ID = "<unk>", 3

def fit_tokenizer(text:str, vocab_size:int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        special_tokens=[UNK, PAD, BOS, EOS],
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer

def save_tokenizer(tokenizer:Tokenizer, filename:str):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    filepath = CHECKPOINT_PATH / (filename if filename.endswith(".json") else filename + ".json")
    tokenizer.save(str(filepath))

def load_tokenizer(filename:str) -> Tokenizer:
    filepath = CHECKPOINT_PATH / (filename if filename.endswith(".json") else filename + ".json")
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

if __name__ == "__main__":
    import random
    import argparse

    random.seed(1337)

    parser = argparse.ArgumentParser(description="Fit a Byte-Pair Encoding tokenizer")
    parser.add_argument("data_name", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="tokenizer")
    parser.add_argument("--proportion", type=float, default=0.5)
    parser.add_argument("--vocab_size", type=int, default=8096)
    args = parser.parse_args()

    with open("../data/" + (args.data_name if args.data_name.endswith(".txt") else args.data_name + ".txt"), "r") as f:
        f.readline()  # the first line is just "text"
        data = f.read()
    
    data = data[:int(len(data) * args.proportion)]
    
    # make sure all characters are in the data
    data += ''.join([chr(i) for i in range(32, 127)])

    tokenizer = fit_tokenizer(data, args.vocab_size)
    save_tokenizer(tokenizer, args.tokenizer_name)
