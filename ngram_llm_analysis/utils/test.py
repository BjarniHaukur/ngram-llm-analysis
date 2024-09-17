import argparse
from pathlib import Path
from typing import Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def fit_huggingface_tokenizer(data_path: str, iterations: int, tokenizer_name: str, proportion: float = 1.0):
    # Read the data
    with open(data_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip the first line (assuming it's "text")
        data = f.read()
    
    # Apply proportion if needed
    data = data[:int(len(data) * proportion)]
    
    # Add ASCII characters to ensure they're all in the vocabulary
    data += ''.join([chr(i) for i in range(32, 127)])

    # Initialize the tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the trainer
    trainer = BpeTrainer(
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        vocab_size=iterations + 256,  # Base vocabulary size + number of merges
        min_frequency=2,
        show_progress=True,
    )

    # Train the tokenizer
    tokenizer.train_from_iterator([data], trainer=trainer)

    # Save the tokenizer
    output_path = Path("../checkpoints/tokenizer")
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / f"{tokenizer_name}.json"))
    
    
RESET_BG = '\x1b[0m'
COLORS = [
    (194, 224, 255),
    (255, 218, 194),
    (194, 255, 208),
    (255, 194, 224),
    (218, 255, 194),
]

def ansi_color(rgb: Tuple[int, int, int]) -> str:
    return f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def html_color(rgb: Tuple[int, int, int]) -> str:
    return f'style="background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"'

def color_text_ansi(tokenizer: Tokenizer, text: str) -> str:
    encoding = tokenizer.encode(text)
    print(encoding)
    colored = ""
    for i, (token, word_id) in enumerate(zip(encoding.tokens, encoding.word_ids)):
        if word_id is not None:
            color = COLORS[word_id % len(COLORS)]
            colored += ansi_color(color) + token + RESET_BG
        else:
            colored += token
    return colored

def color_text_html(tokenizer: Tokenizer, text: str) -> str:
    encoding = tokenizer.encode(text)
    colored = ""
    for i, (token, word_id) in enumerate(zip(encoding.tokens, encoding.word_ids)):
        if word_id is not None:
            color = COLORS[word_id % len(COLORS)]
            token_text = token.replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            colored += f'<span {html_color(color)}>{token_text}</span>'
        else:
            colored += token.replace(" ", "&nbsp;")
    return colored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a Hugging Face BPE tokenizer")
    parser.add_argument("--file_names", type=str, default="cleaned_data.txt", help="Name of the data file")
    parser.add_argument("--iterations", type=int, default=100, help="Number of BPE merges")
    parser.add_argument("--tokenizer_name", type=str, default="hf_tokenizer", help="Name for the saved tokenizer")
    parser.add_argument("--proportion", type=float, default=1.0, help="Proportion of data to use")
    args = parser.parse_args()

    data_path = Path("../data") / (args.data_name if args.data_name.endswith(".txt") else f"{args.data_name}.txt")

    fit_huggingface_tokenizer(
        data_path=str(data_path),
        iterations=args.iterations,
        tokenizer_name=args.tokenizer_name,
        proportion=args.proportion
    )

    print(f"Tokenizer saved as {args.tokenizer_name}.json in the ../checkpoints/tokenizer directory.")