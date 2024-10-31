# PyGPT

Use python3.12.
Be certain python3.12-dev and c compiler are installed (for torch.compile)

## Setup Instructions

First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then running any script with uv will automatically install the dependencies. Optionally, you can install the dependencies yourself with:

```bash
uv sync
```

## Download data
- TODO

## RUN
cd into ngram_llm_analysis/

### Examples
Training

`uv run train.py --config llama_small --dataset small_train`

Building a specific tokenizer

`uv run utils/tokenizer.py <dataset> --name <tokenizer_name>`

Using the specialized tokenizer
`uv run train.py --config llama_small --dataset small_train --tokenizer <tokenizer_name>`


