# Understanding Transformers via _N_-gram statistics
[Original Paper](https://www.arxiv.org/pdf/2407.12034)


How do transformers relate to short context statistical patterns such as N-grams?

## Setup Instructions
Use python3.12.
Be certain python3.12-dev and c compiler are installed (for torch.compile)


First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then running any script with uv will automatically install the dependencies. Optionally, you can install the dependencies yourself with:

```bash
uv sync
```

## Download data
Add data to `data/`

## Usage
Given wikipedia is a dataset in `data/`

### Build Tokenizer
```bash
 uv run utils/tokenizer.py tinystories_1gb --name tinystories_1gb

```

### Build Dataset

```bash
#uv run utils/dataset.py [dataset] [tokenizer] --batch_size 100000
uv run utils/dataset.py tinystories_1gb tinystories_1gb --delineate
```

### Build Trie

```bash
uv run utils/ngram.py tinystories_1gb --tokenizer_name tinystories_1gb --ngram_file tinystories_1gb --ngram_size 8
```
### Start N-gram server
```bash
cargo run
```

### Train transformer and log metrics
```bash
uv run train.py --config llama_medium --dataset tinystories_1gb
```