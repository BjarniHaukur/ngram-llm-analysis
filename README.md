# PyGPT

Use python3.12.
Be certain python3.12-dev and c compiler are installed (for torch.compile)

## Setup Instructions

**MacOS / Linux:**

```bash
python3 -m venv .venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.\venv\Scripts\activate
```

**Install Dependencies**

With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

Let VSCode install the requirements for running the notebooks.

**CUDA**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Download data
- TODO

## RUN
cd into ngram_llm_analysis/

### Examples

Fit Tokenizer
```bash
python3 utils/tokenizer.py cleaned_train --name tokenizer_bytes --vocab_size 16384
```

Tokenize Data
```bash
python3 utils/dataset.py cleaned_train tokenizer_bytes
```

Train Transformer

```bash
python3 train.py --config llama_small --dataset small_train
```



