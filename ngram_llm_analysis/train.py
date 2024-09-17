import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import ConcatDataset, random_split
from transformers import Trainer, TrainingArguments, LlamaForCausalLM, LlamaConfig, GPT2LMHeadModel, GPT2Config

from utils.dataset import MemmapDataset
from utils.tokenizer import load_tokenizer
from utils.metrics import bleu_score

torch.random.manual_seed(1337)

CHECKPOINT_PATH = Path("../checkpoints/models")

def collate_fn(batch, tokenizer, max_len=2048):
    batch = [x[:max_len] for x in batch]
    batch = [
        torch.cat([torch.tensor([tokenizer.token_to_id("<bos>")], dtype=torch.long), x, torch.tensor([tokenizer.token_to_id("<eos>")], dtype=torch.long)])
        for x in batch
    ]
    return {
        'input_ids': torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.token_to_id("<pad>")
        ),
        'labels': torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=-100
        )
    }

def model_from_config(config_name:str, vocab_size:int):
    with open("../configs/" + (config_name if config_name.endswith(".yaml") else config_name + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
    
    model_type = config.get('model_type', '').lower()
    config["vocab_size"] = vocab_size
    
    if model_type == 'llama':
        model_config = LlamaConfig(**config)
        return LlamaForCausalLM(model_config)
    elif model_type == 'gpt2':
        model_config = GPT2Config(**config)
        return GPT2LMHeadModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {DEVICE}")

    tokenizer = load_tokenizer("tokenizer")
    tokenizer.pad_token = tokenizer.token_to_id("<pad>")
    
    full_ds = MemmapDataset(args.dataset, args.seq_len)
    
    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, _ = random_split(full_ds, [train_size, val_size, test_size])

    model = model_from_config(args.config, tokenizer.get_vocab_size())

    print(f"Training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    model_path = CHECKPOINT_PATH / (args.run_name or "model")
    model_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        
        return {"bleu_score": bleu_score(labels.tolist(), predictions.tolist(), n_gram=4)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, tokenizer, args.seq_len-1),
        compute_metrics=compute_metrics,
    )

    if args.continue_from:
        trainer.train(resume_from_checkpoint=str(CHECKPOINT_PATH / args.continue_from))
    else:
        trainer.train()

    trainer.save_model(str(model_path / "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language model for Python code generation")
    parser.add_argument("--config", type=str, required=True, help="Name of the model configuration file")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the memmap file")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()
    main(args)
