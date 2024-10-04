import argparse
from functools import partial
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb
import yaml
from torch.utils.data import random_split, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
from utils.dataset import MemmapDataset
from utils.metrics import bleu_score
from utils.tokenizer import build_tokenizer, load_tokenizer

torch.random.manual_seed(1337)

CHECKPOINT_PATH = Path("../checkpoints/models")

def collate_fn(batch, tokenizer, max_len=2048):
    batch = [x[:max_len] for x in batch]
    batch = [
        torch.cat([torch.tensor([tokenizer.token_to_id("<bos>")], dtype=torch.long), x, torch.tensor([tokenizer.token_to_id("<eos>")], dtype=torch.long)])
        for x in batch
    ]
    
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))

def model_from_config(config:dict):    
    model_type = config.get('model_type', '').lower()
    
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
    DTYPE = torch.bfloat16 if DEVICE=="cuda" else torch.float16
    USE_FUSED = DEVICE=="cuda"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Training on device: {DEVICE}")
    
    with open("../configs/" + (args.config if args.config.endswith(".yaml") else args.config + ".yaml"), 'r') as file:
        config = yaml.safe_load(file)
        
    seq_len = config["max_position_embeddings"] - 1
    
    tokenizer_name = config.get("tokenizer_name", "tokenizer")
    print(f"{tokenizer_name=}")
    # ensure tokenizer exists
    
    try:
        tokenizer = load_tokenizer(tokenizer_name)
    except FileNotFoundError:
        print(f"Tokenizer file '{tokenizer_name}' not found. Building tokenizer...")
        tokenizer = build_tokenizer(data_name=args.dataset, name=tokenizer_name, vocab_size=config["vocab_size"])
    tokenizer.pad_token = tokenizer.token_to_id("<pad>")
    
    # ensure memmap exists corresponding to the tokenizer
    if not MemmapDataset.exists(dataset_file=args.dataset, tokenizer_name=tokenizer_name):
        print(f"Memmap file '{args.dataset}' not found. Building memmap...")
        MemmapDataset.build_memmap(args.dataset, tokenizer_name)
    
    full_ds = MemmapDataset(dataset_file=args.dataset, tokenizer_name=tokenizer_name, num_tokens=seq_len)
    
    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, _ = random_split(full_ds, [train_size, val_size, test_size])

    collate = partial(collate_fn, max_len=seq_len, tokenizer=tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=True, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model = model_from_config(config).to(DEVICE)
    optim = AdamW(model.parameters(), lr=args.lr, fused=USE_FUSED)
    scaler = torch.amp.GradScaler(enabled=USE_FUSED)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)

    print(f"Training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    model_path = CHECKPOINT_PATH / (args.run_name or "model")
    model_path.mkdir(parents=True, exist_ok=True)

    if args.continue_from:
        checkpoint = torch.load(CHECKPOINT_PATH / args.continue_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for param_group in optim.param_groups: param_group["lr"] = args.lr
        wandb.init(project=config["wandb_project"], id=checkpoint['wandb_id'], resume="must")
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from checkpoint {args.continue_from}, starting at epoch {start_epoch}")
    else:
        wandb.init(
            name=args.run_name,
            project=config["wandb_project"],
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "n_training_examples": len(train_ds),
                "n_validation_examples": len(val_ds),
                "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
                **vars(args),
                **dict(config),
            },
            group=config["model_type"]
        )
        start_epoch = 0
        
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    model_path = CHECKPOINT_PATH / wandb.run.name
    model_path.mkdir(parents=True, exist_ok=True)
    
    # model = torch.compile(model)
    model.train()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs} Training")
        total_train_loss = 0

        for i, batch in enumerate(train_tqdm):
            batch = batch.to(DEVICE)
            x = batch[..., :-1]
            y = batch[..., 1:]
            
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                output = model(x)
                y_hat = output.logits
                loss = criterion(y_hat.reshape(-1, tokenizer.get_vocab_size()), y.reshape(-1))

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            train_loss = loss.detach().cpu().numpy()
            total_train_loss += train_loss
            train_tqdm.set_postfix({"loss": f"{train_loss:.3f}"})

            if i % args.log_interval == 0:
                wandb.log({"train_loss": train_loss}, step=epoch * len(train_dl) + i, commit=True)

        scheduler.step()
        wandb.log({"avg_train_loss": total_train_loss / len(train_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis

        model.eval()
        total_val_loss = 0.0
        total_val_perplex = 0.0
        with torch.no_grad():
            val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs} Validation")
            for val_batch in val_tqdm:
                val_batch = val_batch.to(DEVICE)
                x_val = val_batch[..., :-1]
                y_val = val_batch[..., 1:]

                with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                    output = model(x_val)
                    y_hat = output.logits
                    loss = criterion(y_hat.reshape(-1, tokenizer.get_vocab_size()), y_val.reshape(-1))
                    
                val_loss = loss.detach().cpu().numpy()
                total_val_loss += val_loss
                val_tqdm.set_postfix({"val_loss": f"{val_loss:.3f}"})
                
                total_val_perplex += np.exp(val_loss)
                
        wandb.log({"avg_val_loss": total_val_loss / len(val_dl)}, step=(epoch+1) * len(train_dl)) # to get it on the same axis
        wandb.log({"avg_val_perplexity": total_val_perplex / len(val_dl)}, step=(epoch+1) * len(train_dl))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_train_loss / len(train_dl),
            'wandb_id': wandb.run.id,
            'config_name': args.config
        }, model_path / f"epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset file. Type: .txt")
    parser.add_argument("--config", type=str, required=True, help="Name of the model configuration file. Type: .yaml")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100, help="Number of batches between logging training status to Wandb")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()

    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

