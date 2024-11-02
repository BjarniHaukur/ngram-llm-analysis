import math
import argparse
from pathlib import Path
from functools import partial

import yaml
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from utils.ngram import NGramTrie
from utils.model import model_from_config
from utils.sample import stream_generation
from utils.dataset import MemmapDataset
from utils.tokenizer import load_tokenizer, color_text_html

torch.random.manual_seed(1337)
if torch.cuda.is_available(): torch.cuda.manual_seed(1337)

CHECKPOINT_PATH = Path("../checkpoints/models")


def cycle(dl:DataLoader):  # itertools.cycle can causes memory leak with computationally heavy tasks
    while True:
        yield from dl

def main(args):
    print("Loading trie...")
    trie = NGramTrie(args.ngram_file)
    print("Trie loaded.")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE=="cuda" else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    print(f"Training on device: {DEVICE}")
    print(f"Training in dtype: {DTYPE}")
    
    with open("../configs/" + (args.config if args.config.endswith(".yaml") else args.config + ".yaml"), "r") as file:
        config = yaml.safe_load(file)
        
    seq_len = config["max_position_embeddings"] - 1  # account for <bos> and <eos>
    tokenizer_name = config.get("tokenizer_name", "tokenizer")
    print(f"Using tokenizer: {tokenizer_name}")
    
    tokenizer = load_tokenizer(tokenizer_name)
    tokenizer.pad_token = tokenizer.token_to_id("<pad>")
    full_ds = MemmapDataset(dataset_file=args.dataset, tokenizer_name=tokenizer_name, num_tokens=seq_len)
    
    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, _ = random_split(full_ds, [train_size, val_size, test_size])

    CURRENT_STEP = 0
    STEPS_PER_EPOCH_TRAIN = len(train_ds) // args.batch_size  # Drop last is True
    TOTAL_STEPS = STEPS_PER_EPOCH_TRAIN * args.epochs
    VAL_INTERVAL = STEPS_PER_EPOCH_TRAIN // 100
    NUM_VAL_STEPS = STEPS_PER_EPOCH_TRAIN // VAL_INTERVAL
    WARMUP_STEPS = int(0.03 * TOTAL_STEPS)

    CHECKPOINT_INTERVAL = TOTAL_STEPS // 10

    def get_lr(step):
        if step < WARMUP_STEPS:  # 1) linear warmup for WARMUP_STEPS steps
            return args.max_lr * (step + 1) / WARMUP_STEPS
        if step > TOTAL_STEPS:  # 2) if it > TOTAL_STEPS, return min learning rate
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return args.min_lr + coeff * (args.max_lr - args.min_lr)

    # make the dataloaders cycle on end of dataset, so we can keep calling next() on them
    train_dl = cycle(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True))
    val_dl = cycle(DataLoader(val_ds, batch_size=args.batch_size, prefetch_factor=args.prefetch_factor, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True))

    model = model_from_config(config).to(DEVICE)
    model = torch.compile(model, backend="aot_eager")
    optim = AdamW(model.parameters(), lr=args.max_lr, fused=DEVICE=="cuda")

    print(f"Training model with {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f}M parameters")

    model_path = CHECKPOINT_PATH / (args.run_name or "model")
    model_path.mkdir(parents=True, exist_ok=True)

    if args.continue_from:
        checkpoint = torch.load(model_path / args.continue_from, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optim.param_groups: param_group["lr"] = get_lr(CURRENT_STEP)
        wandb.init(project=config["wandb_project"], id=checkpoint["wandb_id"], resume="must")
        CURRENT_STEP = checkpoint["current_step"]
        print(f"Resuming training from checkpoint {args.continue_from}, starting at step {CURRENT_STEP}")
    else:
        wandb.init(
            name=args.run_name,
            project=config["wandb_project"],
            config={
                "learning_rate": args.max_lr,
                "epochs": args.epochs,
                "n_training_examples": len(train_ds),
                "n_validation_examples": len(val_ds),
                "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
                **vars(args),
                **dict(config),
            },
            group=config["model_type"]
        )
        
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    model_path = CHECKPOINT_PATH / wandb.run.name
    model_path.mkdir(parents=True, exist_ok=True)
    
    model.train()

    train_loss = float("inf")
    val_loss = float("inf")

    step_tqdm = tqdm(range(CURRENT_STEP, TOTAL_STEPS), desc="Training...", initial=CURRENT_STEP, total=TOTAL_STEPS)
    for step in step_tqdm:
        batch = next(train_dl).to(DEVICE, non_blocking=True)
        x = batch[..., :-1]
        y = batch[..., 1:]

        lr = get_lr(step)
        wandb.log({"learning_rate": lr}, step=step)
        for param_group in optim.param_groups: param_group["lr"] = lr
            
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
            output = model(x)
            y_hat = output.logits
            loss = criterion(y_hat.reshape(-1, tokenizer.get_vocab_size()), y.reshape(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss = loss.detach().cpu().numpy()
        perplexity = np.exp(train_loss)
        wandb.log({
            "train_loss": train_loss,
            "train_perplexity": perplexity
        }, step=step)
        step_tqdm.set_postfix({"train_loss": f"{train_loss:.3f}", "val_loss": f"{val_loss:.3f}"})

        if step % VAL_INTERVAL == 0 and step != 0:
            model.eval()
            step_tqdm.set_description("Validating...")
            with torch.no_grad():
                total_val_loss = 0
                total_val_perplexity = 0
                for val_step in range(NUM_VAL_STEPS):
                    val_batch = next(val_dl).to(DEVICE, non_blocking=True)
                    x_val = val_batch[..., :-1]
                    y_val = val_batch[..., 1:]

                    with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE=="cuda"):
                        output = model(x_val)
                        y_hat = output.logits
                        loss = criterion(y_hat.reshape(-1, tokenizer.get_vocab_size()), y_val.reshape(-1))
                    
                    val_loss = loss.detach().cpu().numpy()
                    perplexity = np.exp(val_loss)
                    total_val_loss += val_loss
                    total_val_perplexity += perplexity
                    step_tqdm.set_postfix({"train_loss": f"{train_loss:.3f}", "val_loss": f"{val_loss:.3f}"})

                    if val_step > 0: continue  # this is getting too nested (only perform once)

                    trie.log_metrics_async(x_val.numpy(), y_hat.numpy(), step)

                    continuation = "".join(list(stream_generation(model, tokenizer, max_length=50, temperature=0.7)))
                    wandb.log({"generated_text": wandb.Html(color_text_html(tokenizer, continuation))}, step=step)
                                
                wandb.log({
                    "val_loss": total_val_loss / NUM_VAL_STEPS,
                    "val_perplexity": total_val_perplexity / NUM_VAL_STEPS
                }, step=step)

            model.train()
            step_tqdm.set_description("Training...")


        if (step % CHECKPOINT_INTERVAL == 0 or step == TOTAL_STEPS - 1) and step != 0:
            torch.save({
                "current_step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "wandb_id": wandb.run.id,
                "config_name": args.config
            }, model_path / f"{step + 1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset file. Type: .txt")
    parser.add_argument("--config", type=str, required=True, help="Name of the model configuration file. Type: .yaml")
    parser.add_argument("--ngram_file", type=str, default="ngram", help="Path to ngram file to use for smoothed trie.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--log_interval", type=int, default=100, help="Number of batches between logging training status to Wandb")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()

    main(args)