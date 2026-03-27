"""
Prompt Fungineer v2: Full fine-tune of Qwen3-0.6B on Claude-generated prompt pairs.

Usage:
    python train.py

Hardware: Apple M4 Pro, 64GB unified memory, MPS backend
Expected time: 1-3 hours
Expected memory: ~8GB
"""

import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import numpy as np
from pathlib import Path
import wandb


# ── Config ─────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "data/training_data.jsonl"
OUTPUT_DIR = "checkpoints/prompt-fungineer-v2"
VAL_SPLIT = 0.05
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 4  # effective batch size = 32
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
SAVE_STEPS = 500
EVAL_STEPS = 250
SEED = 42
# ────────────────────────────────────────────────────────


class PromptDataset(Dataset):
    """Format: <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"""

    def __init__(self, pairs, tokenizer, max_len=512):
        self.examples = []
        self.labels = []

        for pair in pairs:
            # Format as chat
            text = (
                f"<|im_start|>user\n"
                f"Expand this into a detailed image generation prompt: {pair['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{pair['output']}<|im_end|>"
            )

            tokens = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            # Create labels: mask the user portion (only train on assistant output)
            labels = input_ids.clone()
            # Find where assistant response starts
            assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            text_ids = input_ids.tolist()

            # Find the assistant marker position
            marker_pos = -1
            for i in range(len(text_ids) - len(assistant_marker)):
                if text_ids[i:i + len(assistant_marker)] == assistant_marker:
                    marker_pos = i + len(assistant_marker)
                    break

            # Mask everything before assistant response
            if marker_pos > 0:
                labels[:marker_pos] = -100

            # Mask padding
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(path):
    """Load JSONL training data."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def evaluate(model, val_loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Count non-masked tokens for proper averaging
            non_masked = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * non_masked
            total_tokens += non_masked

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)

    # Initialize W&B
    wandb.init(
        project="prompt-fungineer-v2",
        name=f"qwen3-0.6B-full-finetune",
        config={
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_len": MAX_SEQ_LEN,
            "warmup_ratio": WARMUP_RATIO,
            "grad_accum": GRAD_ACCUM,
            "device": str(device),
            "seed": SEED,
        },
        tags=["full-finetune", "qwen3-0.6B", "prompt-engineering", "distillation"],
    )

    # Load tokenizer and model
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    # Use float32 on MPS — BF16 training on MPS can hang
    model_dtype = torch.float32 if device.type == "mps" else torch.bfloat16
    print(f"  Using dtype: {model_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.train()
    print(f"  Model on {device}", flush=True)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable_count:,} trainable", flush=True)
    print(f"Model memory: {param_count * 2 / 1e9:.2f} GB (BF16)", flush=True)

    # Load and split data
    print(f"\nLoading data from {DATA_PATH}...", flush=True)
    all_pairs = load_data(DATA_PATH)
    np.random.shuffle(all_pairs)

    val_size = int(len(all_pairs) * VAL_SPLIT)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}", flush=True)

    # Create datasets
    print("Tokenizing train set...", flush=True)
    train_dataset = PromptDataset(train_pairs, tokenizer, MAX_SEQ_LEN)
    print("Tokenizing val set...", flush=True)
    val_dataset = PromptDataset(val_pairs, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    print(f"\nTraining config:", flush=True)
    print(f"  Epochs: {EPOCHS}", flush=True)
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})", flush=True)
    print(f"  Steps per epoch: {len(train_loader)}", flush=True)
    print(f"  Total optimization steps: {total_steps}", flush=True)
    print(f"  Warmup steps: {warmup_steps}", flush=True)
    print(f"  Learning rate: {LR}", flush=True)
    print(f"  Max seq length: {MAX_SEQ_LEN}", flush=True)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    log_file = open(os.path.join(OUTPUT_DIR, "training_log.jsonl"), "w")

    print(f"\n{'='*60}", flush=True)
    print(f"Starting training...", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_tokens = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            non_masked = (labels != -100).sum().item()
            epoch_loss += outputs.loss.item() * non_masked
            epoch_tokens += non_masked

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % 25 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + (step / len(train_loader)),
                        "train/elapsed_min": elapsed / 60,
                    }, step=global_step)
                    print(f"  Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"Time: {elapsed/60:.1f}m", flush=True)

                # Evaluate
                if global_step % EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader, device)
                    wandb.log({
                        "val/loss": val_loss,
                        "val/best_loss": min(best_val_loss, val_loss),
                    }, step=global_step)
                    print(f"  >>> Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)

                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss / epoch_tokens if epoch_tokens > 0 else 0,
                        "val_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_min": (time.time() - start_time) / 60,
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(OUTPUT_DIR, "best")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"  >>> Saved best model (val_loss={val_loss:.4f})", flush=True)

                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"step_{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"  >>> Saved checkpoint: {ckpt_path}", flush=True)

        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time

        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch+1}/{EPOCHS}", flush=True)
        print(f"  Train loss: {avg_epoch_loss:.4f}", flush=True)
        print(f"  Val loss:   {val_loss:.4f}", flush=True)
        print(f"  Best val:   {best_val_loss:.4f}", flush=True)
        print(f"  Time:       {elapsed/60:.1f}m", flush=True)
        print(f"{'='*60}\n", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    log_file.close()

    # Final save
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete!", flush=True)
    print(f"  Total time: {total_time/60:.1f} minutes", flush=True)
    print(f"  Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"  Best model: {os.path.join(OUTPUT_DIR, 'best')}", flush=True)
    print(f"  Final model: {final_path}", flush=True)

    wandb.log({
        "final/best_val_loss": best_val_loss,
        "final/total_time_min": total_time / 60,
        "final/total_steps": global_step,
    })
    wandb.finish()


if __name__ == "__main__":
    train()
