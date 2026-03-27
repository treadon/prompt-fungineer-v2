"""
Prompt Fungineer v2: Full fine-tune using HuggingFace Trainer.
Uses Trainer for proper MPS/device handling.
"""

import os
import json
import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "data/training_data.jsonl"
OUTPUT_DIR = "checkpoints/prompt-fungineer-v2"
SEED = 42


def load_and_format_data(path, tokenizer, max_len=384):
    """Load data and tokenize."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))

    def format_pair(pair):
        return (
            f"<|im_start|>user\n"
            f"Expand this into a detailed image generation prompt: {pair['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{pair['output']}<|im_end|>"
        )

    texts = [format_pair(p) for p in pairs]

    # Tokenize
    tokenized = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # Create labels (mask user portion)
    assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    all_input_ids = []
    all_labels = []

    for ids in tokenized["input_ids"]:
        labels = list(ids)

        # Find assistant marker
        marker_pos = -1
        for i in range(len(ids) - len(assistant_marker)):
            if ids[i:i + len(assistant_marker)] == assistant_marker:
                marker_pos = i + len(assistant_marker)
                break

        if marker_pos > 0:
            for i in range(marker_pos):
                labels[i] = -100

        all_input_ids.append(ids)
        all_labels.append(labels)

    ds = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": tokenized["attention_mask"],
    })

    return ds


def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}", flush=True)

    print("Tokenizing data...", flush=True)
    dataset = load_and_format_data(DATA_PATH, tokenizer)

    # Split
    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    # Custom data collator that handles labels properly
    def collate_fn(examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        input_ids = []
        attention_mask = []
        labels = []
        for e in examples:
            pad_len = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(e["attention_mask"] + [0] * pad_len)
            labels.append(e["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        max_grad_norm=1.0,
        seed=SEED,
        report_to="wandb",
        run_name="qwen3-0.6B-full-finetune",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # W&B init
    wandb.init(
        project="prompt-fungineer-v2",
        name="qwen3-0.6B-full-finetune",
        config={
            "model": MODEL_NAME,
            "params": param_count,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        tags=["full-finetune", "qwen3-0.6B", "prompt-engineering", "distillation"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    print("Starting training...", flush=True)
    trainer.train()

    print("Saving best model...", flush=True)
    trainer.save_model(os.path.join(OUTPUT_DIR, "best"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))

    # Evaluate
    metrics = trainer.evaluate()
    print(f"\nFinal eval: {metrics}", flush=True)

    wandb.finish()
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
