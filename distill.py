"""
Distillation: collect run-level data from logs, train a student LoRA.
Requires: torch, transformers, peft, accelerate, datasets.
"""

import json
import os


def collect_data(log_path: str):
    """Read JSONL run-level log and return a HuggingFace Dataset with input/target columns."""
    from datasets import Dataset
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            inp = record.get("input") or record.get("prompt", "")
            tgt = record.get("target") or record.get("answer", "")
            rows.append({"input": inp, "target": tgt})
    return Dataset.from_list(rows)


def train_student(
    model_id: str,
    log_path: str,
    output_dir: str,
    max_steps: int = 100,
    batch_size: int = 2,
    lr: float = 5e-5,
    max_length: int = 512,
) -> None:
    """Load base model + LoRA, SFT on distillation data, save adapter."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05,
    )
    model = get_peft_model(base, lora_config)

    dataset = collect_data(log_path)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; log_path must contain at least one run record")

    def _tokenize(examples):
        texts = [
            f"Input: {inp or ''}\nTarget: {tgt or ''}"
            for inp, tgt in zip(examples["input"], examples["target"])
        ]
        out = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=max(1, max_steps // 5),
        save_strategy="no",
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train student LoRA from distillation logs")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", default="student_adapter")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    train_student(args.model, args.log_path, args.output_dir, max_steps=args.max_steps)
