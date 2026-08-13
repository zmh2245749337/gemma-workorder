"""Fine-tune Gemma for the constrained Gemma-WorkOrder JSON contract.

Run this on a Colab GPU.  It trains only LoRA adapters; model weights and
dataset remain local to the notebook runtime and are not committed to Git.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gemma_eval.workorder_data import canonical_json, load_jsonl, workorder_prompt


def encode_example(tokenizer, row: dict, max_length: int) -> dict[str, list[int]]:
    prompt_encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": workorder_prompt(row["input_text"])}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Transformers 4 returns a list here; newer releases may return a
    # BatchEncoding.  Normalise both shapes before concatenating labels.
    prompt_ids = prompt_encoded["input_ids"] if hasattr(prompt_encoded, "get") and "input_ids" in prompt_encoded else prompt_encoded
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    answer_ids = tokenizer(canonical_json(row["output"]), add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = (prompt_ids + answer_ids + ([eos] if eos is not None else []))[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids + ([eos] if eos is not None else []))[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class WorkOrderDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rows: list[dict], max_length: int):
        self.items = [encode_example(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for the Gemma-WorkOrder controlled dataset")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--train", type=Path, default=Path("data/workorder/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/workorder/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/workorder_qlora_adapter"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows, validation_rows = load_jsonl(args.train), load_jsonl(args.validation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, token=True, device_map="auto", quantization_config=quantization)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    train_dataset = WorkOrderDataset(tokenizer, train_rows, args.max_length)
    validation_dataset = WorkOrderDataset(tokenizer, validation_rows, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "scope": "Synthetic controlled work-order extraction; not enterprise production data.",
        "base_model": args.model_id,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "training_args": vars(args),
        "train_metrics": train_result.metrics,
        "validation_metrics": metrics,
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
