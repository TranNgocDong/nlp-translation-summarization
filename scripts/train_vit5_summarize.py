import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class JsonlSummarizeDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        text_key: str,
        summary_key: str,
        max_input: int = 512,
        max_target: int = 128,
        prefix: str = "summarize: ",
    ):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.summary_key = summary_key
        self.max_input = max_input
        self.max_target = max_target
        self.prefix = prefix

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = self.prefix + str(row[self.text_key])
        summary = str(row[self.summary_key])
        enc = self.tokenizer(
            text,
            max_length=self.max_input,
            truncation=True,
            padding=False,
        )
        tgt = self.tokenizer(
            summary,
            max_length=self.max_target,
            truncation=True,
            padding=False,
        )
        enc["labels"] = tgt["input_ids"]
        return enc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["vi", "en"], required=True)
    p.add_argument(
        "--train_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train.jsonl",
    )
    p.add_argument(
        "--val_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "val.jsonl",
    )
    p.add_argument("--model_name", default="VietAI/vit5-base")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_input", type=int, default=512)
    p.add_argument("--max_target", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.lang == "vi":
        text_key, summary_key = "text_vi", "summary_vi"
    else:
        text_key, summary_key = "text_en", "summary_en"

    out = args.output_dir or (PROJECT_ROOT / "models" / f"vit5-summarize-{args.lang}")
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = JsonlSummarizeDataset(
        args.train_path,
        tokenizer,
        text_key,
        summary_key,
        max_input=args.max_input,
        max_target=args.max_target,
    )
    val_ds = JsonlSummarizeDataset(
        args.val_path,
        tokenizer,
        text_key,
        summary_key,
        max_input=args.max_input,
        max_target=args.max_target,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        predict_with_generate=False,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(out / "best_checkpoint"))
    tokenizer.save_pretrained(str(out / "best_checkpoint"))
    metrics = trainer.evaluate()
    with open(out / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
