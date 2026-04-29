"""
models/summarization/train_en.py
-------------------------------
Script chuyên dụng để fine-tune ViT5 cho tiếng Anh.
"""

import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.summarization.config import EN_TRAIN, TRAIN_PATH, VAL_PATH
from models.summarization.dataset import JsonlSummarizeDataset

def train():
    cfg = EN_TRAIN
    print(f"Starting training EN with model: {cfg['model_name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_name'])
    
    train_ds = JsonlSummarizeDataset(
        TRAIN_PATH,
        tokenizer,
        text_key=cfg['text_key'],
        summary_key=cfg['summary_key'],
        max_input=cfg['max_input'],
        max_target=cfg['max_target'],
        prefix=cfg['prefix']
    )
    
    val_ds = JsonlSummarizeDataset(
        VAL_PATH,
        tokenizer,
        text_key=cfg['text_key'],
        summary_key=cfg['summary_key'],
        max_input=cfg['max_input'],
        max_target=cfg['max_target'],
        prefix=cfg['prefix']
    )
    
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cfg['output_dir']),
        num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        learning_rate=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Lưu checkpoint tốt nhất
    best_path = cfg['output_dir'] / "best_checkpoint"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"Training EN complete. Best model saved to: {best_path}")

if __name__ == "__main__":
    train()
