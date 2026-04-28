import os
import sys
import json
import gc
import re
import inspect
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# BẬT CHỐNG PHÂN MẢNH VRAM KHI PHÙ HỢP.
# Trên một số build Windows sẽ cảnh báo "expandable_segments not supported".
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
try:
    from rouge_score import rouge_scorer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge_score"])
    from rouge_score import rouge_scorer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    import config
except ImportError:
    config = None


# Tránh lỗi UnicodeEncodeError trên Windows console (cp1258/cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for child in output_dir.glob("checkpoint-*"):
        if not child.is_dir():
            continue
        suffix = child.name.replace("checkpoint-", "")
        if suffix.isdigit():
            candidates.append((int(suffix), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def read_completed_epoch(checkpoint_dir: Path) -> float:
    state_file = checkpoint_dir / "trainer_state.json"
    if not state_file.exists():
        return 0.0
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
        return float(state.get("epoch", 0.0) or 0.0)
    except Exception:
        return 0.0


def trainer_state_to_dict(state) -> dict:
    if hasattr(state, "to_dict"):
        return dict(state.to_dict())
    if is_dataclass(state):
        return asdict(state)
    if hasattr(state, "__dict__"):
        return dict(state.__dict__)
    return {}


def checkpoint_has_model_weights(checkpoint_dir: Path) -> bool:
    possible_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    return any((checkpoint_dir / name).exists() for name in possible_files)


class JsonlSummarizeDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        text_key: str,
        summary_key: str,
        max_input: int = 1024,
        max_target: int = 128,
        prefix: str = "",
        text_clean_mode: str = "keep",
        allow_field_fallback: bool = True,
    ):
        self.rows = []
        self.source_paths = [Path(p) for p in ([path] if isinstance(path, Path) else path)]
        paths = [path] if isinstance(path, Path) else path
        for p in paths:
            with open(p, encoding="utf-8") as f:
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
        self.text_clean_mode = text_clean_mode
        self.allow_field_fallback = allow_field_fallback
        self.fallback_stats = {"text": 0, "summary": 0}

        def _pick_value(row: dict, wanted_key: str, kind: str) -> str:
            direct = str(row.get(wanted_key, "")).strip()
            if direct:
                return direct
            if not self.allow_field_fallback:
                return ""

            fallback_key = None
            if wanted_key == "text_en":
                fallback_key = "text_vi"
            elif wanted_key == "summary_en":
                fallback_key = "summary_vi"

            if fallback_key is None:
                return ""

            fallback_value = str(row.get(fallback_key, "")).strip()
            if fallback_value:
                self.fallback_stats[kind] += 1
                return fallback_value
            return ""

        prepared_rows: list[dict] = []
        for row in self.rows:
            text_val = _pick_value(row, self.text_key, "text")
            summary_val = _pick_value(row, self.summary_key, "summary")
            if not text_val or not summary_val:
                continue
            row_copy = dict(row)
            row_copy["_resolved_text"] = text_val
            row_copy["_resolved_summary"] = summary_val
            prepared_rows.append(row_copy)

        raw_count = len(self.rows)
        self.rows = prepared_rows
        dropped = raw_count - len(self.rows)
        print(
            f"--- Đã tải {raw_count} mẫu từ {len(paths)} file dữ liệu | "
            f"hợp lệ cho ({self.text_key}, {self.summary_key}): {len(self.rows)} | bỏ: {dropped} ---"
        )
        if self.fallback_stats["text"] or self.fallback_stats["summary"]:
            print(
                "--- CẢNH BÁO: đang fallback field train --- "
                f"text_fallback={self.fallback_stats['text']}, "
                f"summary_fallback={self.fallback_stats['summary']} "
                "(vd text_en<-text_vi, summary_en<-summary_vi)."
            )
        if not self.rows:
            joined_paths = ", ".join(str(p) for p in self.source_paths)
            raise ValueError(
                "Khong co mau hop le cho train.\n"
                f"- Yeu cau truong: {self.text_key}, {self.summary_key}\n"
                f"- File da doc: {joined_paths}\n"
                "Hay tao/chi dinh dung bo du lieu truoc khi train (vi du data translated cho nhanh EN)."
            )

    def _clean_text(self, text: str) -> str:
        if self.text_clean_mode == "keep":
            return text

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return text.strip()

        marker_patterns = (
            r"^Từ khóa bắt buộc\s*:",
            r"^Tu khoa bat buoc\s*:",
            r"^Tiêu đề\s*:",
            r"^Tieu de\s*:",
            r"^Tóm tắt nhanh\s*:",
            r"^Tom tat nhanh\s*:",
            r"^Nội dung\s*:",
            r"^Noi dung\s*:",
        )

        if self.text_clean_mode == "strip_headers":
            kept = []
            for ln in lines:
                if any(re.match(pat, ln, flags=re.IGNORECASE) for pat in marker_patterns):
                    continue
                kept.append(ln)
            return " ".join(kept).strip()

        if self.text_clean_mode == "content_only":
            joined = "\n".join(lines)
            m = re.search(r"(Nội dung|Noi dung)\s*:\s*", joined, flags=re.IGNORECASE)
            if m:
                return joined[m.end():].strip()

            kept = []
            for ln in lines:
                if any(re.match(pat, ln, flags=re.IGNORECASE) for pat in marker_patterns):
                    continue
                kept.append(ln)
            return " ".join(kept).strip()

        return text.strip()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text_content = self._clean_text(str(row.get("_resolved_text", "")))
        
        # Nếu đoạn text đã có prefix giống như "Từ khóa bắt buộc" thì không cần thêm rác.
        if "Từ khóa bắt buộc:" in text_content or "Nội dung:" in text_content:
            text = text_content
        else:
            text = self.prefix + text_content
            
        summary = str(row.get("_resolved_summary", ""))
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


def build_compute_metrics_fn(tokenizer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge1, rouge2, rougeL = [], [], []
        for p, l in zip(decoded_preds, decoded_labels):
            scores = scorer.score(l.strip(), p.strip())
            rouge1.append(scores['rouge1'].fmeasure)
            rouge2.append(scores['rouge2'].fmeasure)
            rougeL.append(scores['rougeL'].fmeasure)
            
        return {
            "rouge1": np.mean(rouge1) * 100,
            "rouge2": np.mean(rouge2) * 100,
            "rougeL": np.mean(rougeL) * 100,
        }
    return compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["vi", "en"], required=True)
    p.add_argument(
        "--train_path",
        type=Path,
        nargs="+",
        default=config.TRAIN_PATH if config else [PROJECT_ROOT / "data" / "dataset_translated.jsonl"],
    )
    p.add_argument(
        "--val_path",
        type=Path,
        default=config.VAL_PATH if config else PROJECT_ROOT / "data" / "processed" / "val.jsonl",
    )
    p.add_argument("--model_name", default=config.MODEL_NAME if config else "VietAI/vit5-base")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--epochs", type=float, default=config.EPOCHS if config else 4.0)
    p.add_argument("--resume_additional_epochs", type=float, default=config.RESUME_ADDITIONAL_EPOCHS if config else 1.0)
    p.add_argument(
        "--resume_mode",
        choices=["weights_only", "stateful"],
        default="weights_only",
        help="weights_only: nap trong so tu checkpoint roi train moi; stateful: resume full trainer state.",
    )
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE if config else 2)
    p.add_argument("--lr", type=float, default=config.LR if config else 3e-5)
    p.add_argument("--max_input", type=int, default=config.MAX_INPUT_LENGTH_TRAIN if config else 1024)
    p.add_argument("--max_target", type=int, default=config.MAX_TARGET_LENGTH_TRAIN if config else 128)
    p.add_argument("--seed", type=int, default=config.SEED if config else 42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    p.add_argument("--resume", action="store_true", help="Chạy tiếp từ checkpoint mới nhất")
    p.add_argument("--save_strategy", choices=["epoch", "steps"], default="epoch")
    p.add_argument("--eval_strategy", choices=["epoch", "steps"], default="epoch")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--disable_eval", action="store_true", help="Bo qua val/evaluate de train nhanh hon")
    p.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Luu model dang safetensors. Mac dinh tat de tranh loi non-contiguous tensor khi save.",
    )
    p.add_argument(
        "--text_clean_mode",
        choices=["keep", "strip_headers", "content_only"],
        default="keep",
        help="Cach lam sach text train: keep (giu nguyen), strip_headers (bo dong metadata), content_only (lay phan sau 'Noi dung:').",
    )
    p.add_argument(
        "--strict_fields",
        action="store_true",
        help="Tat fallback field (khong cho text_en<-text_vi).",
    )
    args = p.parse_args()
    if not args.disable_eval and args.eval_strategy != args.save_strategy:
        print(
            "--- eval_strategy va save_strategy khac nhau, "
            f"tu dong dong bo save_strategy={args.eval_strategy} de hop le voi load_best_model_at_end ---"
        )
        args.save_strategy = args.eval_strategy

    if args.lang == "vi":
        text_key, summary_key = "text_vi", "summary_vi"
    else:
        text_key, summary_key = "text_en", "summary_en"

    out = args.output_dir or (PROJECT_ROOT / "models" / f"vit5-summarize-{args.lang}")
    out.mkdir(parents=True, exist_ok=True)

    resume_checkpoint: Path | None = None
    target_num_train_epochs = args.epochs
    model_init_path = args.model_name
    resume_from_checkpoint: str | None = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(out)
        if resume_checkpoint is None:
            print("--- Khong tim thay checkpoint-* de resume. Chuyen sang train moi. ---")
        else:
            if not checkpoint_has_model_weights(resume_checkpoint):
                print(
                    f"--- Checkpoint {resume_checkpoint} khong co file trong so hop le "
                    "(pytorch_model.bin/model.safetensors). Bo qua resume de tranh crash. ---"
                )
                resume_checkpoint = None

        if resume_checkpoint is not None:
            completed_epoch = read_completed_epoch(resume_checkpoint)
            if args.resume_mode == "stateful":
                resume_from_checkpoint = str(resume_checkpoint)
                target_num_train_epochs = max(
                    args.epochs,
                    completed_epoch + args.resume_additional_epochs,
                )
                print(
                    f"--- Resume STATEFUL tu: {resume_checkpoint} | completed_epoch={completed_epoch:.3f} "
                    f"-> target_num_train_epochs={target_num_train_epochs:.3f} ---"
                )
            else:
                model_init_path = str(resume_checkpoint)
                target_num_train_epochs = max(0.1, args.resume_additional_epochs)
                print(
                    f"--- Resume WEIGHTS_ONLY tu: {resume_checkpoint} | completed_epoch={completed_epoch:.3f} "
                    f"-> train_them={target_num_train_epochs:.3f} epoch ---"
                )
    if args.resume and args.resume_mode == "stateful":
        # PyTorch 2.6+ doi mac dinh torch.load(weights_only=True),
        # co the lam Trainer resume stateful bi vo (optimizer/rng state).
        # Bat env nay de giu hanh vi tuong thich khi resume tu checkpoint noi bo.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # 1. Giới hạn bộ nhớ GPU (Để lại một ít cho hệ điều hành và tránh crash)
    if not args.cpu and torch.cuda.is_available():
        # Hạ xuống 85% để GPU ổn định hơn trên Windows
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        print(f"--- GPU 4GB: Đã cấu hình ổn định (85% VRAM, Adafactor) ---")
    elif args.cpu:
        print(f"--- Chạy bằng CPU (Lưu ý: Tốc độ sẽ rất chậm) ---")

    # 2. Chỉ bật CUDA_LAUNCH_BLOCKING khi cần debug sâu
    if os.getenv("CUDA_DEBUG_BLOCKING", "").strip() == "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    def load_tokenizer_with_fallback(primary_path: str, fallback_path: str):
        attempts = [
            (primary_path, True, "primary-fast"),
            (primary_path, False, "primary-slow"),
        ]
        if str(fallback_path) != str(primary_path):
            attempts.extend(
                [
                    (fallback_path, False, "fallback-slow"),
                    (fallback_path, True, "fallback-fast"),
                ]
            )

        errors: list[str] = []
        for path, use_fast, label in attempts:
            try:
                tok = AutoTokenizer.from_pretrained(path, use_fast=use_fast)
                if label != "primary-fast":
                    print(f"--- Tokenizer load fallback thanh cong: {label} ({path}) ---")
                return tok
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        raise RuntimeError("Khong the load tokenizer.\n" + "\n".join(errors))

    def load_model_with_fallback(primary_path: str, fallback_path: str):
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(primary_path)
        except Exception as exc:
            if str(fallback_path) == str(primary_path):
                raise
            print(f"--- Canh bao: load model tu {primary_path} that bai ({exc}). Fallback sang {fallback_path}. ---")
            return AutoModelForSeq2SeqLM.from_pretrained(fallback_path)

    tokenizer = load_tokenizer_with_fallback(str(model_init_path), str(args.model_name))
    model = load_model_with_fallback(str(model_init_path), str(args.model_name))
    # Tranh warning "use_cache=True is incompatible with gradient checkpointing".
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_ds = JsonlSummarizeDataset(
        args.train_path,
        tokenizer,
        text_key,
        summary_key,
        max_input=args.max_input,
        max_target=args.max_target,
        text_clean_mode=args.text_clean_mode,
        allow_field_fallback=not args.strict_fields,
    )
    val_ds = None
    if not args.disable_eval:
        val_ds = JsonlSummarizeDataset(
            args.val_path,
            tokenizer,
            text_key,
            summary_key,
            max_input=args.max_input,
            max_target=args.max_target,
            text_clean_mode=args.text_clean_mode,
            allow_field_fallback=not args.strict_fields,
        )
    else:
        print("--- Disable eval: se bo qua val dataset va evaluate() de uu tien toc do ---")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 3. Cấu hình training siêu tiết kiệm VRAM với Adafactor
    train_args_kwargs = {
        "output_dir": str(out),
        "num_train_epochs": target_num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "eval_strategy": "no" if args.disable_eval else args.eval_strategy,
        "save_strategy": args.save_strategy,
        "load_best_model_at_end": (not args.disable_eval),
        "save_total_limit": 2,
        "logging_steps": 10,
        
        # --- CÁC THAY ĐỔI QUAN TRỌNG ĐỂ CỨU VRAM ---
        "predict_with_generate": True,
        "generation_num_beams": 1,
        "generation_max_length": args.max_target,
        
        "fp16": False,                  # Tắt fp16 trên Windows để ổn định nhất cho T5
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
        "optim": "adafactor",
        "save_safetensors": args.save_safetensors,
        "report_to": "none",
        "no_cuda": args.cpu,
        "eval_accumulation_steps": 1,
    }
    # Tranh warning use_reentrant cua torch.checkpoint tren cac ban torch moi.
    if "gradient_checkpointing_kwargs" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
        train_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if not args.disable_eval:
        train_args_kwargs["metric_for_best_model"] = "eval_rougeL"
        train_args_kwargs["greater_is_better"] = True
    if (not args.disable_eval) and args.eval_strategy == "steps":
        train_args_kwargs["eval_steps"] = args.eval_steps
    if args.save_strategy == "steps":
        train_args_kwargs["save_steps"] = args.save_steps

    training_args = Seq2SeqTrainingArguments(**train_args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=None if args.disable_eval else build_compute_metrics_fn(tokenizer),
    )

    # Chi resume khi co --resume.
    # Neu nguoi dung bat safetensors va gap loi non-contiguous tensor, tu dong fallback qua .bin.
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except ValueError as exc:
        err = str(exc)
        if args.save_safetensors and "non contiguous tensor" in err:
            print(
                "--- Phat hien loi save safetensors voi non-contiguous tensor. "
                "Tu dong fallback sang luu .bin va train lai tu dau lan chay nay. ---"
            )
            trainer.args.save_safetensors = False
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            raise
    best_dir = out / "best_checkpoint"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Chuẩn hóa trainer_state trong best_checkpoint để dễ theo dõi lần train gần nhất
    state_dict = trainer_state_to_dict(trainer.state)
    state_dict["best_model_checkpoint_source"] = state_dict.get("best_model_checkpoint")
    state_dict["best_model_checkpoint"] = str(best_dir)
    state_dict["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(best_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)
    
    # --- DỌN SẠCH VRAM RÁC TỪ QUÁ TRÌNH TRAIN TRƯỚC KHI ĐÁNH GIÁ ---
    if not args.cpu and torch.cuda.is_available():
        print("\nĐang dọn dẹp VRAM trước khi tính điểm Evaluate...")
        gc.collect()
        torch.cuda.empty_cache()

    if not args.disable_eval:
        metrics = trainer.evaluate()
        with open(out / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
