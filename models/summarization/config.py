from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH   = DATA_DIR / "val.jsonl"

# Model Checkpoints
CHECKPOINT_VI = PROJECT_ROOT / "models" / "vit5-summarize-vi" / "best_checkpoint"
CHECKPOINT_EN = PROJECT_ROOT / "models" / "vit5-summarize-en" / "best_checkpoint"

# Training Hyperparameters (SFT)
VI_TRAIN = dict(
    model_name   = "VietAI/vit5-base",
    text_key     = "text_vi",
    summary_key  = "summary_vi",
    prefix       = "summarize: ",
    max_input    = 512,
    max_target   = 256,
    epochs       = 0.5,
    batch_size   = 4,
    lr           = 3e-5,
    weight_decay = 0.01,
    seed         = 42,
    output_dir   = PROJECT_ROOT / "models" / "vit5-summarize-vi",
)

EN_TRAIN = dict(
    model_name   = "VietAI/vit5-base",
    text_key     = "text_en",
    summary_key  = "summary_en",
    prefix       = "summarize: ",
    max_input    = 512,
    max_target   = 128,
    epochs       = 4,
    batch_size   = 4,
    lr           = 3e-5,
    weight_decay = 0.01,
    seed         = 42,
    output_dir   = PROJECT_ROOT / "models" / "vit5-summarize-en",
)

# Hierarchical Summarization Config
HIERARCHICAL = {
    "chunk_tokens": 400,
    "overlap_sents": 2,
    "max_levels": 2,
}

# Legacy Compatibility (nếu các script cũ vẫn dùng tên này)
VI_CHECKPOINT = CHECKPOINT_VI
EN_CHECKPOINT = CHECKPOINT_EN
