import os
from pathlib import Path

# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================
# 1. PATH CONFIGURATION (ĐƯỜNG DẪN)
# ==========================================
TRAIN_PATH = [PROJECT_ROOT / "data" / "dataset_translated.jsonl"]
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.jsonl"
OUTPUT_DIR_VI = PROJECT_ROOT / "models" / "vit5-summarize-vi"
OUTPUT_DIR_EN = PROJECT_ROOT / "models" / "vit5-summarize-en"

# ==========================================
# 2. TRAINING CONFIGURATION (HUẤN LUYỆN)
# ==========================================
MODEL_NAME = "VietAI/vit5-base"
EPOCHS = 4.0
RESUME_ADDITIONAL_EPOCHS = 1.0
BATCH_SIZE = 2
LR = 3e-5
SEED = 42
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
SAVE_STEPS = 100
EVAL_STEPS = 100
MAX_INPUT_LENGTH_TRAIN = 1024
MAX_TARGET_LENGTH_TRAIN = 128

# ==========================================
# 3. INFERENCE CONFIGURATION (SUY LUẬN/TẠO)
# ==========================================
MAX_INPUT_LENGTH_INFERENCE = 1024
MAX_NEW_TOKENS = 160
MIN_NEW_TOKENS = 24
NUM_BEAMS = 5
LENGTH_PENALTY = 1.0
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 2
REPETITION_PENALTY = 1.25

# ==========================================
# 4. HIERARCHICAL CONFIGURATION (TÓM TẮT PHÂN CẤP)
# ==========================================
HIERARCHICAL = {
    "chunk_tokens": 400,
    "overlap_sents": 2,
    "max_levels": 2,
}

