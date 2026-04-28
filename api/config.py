import os
import sys
from pathlib import Path

# API Server Settings
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 8000))
RELOAD = os.getenv("API_RELOAD", "False").lower() == "true"

# CORS
CORS_ORIGINS = ["*"]

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load centralized config if available
try:
    import config
    VI_CHECKPOINT = config.OUTPUT_DIR_VI / "best_checkpoint"
    EN_CHECKPOINT = config.OUTPUT_DIR_EN / "best_checkpoint"
except ImportError:
    VI_CHECKPOINT = PROJECT_ROOT / "models" / "vit5-summarize-vi" / "best_checkpoint"
    EN_CHECKPOINT = PROJECT_ROOT / "models" / "vit5-summarize-en" / "best_checkpoint"

SUMMARY_CHECKPOINTS = {
    "vi": VI_CHECKPOINT,
    "en": EN_CHECKPOINT,
}

