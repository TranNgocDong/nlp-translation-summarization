import os
from pathlib import Path

# API Server Settings
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 8000))
RELOAD = os.getenv("API_RELOAD", "False").lower() == "true"

# CORS
CORS_ORIGINS = ["*"]

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model Paths (linked to summarization config)
from models.summarization.config import VI_CHECKPOINT, EN_CHECKPOINT

SUMMARY_CHECKPOINTS = {
    "vi": VI_CHECKPOINT,
    "en": EN_CHECKPOINT,
}
