# Translation model configurations
MODEL_VI_TO_EN = "Helsinki-NLP/Opus-MT-vi-en"
MODEL_EN_TO_VI = "Helsinki-NLP/Opus-MT-en-vi"

# Model settings
MAX_LENGTH = 512      # Max input length (tokens)
BATCH_SIZE = 4        # Batch size for processing
NUM_BEAMS = 4         # Beam search width
DEVICE = "cpu"        # "cpu" or "cuda" - will fallback to cpu if gpu unavailable

# Generation settings (optional - can be used in future)
TEMPERATURE = 0.9     # Temperature for sampling (not used in current deterministic mode)
TOP_P = 0.95          # Top-p nucleus sampling (not used in current deterministic mode)

# Output settings
OUTPUT_MAX_LENGTH = 256  # Maximum output length