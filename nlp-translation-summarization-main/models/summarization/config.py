import os

# API URL for Streamlit to connect to
API_URL = os.getenv("API_URL", "http://localhost:8000")

# UI Settings
PAGE_TITLE = "NLP Translation & Summarization"
DEFAULT_SOURCE_LANG = "vi"
DEFAULT_TARGET_LANG = "en"

# Max lengths for input
MAX_TEXT_AREA_HEIGHT = 280

# Summarization config (THÊM)
MAX_INPUT_LENGTH = 1024      # Max tokens for input
MAX_OUTPUT_LENGTH = 256      # Max tokens for summary
MIN_OUTPUT_LENGTH = 48       # Min tokens for summary
NUM_BEAMS = 4               # Beam search width
LENGTH_PENALTY = 1.15       # Length penalty
NO_REPEAT_NGRAM_SIZE = 3    # No repeat n-gram