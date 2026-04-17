import os

# API URL for Streamlit to connect to
API_URL = os.getenv("API_URL", "http://localhost:8000")

# UI Settings
PAGE_TITLE = "NLP Translation & Summarization"
DEFAULT_SOURCE_LANG = "vi"
DEFAULT_TARGET_LANG = "en"

# Max lengths for input
MAX_TEXT_AREA_HEIGHT = 280
