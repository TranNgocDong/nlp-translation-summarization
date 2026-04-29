import sys
from pathlib import Path

# Add project root to sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from summarization import VIT5Summarizer, HierarchicalSummarizer
from models.summarization.config import EN_CHECKPOINT


def get_english_summarizer():
    # Load base summarizer
    base = VIT5Summarizer(EN_CHECKPOINT, lang_label="en")
    return HierarchicalSummarizer(base)


def summarize_en(text: str):
    summarizer = get_english_summarizer()
    result = summarizer.summarize(text)
    return result

if __name__ == "__main__":
    sample_text = "This is a sample English text to test the summarization feature."
    print(f"Original: {sample_text}")
    try:
        print(f"Summary: {summarize_en(sample_text)}")
    except Exception as e:
        print(f"Error: {e}")
