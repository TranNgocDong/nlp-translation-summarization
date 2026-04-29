import sys
from pathlib import Path

# Add project root to sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from summarization import VIT5Summarizer, HierarchicalSummarizer
from models.summarization.config import VI_CHECKPOINT


def get_vietnamese_summarizer():
    # Load base summarizer
    base = VIT5Summarizer(VI_CHECKPOINT, lang_label="vi")
    return HierarchicalSummarizer(base)


def summarize_vi(text: str):
    summarizer = get_vietnamese_summarizer()
    result = summarizer.summarize(text)
    return result

if __name__ == "__main__":
    sample_text = "Đây là văn bản tiếng Việt mẫu để kiểm tra tính năng tóm tắt."
    print(f"Original: {sample_text}")
    try:
        print(f"Summary: {summarize_vi(sample_text)}")
    except Exception as e:
        print(f"Error: {e}")
