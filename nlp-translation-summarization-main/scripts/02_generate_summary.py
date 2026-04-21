"""
Generate summaries for raw Vietnamese articles
Uses VIT5 model for fast and accurate summarization
"""

import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm
from pathlib import Path

# ✅ FIX: Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Encoding config for Windows
if (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Import VIT5Summarizer from models
from models.summarization.vit5_wrapper import VIT5Summarizer

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE.upper()}")

INPUT_FILE = "data/raw_data.jsonl"
OUTPUT_FILE = "data/summary_data.jsonl"

# ==================== UTILITIES ====================

def load_cached_urls(path: str) -> set:
    """Load set of URLs already processed"""
    if not os.path.exists(path):
        return set()
    cached = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                url = o.get("url")
                if url:
                    cached.add(url)
            except json.JSONDecodeError:
                continue
    return cached


def is_valid_summary(summary: str, title: str = "", sapo: str = "") -> bool:
    """
    Validate summary quality
    
    Returns True if summary is valid, False otherwise
    """
    if not summary or not summary.strip():
        return False
    
    word_count = len(summary.split())
    
    # Check length
    if word_count < 15 or word_count > 150:
        return False
    
    # Check if it's just copying title or sapo
    if title and summary.lower() == title.lower():
        return False
    if sapo and summary.lower() == sapo.lower():
        return False
    
    # Check for corruption
    corruption_patterns = ['extra_id_', 'dung:', 'marize:', '<extra_id', 'nội dung:']
    if any(p in summary.lower() for p in corruption_patterns):
        return False
    
    return True


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Generate summaries for articles")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Delete old file and regenerate all summaries"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for processing (default: 2)"
    )
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    
    # Load/reset cached URLs
    if args.replace and os.path.exists(OUTPUT_FILE):
        print(f"🗑️  Removing old file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)
        cached_urls = set()
    else:
        cached_urls = load_cached_urls(OUTPUT_FILE)
        if cached_urls:
            print(f"Loaded {len(cached_urls)} cached URLs")

    # Load raw data
    print("--- Loading raw data ---")
    raw_data = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        url = item.get("url")
                        if url and url not in cached_urls:
                            raw_data.append(item)
                    except json.JSONDecodeError:
                        continue

    if not raw_data:
        print("✅ No new articles to summarize")
        return

    print(f"Found {len(raw_data)} new articles to summarize")
    
    # ✅ Initialize model ONCE outside loop
    print("\n⏳ Loading VIT5 model...")
    summarizer = VIT5Summarizer("VietAI/vit5-base", lang_label="vi")
    print("✅ Model ready\n")
    
    count_written = 0
    count_invalid = 0
    BATCH_SIZE = args.batch_size
    
    # Process articles
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(raw_data), BATCH_SIZE), desc="Summarizing"):
            batch_items = raw_data[i : i + BATCH_SIZE]
            
            for item in batch_items:
                title = item.get("title", "")
                sapo = item.get("sapo_vi", "")
                content = item.get("content_vi", "")
                
                # Skip if content too short
                if not content or len(content.split()) < 50:
                    continue
                
                # Prepare input (with structure)
                full_text = f"{title}\n{sapo}\n{content}".strip()
                full_text = re.sub(r'[ \t]+', ' ', full_text).strip()
                
                # Summarize
                try:
                    result = summarizer.summarize(
                        full_text,
                        max_input_length=1024,
                        max_new_tokens=150,
                        min_new_tokens=20,
                        num_beams=2,  # Faster
                        length_penalty=1.5
                    )
                    summary_vi = result.get("summary", "").strip()
                    
                    # Validate summary
                    if not is_valid_summary(summary_vi, title, sapo):
                        count_invalid += 1
                        continue
                    
                    # Save to file
                    row = item.copy()
                    row["summary_vi"] = summary_vi
                    row["text_vi"] = full_text
                    
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count_written += 1
                
                except Exception as e:
                    print(f"❌ Error summarizing: {e}")
                    continue
            
            f.flush()
            
            # Clear CUDA cache
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Completed!")
    print(f"  - Written: {count_written} summaries")
    print(f"  - Invalid: {count_invalid} (filtered out)")
    print(f"  - Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()