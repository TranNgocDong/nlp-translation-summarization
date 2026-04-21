"""
Translate Vietnamese summaries to English
Uses Helsinki-NLP/Opus-MT model
"""

import argparse
import json
import os
import re
from tqdm import tqdm

import torch
from transformers import MarianMTModel, MarianTokenizer
import sys
from pathlib import Path

# ✅ FIX: Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Setup model and device
print("Loading translation model...")
model_name = "Helsinki-NLP/opus-mt-vi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
print(f"✅ Model loaded on {DEVICE.upper()}")

INPUT_FILE = "data/summary_data.jsonl"
OUTPUT_FILE = "data/translated_data.jsonl"


# ==================== TEXT CLEANING ====================

def _clean_text(text: str) -> str:
    """
    Clean corrupted translation output
    
    Removes:
    - T5 sentinel tokens
    - Weird/corrupted characters
    - Extra whitespace
    """
    
    if not text:
        return ""
    
    # ========== Remove T5 Sentinel Tokens ==========
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'extra_id_\d+', '', text)
    
    # ========== Remove Corruption Patterns ==========
    # Patterns that appear in corrupted output
    corruption_patterns = [
        r'dung:',
        r'marize:',
        r'Content:',
        r'nội dung:',
        r'\.com/\S+',      # URLs
        r'https?://\S+',   # URLs
    ]
    
    for pattern in corruption_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # ========== Remove Weird Characters ==========
    weird_chars = [
        # ASCII symbols
        '+', '_', 'Z', 'j', '&', 'W',
        # Vietnamese corrupted diacritics
        'Ỵ', 'Ẽ', 'Ơ', 'Ẫ', 'Ẳ', 'Ỡ',
        'Ế', 'Ỹ', 'Ữ', 'Ư', 'Ỗ', 'Ặ',
        'Õ', 'Ẻ', 'Ỷ', 'Ẹ', 'Ỏ', 'Ừ',
        # Special symbols
        '|', '`', '~', '^', '¬', '§', '¶', '†', '‡', '¿', '¡',
        '©', '®', '™', '€', '£', '¥', '¢',
    ]
    
    for char in weird_chars:
        text = text.replace(char, ' ')
    
    # ========== Fix Spacing ==========
    text = re.sub(r'\s+', ' ', text)          # Multiple spaces
    text = re.sub(r'\.+', '.', text)          # Multiple dots
    text = re.sub(r',+', ',', text)           # Multiple commas
    text = re.sub(r'!+', '!', text)           # Multiple !
    text = re.sub(r'\?+', '?', text)          # Multiple ?
    text = re.sub(r':+', ':', text)           # Multiple :
    
    # ========== Fix Sentence Structure ==========
    # Add space after punctuation + letter
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    text = re.sub(r',([A-Za-z])', r', \1', text)
    text = re.sub(r'!([A-Za-z])', r'! \1', text)
    text = re.sub(r'\?([A-Za-z])', r'? \1', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    return text.strip()


def translate_text(text: str) -> str:
    """
    Translate Vietnamese text to English
    
    Args:
        text: Vietnamese text to translate
    
    Returns:
        Translated English text (cleaned)
    """
    if not text or not text.strip():
        return ""
    
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        # Translate
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        
        # Decode
        result = tokenizer.decode(
            translated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # ✅ CLEAN OUTPUT
        result = _clean_text(result)
        
        return result
    
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ""


def load_translated_map(path: str) -> dict:
    """Load map of already translated items"""
    m = {}
    if not os.path.exists(path):
        return m
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                # Use URL as unique key
                key = o.get("url") or o.get("text_vi", "")
                if key:
                    m[key] = o
            except json.JSONDecodeError:
                continue
    
    return m


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Translate Vietnamese summaries to English")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Translate all items again"
    )
    args = ap.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Please run script 02_generate_summary.py first")
        return

    # Load cached translations
    cached = {} if args.replace else load_translated_map(OUTPUT_FILE)
    if cached:
        print(f"Loaded {len(cached)} cached translations")
    
    # Load input data
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    print(f"--- Translating {len(raw_lines)} items (VI → EN) ---\n")
    data_out = []

    for line in tqdm(raw_lines, desc="Translation"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Use URL as key
        key = item.get("url") or item.get("text_vi", "")
        
        # Check if already cached
        if key in cached:
            data_out.append(cached[key])
            continue
        
        # Translate summary and text
        summary_vi = item.get("summary_vi", "")
        text_vi = item.get("text_vi", "")
        
        summary_en = translate_text(summary_vi)
        text_en = translate_text(text_vi[:800]) if text_vi else ""
        
        # Add translations to item
        item["summary_en"] = summary_en
        item["text_en"] = text_en
        data_out.append(item)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ Translation complete!")
    print(f"  - Total items: {len(data_out)}")
    print(f"  - Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()