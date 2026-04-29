import argparse
import json
import os
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import MarianMTModel, MarianTokenizer

# Khởi tạo model và tokenizer
model_name = "Helsinki-NLP/opus-mt-vi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

INPUT_FILE = "data/summary_data.jsonl"
OUTPUT_FILE = "data/translated_data.jsonl"

def translate_text(text):
    if not text or text.strip() == "": return "", False
    try:
        # Kiểm tra xem có bị truncate bởi tokenizer không (max_length=512 tokens)
        token_count = len(tokenizer.encode(text))
        truncated = token_count > 512
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True), truncated
    except Exception as e:
        print(f"Lỗi khi dịch: {e}")
        return "", False

def load_translated_map(path):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            # Ưu tiên dùng URL làm khóa duy nhất
            key = o.get("url") or o.get("text_vi")
            if key: m[key] = o
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true", help="Dịch lại tất cả")
    args = ap.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy {INPUT_FILE}. Hãy chạy bước 02 trước.")
        return

    cached = {} if args.replace else load_translated_map(OUTPUT_FILE)
    
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    print(f"--- Đang dịch {len(raw_lines)} dòng (VI -> EN) ---")
    data_out = []

    for line in tqdm(raw_lines, desc="Dịch thuật"):
        item = json.loads(line)
        key = item.get("url") or item.get("text_vi")
        
        if key in cached:
            data_out.append(cached[key])
            continue
            
        # Dịch tóm tắt và toàn bộ văn bản gốc (không truncate cứng 800 ký tự)
        summary_en, _ = translate_text(item.get("summary_vi", ""))
        text_en, truncated = translate_text(item.get("text_vi", ""))
        
        item["summary_en"] = summary_en
        item["text_en"] = text_en
        
        # Lưu metadata
        item["translated_by"] = "Helsinki-NLP/opus-mt-vi-en"
        item["translated_at"] = datetime.now().isoformat()
        item["source_length"] = len(item.get("text_vi", ""))
        item["truncated"] = truncated
        
        data_out.append(item)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã hoàn thành dịch thuật tại {OUTPUT_FILE}")

if __name__ == "__main__":
    main()