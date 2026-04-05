import argparse
import json
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

# Khởi tạo model và tokenizer trực tiếp
model_name = "Helsinki-NLP/opus-mt-vi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

INPUT_FILE = "data/summary_data.jsonl"
OUTPUT_FILE = "data/translated_data.jsonl"

def translate_text(text):
    if not text or text.strip() == "": return ""
    try:
        # Chuẩn bị input cho model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Thực hiện dịch
        with torch.no_grad():
            translated = model.generate(**inputs)
        # Giải mã kết quả
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Lỗi khi dịch: {e}")
        return ""

def load_translated_map(path):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            tv = o.get("text_vi")
            if tv:
                m[tv] = o
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Dich lai tat ca dong trong summary_data, ghi de translated_data.jsonl",
    )
    args = ap.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"Khong tim thay {INPUT_FILE}. Chay buoc 02 truoc.")
        return

    cached = {} if args.replace else load_translated_map(OUTPUT_FILE)
    data_out = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        lines = f.readlines()
        print(f"--- Dich {len(lines)} dong (VI -> EN) ---")

        for i, line in enumerate(lines):
            item = json.loads(line)
            tv = item["text_vi"]
            if tv in cached:
                print(f"[{i+1}/{len(lines)}] Bo qua (da co ban dich)")
                data_out.append(cached[tv])
                continue
            print(f"[{i+1}/{len(lines)}] Dang xu ly...")
            item["summary_en"] = translate_text(item["summary_vi"])
            item["text_en"] = translate_text(item["text_vi"][:1000])
            data_out.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Xong: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()