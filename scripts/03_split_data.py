import sys
import random

# Thiết lập encoding cho stdout để tránh lỗi Unicode trên Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import json
import os

# File đầu vào (có thể từ bước 02 hoặc bước 04)
INPUT_FILE = "data/summary_data.jsonl" # Chỉnh lại để có thể dùng ngay sau bước 02
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"

def load_data(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Lọc dữ liệu tối thiểu phải có Văn bản và Tóm tắt
            if item.get("text_vi") and item.get("summary_vi"):
                data.append(item)
    return data

def split_data(data, ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * ratio)
    return data[:split_idx], data[split_idx:]

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true", help="Ghi đè hoàn toàn tập Train/Val")
    ap.add_argument("--input", type=str, default=INPUT_FILE, help="File đầu vào")
    args = ap.parse_args()

    # Thử tìm file translated trước, nếu không có thì dùng file summary
    actual_input = args.input
    if not os.path.exists(actual_input) and os.path.exists("data/translated_data.jsonl"):
        actual_input = "data/translated_data.jsonl"

    print(f"--- Đang chia dữ liệu từ: {actual_input} ---")
    data = load_data(actual_input)
    
    if not data:
        print("Không có dữ liệu hợp lệ để chia.")
        return

    if args.replace:
        train, val = split_data(data)
    else:
        existing_train = load_data(TRAIN_FILE)
        existing_val = load_data(VAL_FILE)
        
        # Dùng URL hoặc text_vi để kiểm tra trùng
        seen = {r.get("url") or r.get("text_vi") for r in existing_train + existing_val}
        new_data = [x for x in data if (x.get("url") or x.get("text_vi")) not in seen]
        
        if not new_data:
            print("Không có dữ liệu mới.")
            return
            
        tr, va = split_data(new_data)
        train = existing_train + tr
        val = existing_val + va

    save_jsonl(train, TRAIN_FILE)
    save_jsonl(val, VAL_FILE)
    print(f"✅ Hoàn tất: Train ({len(train)} bài), Val ({len(val)} bài)")

if __name__ == "__main__":
    main()