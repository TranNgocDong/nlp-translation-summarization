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

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy {INPUT_FILE}. Bạn đã chạy bước 02 chưa?")
        return

    data_out = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"--- Đang dịch {len(lines)} bài viết (VI -> EN) ---")
        
        for i, line in enumerate(lines):
            item = json.loads(line)
            print(f"[{i+1}/{len(lines)}] Đang xử lý...")
            
            # Dịch summary và một phần nội dung chính
            item["summary_en"] = translate_text(item["summary_vi"])
            item["text_en"] = translate_text(item["text_vi"][:1000]) # Cắt bớt để chạy nhanh
            
            data_out.append(item)

    # Lưu file kết quả
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Xong! Đã tạo file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()