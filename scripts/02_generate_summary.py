import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model này chuyên tóm tắt đa ngôn ngữ (có Tiếng Việt)
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

INPUT_FILE = "data/raw_data.jsonl"
OUTPUT_FILE = "data/summary_data.jsonl"

def summarize_text(text):
    try:
        # mT5 cần prefix hoặc format chuẩn
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            max_length=250, 
            min_length=50, 
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Lỗi tóm tắt: {e}")
        return ""

def main():
    if not os.path.exists("data"): os.makedirs("data")
    data_out = []

    print("--- Đang tải dữ liệu và tóm tắt (vui lòng đợi) ---")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            text_vi = item["text_vi"]

            print(f"[{i+1}] Đang tóm tắt bài viết...")
            summary_vi = summarize_text(text_vi)

            data_out.append({
                "text_vi": text_vi,
                "summary_vi": summary_vi
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✔ Done summary!")

if __name__ == "__main__":
    main()