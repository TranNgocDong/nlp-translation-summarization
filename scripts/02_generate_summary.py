import argparse
import json
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

def load_summary_map(path):
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
        help="Tom tat lai tat ca dong trong raw_data, ghi de summary_data.jsonl",
    )
    args = ap.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")
    cached = {} if args.replace else load_summary_map(OUTPUT_FILE)
    data_out = []

    print("--- Dang tai du lieu va tom tat ---")
    with open(INPUT_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            text_vi = item["text_vi"]
            if text_vi in cached:
                print(f"[{i+1}] Bo qua (da co tom tat)")
                data_out.append(cached[text_vi])
                continue
            print(f"[{i+1}] Dang tom tat...")
            summary_vi = summarize_text(text_vi)
            row = {"text_vi": text_vi, "summary_vi": summary_vi}
            data_out.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in data_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done summary!")

if __name__ == "__main__":
    main()