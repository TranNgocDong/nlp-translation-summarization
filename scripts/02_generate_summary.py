import argparse
import json
import os
import torch
from tqdm import tqdm # Thêm tqdm để xem tiến trình
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Thiết lập thiết bị (GPU nếu có, ngược lại CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy mô hình trên: {DEVICE.upper()}")

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

INPUT_FILE = "data/raw_data.jsonl"
OUTPUT_FILE = "data/summary_data.jsonl"

def summarize_text(text):
    try:
        # T5/mT5 hoạt động tốt hơn khi có task prefix (tuỳ biến theo dataset gốc)
        # Tuy bản XLSum được train trên báo chí đa ngôn ngữ, việc đưa thẳng text vẫn ổn, 
        # nhưng ta sẽ dọn dẹp text một chút để tránh các ký tự rác làm model phân tâm.
        clean_text = " ".join(text.split())
        
        inputs = tokenizer([clean_text], max_length=1024, return_tensors="pt", truncation=True).to(DEVICE)
        
        # TỐI ƯU HÓA CHẤT LƯỢNG TÓM TẮT
        summary_ids = model.generate(
            inputs["input_ids"], 
            num_beams=5,                   # Tăng beam search để quét nhiều nhánh hơn (tăng độ chính xác)
            max_length=250,                
            min_length=30,                 # Giảm min_length để tránh model cố bôi chữ với tin ngắn
            no_repeat_ngram_size=3,        # RẤT QUAN TRỌNG: Chống lặp từ (VD: "hôm nay hôm nay hôm nay")
            length_penalty=2.0,            # Khuyến khích model sinh câu dài hơn một chút thay vì cụt lủn
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
            try:
                o = json.loads(line)
                url = o.get("url") # Dùng URL làm định danh duy nhất
                if url and o.get("summary_vi"):
                    m[url] = o
            except json.JSONDecodeError:
                continue
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Xóa file cũ, tóm tắt lại từ đầu toàn bộ data",
    )
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    
    # Nếu chọn replace, xóa trắng file cũ
    if args.replace and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        cached_urls = set()
    else:
        # Chỉ lấy danh sách URL đã làm, KHÔNG load toàn bộ data vào RAM
        cached_urls = set(load_summary_map(OUTPUT_FILE).keys())

    print("--- Đang tải dữ liệu gốc ---")
    raw_data = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    url = item.get("url")
                    if url not in cached_urls:
                        raw_data.append(item)

    if not raw_data:
        print("Hủy bỏ: Không có bài mới nào cần tóm tắt.")
        return

    print(f"Số lượng bài mới phát hiện: {len(raw_data)}")
    
    count_success = 0
    
    # Mở file bằng chế độ "a" (Append) để ghi nối tiếp liên tục
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for item in tqdm(raw_data, desc="Đang xử lý bài mới"):
            title = item.get("title", "")
            sapo = item.get("sapo_vi", "")
            content = item.get("content_vi", "")
            
            # GIỮ NGUYÊN BƯỚC NÀY: Gộp Sapo vào cho AI đọc là một best practice
            full_text = f"{title}\n\n{sapo}\n\n{content}".strip()
            
            if not full_text or len(full_text.split()) < 20:
                continue
                
            # Đưa vào model để tóm tắt
            summary_vi = summarize_text(full_text)
            
            if summary_vi:
                row = item.copy() 
                row["summary_vi"] = summary_vi
                row["text_vi"] = full_text 
                
                # Ghi thẳng vào file ngay lập tức và flush (ép lưu xuống ổ cứng)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush() # Đảm bảo không mất data nếu mất điện đột ngột
                
                count_success += 1

    print(f"Hoàn thành! Đã tóm tắt và lưu {count_success} bài báo mới.")

if __name__ == "__main__":
    main()