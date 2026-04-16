import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm # Thêm tqdm để xem tiến trình
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Cấu hình encoding cho Windows console để in được tiếng Việt
if (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

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
        # Giữ lại cấu trúc ngắt dòng, chỉ xóa khoảng trắng thừa
        clean_text = re.sub(r'[ \t]+', ' ', text).strip()
        
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


def load_cached_urls(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    cached: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = o.get("url")
            if url:
                cached.add(url)
    return cached


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
        cached_urls = load_cached_urls(OUTPUT_FILE)

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
    
    count_written = 0
    BATCH_SIZE = 2 # Với 4GB VRAM, hãy thử số 2 trước. 
    
    # Mở file bằng chế độ "a" (Append) để ghi nối tiếp liên tục
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(raw_data), BATCH_SIZE), desc="Đang xử lý theo Batch"):
            batch_items = raw_data[i : i + BATCH_SIZE]
            
            valid_items = []
            texts_to_summarize = []
            
            # 1. Chuẩn bị text cho cả batch
            for item in batch_items:
                title = item.get("title", "")
                sapo = item.get("sapo_vi", "")
                content = item.get("content_vi", "")
                
                if not content or len(content.split()) < 50:
                    continue # Bỏ qua bài quá ngắn
                    
                # Tạo cấu trúc rõ ràng cho mô hình sau này học
                full_text = f"Tiêu đề: {title}\nTóm tắt nhanh: {sapo}\nNội dung:\n{content}".strip()
                
                # Giữ lại cấu trúc ngắt dòng, chỉ xóa khoảng trắng thừa
                clean_text = re.sub(r'[ \t]+', ' ', full_text).strip()
                
                valid_items.append((item, clean_text))
                texts_to_summarize.append(clean_text)
                
            if not texts_to_summarize:
                continue
                
            # 2. Đưa cả mẻ vào Tokenizer cùng lúc
            try:
                inputs = tokenizer(
                    texts_to_summarize, 
                    max_length=1024, 
                    padding=True,       # Bắt buộc khi dùng batch
                    truncation=True, 
                    return_tensors="pt"
                ).to(DEVICE)
                
                # 3. Generate cho cả mẻ cùng lúc
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    attention_mask=inputs.get("attention_mask"), # Thêm attention_mask
                    num_beams=4,              # Hạ beam xuống 4 để tiết kiệm chút VRAM
                    max_length=250, 
                    min_length=30, 
                    no_repeat_ngram_size=3, 
                    length_penalty=2.0, 
                    early_stopping=True
                )
                
                # 4. Giải mã và lưu từng kết quả
                for idx, (item, clean_text) in enumerate(valid_items):
                    summary_vi = tokenizer.decode(summary_ids[idx], skip_special_tokens=True)
                    
                    # ================= BỘ LỌC CHẤT LƯỢNG NHÃN =================
                    if not summary_vi:
                        continue
                        
                    summary_words = summary_vi.split()
                    title = item.get("title", "")
                    sapo = item.get("sapo_vi", "")
                    
                    # 1. Lọc theo độ dài
                    if len(summary_words) < 15 or len(summary_words) > 200:
                        continue
                        
                    # 2. Chống AI lười (chép y chang Title hoặc Sapo)
                    if summary_vi.lower() in title.lower() or summary_vi.lower() in sapo.lower():
                        continue
                    # =========================================================

                    row = item.copy() 
                    row["summary_vi"] = summary_vi
                    row["text_vi"] = clean_text 
                    
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count_written += 1
                
                f.flush()
                
                # 5. Dọn rác VRAM sau mỗi mẻ
                if DEVICE == "cuda":
                    torch.cuda.empty_cache() 
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[Cảnh báo] Tràn VRAM! Hãy thử giảm BATCH_SIZE xuống.")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"Lỗi: {e}")

    print(f"Hoàn thành! Đã tóm tắt và lưu {count_written} bài báo mới.")

if __name__ == "__main__":
    main()