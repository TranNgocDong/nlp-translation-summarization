import os
import json
import csv
import sys
import hashlib
from pathlib import Path
from tqdm import tqdm

# Đảm bảo import được relation_graph từ Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from relation_graph import build_relation_graph

# Cấu hình encoding cho Windows console
if (sys.stdout.encoding or "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

INPUT_TSV = "vietnews/test.tsv"
OUTPUT_JSONL = "data/vietnews_converted.jsonl"
MAX_SAMPLES = 5000

def get_first_sentence(text):
    if not text:
        return ""
    # Cắt lấy câu đầu tiên làm tiêu đề giả định
    parts = text.split('.', 1)
    return parts[0].strip() + "." if len(parts) > 1 else text

def process_row(content, summary):
    # Dọn dẹp dấu cách thừa
    content = " ".join(content.split())
    summary = " ".join(summary.split())
    
    if len(content.split()) < 50:
        return None
    
    # Giả lập title từ câu đầu
    title = get_first_sentence(content)
    
    # 1. Trích xuất NER tương tự wrap_vi_like_training
    try:
        graph = build_relation_graph(content)
        entities = [str(node["id"]) for node in graph.get("nodes", [])][:7]
        ner_str = ", ".join(entities) if entities else ""
    except:
        ner_str = ""
        
    ner_prefix = f"Từ khóa bắt buộc: {ner_str}\n" if ner_str else ""
    
    # 2. Tạo cấu trúc Prompt (text_vi)
    # Khớp với định dạng trong 02_generate_summary.py cộng thêm NER prefix
    prompt_body = f"Tiêu đề: {title}\nNội dung:\n{content}"
    text_vi = f"{ner_prefix}{prompt_body}"
    
    # Tạo hash làm URL/ID duy nhất
    url_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    return {
        "source": "VietNews",
        "url": f"vietnews://{url_hash}",
        "title": title,
        "sapo_vi": "", # Vietnews tsv không tách riêng sapo
        "content_vi": content,
        "summary_vi": summary,
        "text_vi": text_vi
    }

def main():
    if not os.path.exists(INPUT_TSV):
        print(f"Lỗi: Không tìm thấy file {INPUT_TSV}")
        return

    os.makedirs("data", exist_ok=True)
    
    print(f"--- Đang bắt đầu chuyển đổi (Tối đa {MAX_SAMPLES} bài) ---")
    
    count = 0
    with open(INPUT_TSV, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        
        # Đọc TSV
        reader = csv.reader(f_in, delimiter='\t')
        
        # Dùng tqdm để hiện tiến trình
        for row in tqdm(reader, total=MAX_SAMPLES, desc="Processing"):
            if len(row) < 2:
                continue
                
            content = row[0]
            summary = row[1]
            
            processed = process_row(content, summary)
            if processed:
                f_out.write(json.dumps(processed, ensure_ascii=False) + "\n")
                count += 1
                
            if count >= MAX_SAMPLES:
                break
                
    print(f"✅ Hoàn tất! Đã lưu {count} mẫu vào {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
