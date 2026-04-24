# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import requests
import argparse
import re
import random
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"')

# Danh sách các model Free chất lượng cao để xoay tua
MODELS_POOL = [
    "openrouter/free",                 # Dự phòng cuối
]

INPUT_FILE = "data/raw_data.jsonl"
OUTPUT_FILE = "data/summary_data_openrouter.jsonl"
FAILED_FILE = "data/summary_data_failed.jsonl"

def normalize_url(url: str) -> str:
    """Chuẩn hóa URL để tránh trùng lặp bài báo."""
    if not url: return ""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl().rstrip("/")

def load_cached_urls(path):
    """Load các URL đã xử lý thành công."""
    if not os.path.exists(path): return set()
    cached = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    o = json.loads(line)
                    sum_text = o.get("summary_vi", "")
                    if sum_text and "Vui lòng cung cấp" not in sum_text:
                        if o.get("url"): 
                            cached.add(normalize_url(o["url"]))
                except: continue
    return cached

def generate_batch_summaries_glm(batch_articles, start_model_idx=0):
    """Gom nhiều bài báo và gọi API OpenRouter."""
    if not API_KEY:
        print("❌ Lỗi: OPENROUTER_API_KEY chưa được thiết lập trong file .env")
        return None

    num_articles = len(batch_articles)
    articles_input = ""
    for i, art in enumerate(batch_articles):
        content = art.get('content_vi', '').strip()
        articles_input += f"--- VĂN BẢN #{i} ---\n{content}\n\n"

    prompt = f"""Bạn là một AI chuyên gia biên tập tin tức.
NHIỆM VỤ: Tóm tắt {num_articles} văn bản sau đây sang tiếng Việt.

--- DANH SÁCH VĂN BẢN ĐẦU VÀO ---
{articles_input}
--- KẾT THÚC DANH SÁCH ---

YÊU CẦU VỀ NỘI DUNG:
1. Độ phủ: Phải tóm tắt ĐỦ {num_articles} văn bản, không được gộp bài hay bỏ sót.
2. Tiêu đề: Viết lại tiêu đề mới mang tính báo chí, cực ngắn (dưới 12 chữ).
3. Tóm tắt: Độ dài khoảng 2-3 câu văn (tập trung vào sự kiện chính: Ai? Cái gì? Ở đâu? Tại sao?).
4. Ngôn ngữ: Tiếng Việt tự nhiên, không dùng từ ngữ máy móc.

YÊU CẦU ĐỊNH DẠNG (BẮT BUỘC):
- Trả về DUY NHẤT một mảng JSON. 
- Mỗi phần tử phải có đủ 3 field: "index", "title", "summary".
- Giá trị "index" phải khớp chính xác với số thứ tự của văn bản (từ 0 đến {num_articles-1}).

MẪU JSON ĐẦU RA:
[
  {{"index": 0, "title": "...", "summary": "..."}},
  ...
]"""

    # Thử lần lượt các model trong pool
    for attempt in range(len(MODELS_POOL)):
        model_idx = (start_model_idx + attempt) % len(MODELS_POOL)
        current_model = MODELS_POOL[model_idx]
        
        print(f"      -> Đang thử model: {current_model}...", end="", flush=True)
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Title": "NLP News Summarizer"
        }
        data = {
            "model": current_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            start_t = time.time()
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=120)
            dur = time.time() - start_t
            
            if response.status_code == 429:
                print(f" Bị giới hạn tốc độ (429).")
                time.sleep(5)
                continue

            response.raise_for_status()
            print(f" Xong! ({dur:.1f}s)")
            
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content'].strip()
            
            json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                if len(results) >= num_articles:
                    return results
            
            # Nếu không tìm thấy regex, thử load trực tiếp
            results = json.loads(content)
            if isinstance(results, list):
                return results
                
        except Exception as e:
            print(f" Lỗi: {str(e)[:50]}")
            time.sleep(2)
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    cached_urls = load_cached_urls(OUTPUT_FILE)

    input_file = FAILED_FILE if args.retry else INPUT_FILE
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file đầu vào: {input_file}")
        return

    raw_data = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    url = item.get("url", "")
                    if normalize_url(url) not in cached_urls:
                        raw_data.append(item)
                except: continue

    if not raw_data:
        print("✅ Tất cả bài báo đã được xử lý thành công.")
        return

    to_process = raw_data[:args.limit]
    print(f"--- BẮT ĐẦU TÓM TẮT ---")
    print(f"Tổng cần xử lý: {len(to_process)} bài | Batch size: {args.batch}")

    count_success = 0
    failed_articles = []
    model_idx = 0

    try:
        for i in range(0, len(to_process), args.batch):
            batch = to_process[i : i + args.batch]
            print(f"\n[Lô {i//args.batch + 1}] Đang xử lý {len(batch)} bài...")
            
            results = generate_batch_summaries_glm(batch, start_model_idx=model_idx)
            
            if results and isinstance(results, list):
                result_map = {r.get("index"): r for r in results if r.get("index") is not None}
                
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for idx, original_item in enumerate(batch):
                        res = result_map.get(idx)
                        if res and res.get("summary"):
                            row = original_item.copy()
                            row["summary_vi"] = res["summary"].strip()
                            row["title"] = res.get("title", original_item.get("title")).strip()
                            row["text_vi"] = f"Tiêu đề: {row['title']}\nNội dung: {row.get('content_vi', '').strip()}"
                            
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            count_success += 1
                        else:
                            failed_articles.append(original_item)
                            with open(FAILED_FILE, "a", encoding="utf-8") as ff:
                                ff.write(json.dumps(original_item, ensure_ascii=False) + "\n")
                
                print(f"✅ Hoàn thành lô. Tổng thành công: {count_success}")
                model_idx = (model_idx + 1) % len(MODELS_POOL)
            else:
                print(f"❌ Lô thất bại hoàn toàn. Đang lưu lại để thử sau...")
                failed_articles.extend(batch)
                with open(FAILED_FILE, "a", encoding="utf-8") as ff:
                    for item in batch:
                        ff.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if i + args.batch < len(to_process):
                pause = random.randint(5, 10)
                print(f"Nghỉ {pause}s...")
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n🛑 Người dùng dừng chương trình.")
    finally:
        print(f"\n--- TỔNG KẾT ---")
        print(f"Thành công: {count_success}")
        print(f"Thất bại/Lỗi: {len(failed_articles)}")
        if failed_articles:
            print(f"Dữ liệu lỗi được lưu tại: {FAILED_FILE}")

if __name__ == "__main__":
    main()
