import os
import sys
import json
import time
import requests
import argparse
import re
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

# Danh sách các model Free TỐC ĐỘ CAO để xoay tua (Cập nhật 2026)
MODELS_POOL = [
    "google/gemma-3-27b-it:free",      # Model 0: Mới nhất, nhanh, tốt
    "meta-llama/llama-3.3-70b-instruct:free", # Model 1: Rất khỏe và ổn định
    "google/gemma-3-12b-it:free",      # Model 2: Cực nhanh
    "qwen/qwen-2.5-72b-instruct:free", # Model 3: "Vua" tốc độ model free
    "openrouter/auto",                 # Dự phòng cuối
]

INPUT_FILE = "data/raw_data.jsonl"
OUTPUT_FILE = "data/summary_data_openrouter.jsonl"
FAILED_FILE = "data/summary_data_failed.jsonl"


def normalize_url(url: str) -> str:
    """Chuẩn hóa URL để tránh trùng lặp bài báo."""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl().rstrip("/")


def generate_batch_summaries_glm(batch_articles, start_model_idx=0):
    """
    Gom nhiều bài báo và gọi API. 
    start_model_idx: Vị trí model sẽ thử đầu tiên (để xoay tua).
    """
    if not API_KEY:
        raise ValueError("Lỗi: OPENROUTER_API_KEY chưa được thiết lập.")

    num_articles = len(batch_articles)
    print(f"\n      [LÔ HIỆN TẠI - {num_articles} bài]:")
    for idx, art in enumerate(batch_articles):
        t = art.get('title', 'N/A')[:50]
        print(f"        {idx+1}. {t}...")

    articles_input = ""
    for i, art in enumerate(batch_articles):
        articles_input += f"--- VĂN BẢN #{i} ---\n{art.get('content_vi', '').strip()}\n\n"

    prompt = f"""Bạn là chuyên gia tóm tắt tin tức. Dưới đây là {num_articles} văn bản tiếng Việt riêng biệt.
YÊU CẦU CỰC KỲ QUAN TRỌNG:
1. Bạn PHẢI tóm tắt ĐỦ {num_articles} văn bản, không được bỏ sót bất kỳ văn bản nào.
2. Tóm tắt mỗi đoạn ngắn gọn (30-80 từ).
3. Chỉ trả về một mảng JSON duy nhất chứa {num_articles} phần tử theo định dạng:
[
  {{"index": 0, "summary": "Nội dung tóm tắt 0..."}},
  ...
  {{"index": {num_articles-1}, "summary": "Nội dung tóm tắt {num_articles-1}..."}}
]
Tuyệt đối không có lời dẫn hay văn bản thừa."""

    # Sắp xếp thứ tự xoay tua
    num_models = len(MODELS_POOL)
    order = [(start_model_idx + i) % num_models for i in range(num_models)]
    
    for idx_in_pool in order:
        current_model = MODELS_POOL[idx_in_pool]
        print(f"      -> Thử model: {current_model}...", end="", flush=True)
        
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-Title": "NLP Rotation"}
        data = {
            "model": current_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        try:
            start_t = time.time()
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=300)
            dur = time.time() - start_t
            response.raise_for_status()
            
            print(f" Xong! ({dur:.1f}s)")
            content = response.json()['choices'][0]['message']['content'].strip()
            
            json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                if len(results) < num_articles:
                    print(f"      [Warn] AI chỉ trả về {len(results)}/{num_articles} bài. Đang thử lại...")
                    continue
                return results
            return json.loads(content)
        except Exception as e:
            print(f" Lỗi! ({str(e)[:40]}...)")
            time.sleep(3)
    return None


def load_cached_urls(path):
    """Load các URL đã xử lý. Bỏ qua các bài báo bị lỗi 'Chưa có văn bản' để xử lý lại."""
    if not os.path.exists(path): return set()
    cached = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    o = json.loads(line)
                    # TỰ ĐỘNG PHÁT HIỆN LỖI:
                    # Nếu bài báo bị lỗi tóm tắt trước đó, không coi là đã xong để chạy lại
                    sum_text = o.get("summary_vi", "")
                    if "Chưa có văn bản" in sum_text or not sum_text:
                        continue
                    if o.get("url"): 
                        cached.add(normalize_url(o["url"]))
                except: continue
    return cached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--batch", type=int, default=5) # 5 bài là an toàn nhất cho AI Free
    p.add_argument("--retry", action="store_true")
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)
    cached_urls = load_cached_urls(OUTPUT_FILE)

    input_file = FAILED_FILE if args.retry else INPUT_FILE
    
    raw_data = []
    if os.path.exists(input_file):
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if normalize_url(item.get("url", "")) not in cached_urls:
                        raw_data.append(item)
    
    if not raw_data:
        print("Không có bài báo nào cần xử lý (Hoặc tất cả đã được tóm tắt đúng).")
        return

    to_process = raw_data[:args.limit]
    print(f"--- START SYSTEM CLEANED & ROTATION ---")
    print(f"Tổng: {len(to_process)} bài | Batch size: {args.batch}")
    
    count_success = 0
    failed_articles = []
    model_idx = 0 

    for i in range(0, len(to_process), args.batch):
        current_batch = to_process[i : i + args.batch]
        print(f"\n[Lô {i//args.batch + 1}] Bắt đầu xoay tua model...")
        
        batch_results = generate_batch_summaries_glm(current_batch, start_model_idx=model_idx)
        
        if batch_results and isinstance(batch_results, list):
            result_map = {r.get("index"): r.get("summary", "").strip() 
                         for r in batch_results if r.get("index") is not None}
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for idx, original_item in enumerate(current_batch):
                    summary = result_map.get(idx, "")
                    if not summary or "Chưa có văn bản" in summary:
                        failed_articles.append(original_item)
                        continue
                    
                    row = original_item.copy()
                    row["summary_vi"] = summary
                    # Làm sạch text_vi: Chỉ Tiêu đề + Nội dung
                    row["text_vi"] = f"Tiêu đề: {row.get('title', '').strip()}\nNội dung: {row.get('content_vi', '').strip()}"
                    
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count_success += 1
            
            print(f"✅ Xong lô {i//args.batch + 1}. Thành công lô này: {len(result_map)} | Tổng: {count_success}")
            model_idx = (model_idx + 1) % len(MODELS_POOL)
        else:
            failed_articles.extend(current_batch)
            print(f"❌ Cả lô thất bại.")
        
        if i + args.batch < len(to_process):
            print("Nghỉ 10s...")
            time.sleep(10)

    if failed_articles:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            for item in failed_articles:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n⚠️  Phát hiện {len(failed_articles)} bài còn lỗi. Xem tại: {FAILED_FILE}")

    print(f"\n🎉 HOÀN TẤT: {count_success}/{len(to_process)}")

if __name__ == "__main__":
    main()
