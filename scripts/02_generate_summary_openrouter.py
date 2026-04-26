# -*- coding: utf-8 -*-
import argparse
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"')

# Model pool (ưu tiên model đang ổn định nhất)
MODELS_POOL = [
    "inclusionai/ling-2.6-flash:free",
    "openrouter/free",
]

# Default paths
SUMMARIZE_INPUT_FILE = "data/summary_data_openrouter.jsonl"
TRANSLATE_INPUT_FILE = "data/summary_data_openrouter.jsonl"
OUTPUT_FILE = "data/processed/train.jsonl"
SUMMARIZE_FAILED_FILE = "data/summary_data_failed.jsonl"
TRANSLATE_FAILED_FILE = "data/translate_data_failed.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl().rstrip("/")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def append_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_translation_fields(row: dict[str, Any]) -> bool:
    return bool(str(row.get("summary_en", "")).strip()) and bool(str(row.get("text_en", "")).strip())


def has_summary_fields(row: dict[str, Any]) -> bool:
    return bool(str(row.get("summary_vi", "")).strip())


def load_cached_urls(path: str, mode: str) -> set[str]:
    cached = set()
    for row in load_jsonl(path):
        url = normalize_url(str(row.get("url", "")).strip())
        if not url:
            continue
        if mode == "translate_existing":
            if has_translation_fields(row):
                cached.add(url)
        else:
            if has_summary_fields(row):
                cached.add(url)
    return cached


def extract_json_array(text: str) -> list[dict[str, Any]] | None:
    text = str(text or "").strip()
    if not text:
        return None

    # 1) Try direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2) Try fenced json block
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # 3) Try first [ ... ] block
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
    return None


def call_openrouter(prompt: str, model_name: str, timeout_sec: int = 120) -> tuple[list[dict[str, Any]] | None, str]:
    if not API_KEY:
        return None, "OPENROUTER_API_KEY chưa được thiết lập trong file .env"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "NLP Summary Translation Pipeline",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_sec)
        if response.status_code == 429:
            return None, "rate limit (429)"
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        arr = extract_json_array(content)
        if arr is None:
            return None, "không parse được JSON array từ response"
        return arr, ""
    except Exception as exc:
        return None, str(exc)


def build_summarize_prompt(batch_articles: list[dict[str, Any]]) -> str:
    num_articles = len(batch_articles)
    parts = []
    for i, art in enumerate(batch_articles):
        content = str(art.get("content_vi", "")).strip()
        parts.append(f"--- VĂN BẢN #{i} ---\n{content}\n")
    joined = "\n".join(parts)
    return f"""Bạn là biên tập viên tin tức tiếng Việt.
NHIỆM VỤ: Viết lại tiêu đề ngắn và tóm tắt cho {num_articles} văn bản sau.

{joined}

YÊU CẦU:
1) Không gộp bài, không bỏ sót bài.
2) "title": dưới 12 chữ, rõ ý chính.
3) "summary": 2-3 câu, đúng nội dung gốc.
4) Không bịa thêm chi tiết.

TRẢ VỀ DUY NHẤT JSON ARRAY:
[
  {{"index": 0, "title": "...", "summary": "..."}},
  ...
]
"""


def build_translation_text(item: dict[str, Any], max_chars: int) -> str:
    title = str(item.get("title", "")).strip()
    sapo = str(item.get("sapo_vi", "")).strip()
    content = str(item.get("content_vi", "")).strip()
    if content:
        pieces = [x for x in [title, sapo, content] if x]
        text_vi = "\n\n".join(pieces).strip()
    else:
        text_vi = str(item.get("text_vi", "")).strip()
    if max_chars > 0:
        text_vi = text_vi[:max_chars]
    return text_vi


def build_translate_prompt(batch_articles: list[dict[str, Any]], max_chars: int) -> str:
    num_articles = len(batch_articles)
    parts = []
    for i, art in enumerate(batch_articles):
        summary_vi = str(art.get("summary_vi", "")).strip()
        text_vi = build_translation_text(art, max_chars=max_chars)
        parts.append(
            f"--- BẢN GHI #{i} ---\n"
            f"summary_vi:\n{summary_vi}\n\n"
            f"text_vi:\n{text_vi}\n"
        )
    joined = "\n".join(parts)
    return f"""Bạn là biên dịch viên Việt -> Anh chuyên tin tức.
NHIỆM VỤ: Dịch chính xác {num_articles} bản ghi sau.

{joined}

YÊU CẦU:
1) Dịch đúng nghĩa, giữ nguyên sự kiện và số liệu.
2) Không thêm/bớt thông tin.
3) text_en PHẢI là bản dịch đầy đủ của text_vi, KHÔNG được tóm tắt.
4) Giữ cấu trúc đoạn văn gần với bản gốc (xuống dòng hợp lý).
5) Dùng tiếng Anh tự nhiên, rõ ràng.

TRẢ VỀ DUY NHẤT JSON ARRAY:
[
  {{"index": 0, "summary_en": "...", "text_en": "..."}},
  ...
]
"""


def is_translation_too_short(source_vi: str, target_en: str) -> bool:
    src = str(source_vi or "").strip()
    tgt = str(target_en or "").strip()
    if not src or not tgt:
        return True
    # Với văn bản dài, nếu bản dịch quá ngắn thì gần như model đã tóm tắt thay vì dịch.
    if len(src) >= 300 and len(tgt) < int(len(src) * 0.28):
        return True
    return False


def run_batch_with_model_rotation(prompt: str, start_model_idx: int) -> tuple[list[dict[str, Any]] | None, int, str]:
    last_error = "unknown error"
    for attempt in range(len(MODELS_POOL)):
        model_idx = (start_model_idx + attempt) % len(MODELS_POOL)
        model_name = MODELS_POOL[model_idx]
        print(f"      -> Đang thử model: {model_name}...", end="", flush=True)
        arr, err = call_openrouter(prompt=prompt, model_name=model_name)
        if arr is not None:
            print(" Xong!")
            return arr, (model_idx + 1) % len(MODELS_POOL), ""
        print(f" Lỗi: {err}")
        last_error = err
        if "429" in err:
            time.sleep(5)
        else:
            time.sleep(2)
    return None, start_model_idx, last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["summarize", "translate_existing"],
        default="summarize",
        help="summarize: tom tat VI; translate_existing: dich summary_data_openrouter sang train.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)
    parser.add_argument("--replace_output", action="store_true", help="Xoa output cu va ghi moi.")
    parser.add_argument(
        "--translation_chars",
        type=int,
        default=12000,
        help="So ky tu toi da cua text_vi khi dich trong mode translate_existing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs("data", exist_ok=True)

    if args.mode == "translate_existing":
        default_input = TRANSLATE_FAILED_FILE if args.retry else TRANSLATE_INPUT_FILE
        failed_file = TRANSLATE_FAILED_FILE
    else:
        default_input = SUMMARIZE_FAILED_FILE if args.retry else SUMMARIZE_INPUT_FILE
        failed_file = SUMMARIZE_FAILED_FILE

    input_file = args.input or default_input
    output_file = args.output

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.replace_output and out_path.exists():
        out_path.unlink()
        print(f"--- Da xoa output cu: {output_file} ---")

    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file đầu vào: {input_file}")
        return

    cached_urls = set()
    if out_path.exists():
        cached_urls = load_cached_urls(output_file, mode=args.mode)

    source_rows = load_jsonl(input_file)
    pending_rows: list[dict[str, Any]] = []
    for row in source_rows:
        url = normalize_url(str(row.get("url", "")).strip())
        if not url:
            continue
        if url not in cached_urls:
            pending_rows.append(row)

    if not pending_rows:
        print("✅ Không còn dữ liệu cần xử lý.")
        return

    limit = args.limit if args.limit > 0 else len(pending_rows)
    to_process = pending_rows[:limit]
    mode_text = "DICH VI->EN" if args.mode == "translate_existing" else "TOM TAT VI"
    print(f"--- BAT DAU {mode_text} ---")
    print(f"Tong can xu ly: {len(to_process)} | batch={args.batch} | output={output_file}")
    if args.mode == "translate_existing":
        print(f"translation_chars={args.translation_chars} | goi y batch=1-2 de on dinh parser")

    success_count = 0
    failed_rows: list[dict[str, Any]] = []
    model_idx = 0

    try:
        for i in range(0, len(to_process), max(1, args.batch)):
            batch = to_process[i : i + max(1, args.batch)]
            print(f"\n[Lo {i // max(1, args.batch) + 1}] Dang xu ly {len(batch)} bai...")

            if args.mode == "translate_existing":
                prompt = build_translate_prompt(batch_articles=batch, max_chars=max(0, args.translation_chars))
            else:
                prompt = build_summarize_prompt(batch_articles=batch)

            results, model_idx, err = run_batch_with_model_rotation(prompt=prompt, start_model_idx=model_idx)
            if not results:
                print(f"  ! Lo that bai: {err}")
                failed_rows.extend(batch)
                append_jsonl(batch, failed_file)
                continue

            result_map = {
                r.get("index"): r
                for r in results
                if isinstance(r, dict) and isinstance(r.get("index"), int)
            }

            success_rows: list[dict[str, Any]] = []
            failed_in_batch: list[dict[str, Any]] = []
            for idx, original in enumerate(batch):
                res = result_map.get(idx)
                if not res:
                    failed_in_batch.append(original)
                    continue

                row = dict(original)
                if args.mode == "translate_existing":
                    summary_en = str(res.get("summary_en", "")).strip()
                    text_en = str(res.get("text_en", "")).strip()
                    if not summary_en or not text_en:
                        failed_in_batch.append(original)
                        continue
                    source_vi = build_translation_text(original, max_chars=max(0, args.translation_chars))
                    if is_translation_too_short(source_vi=source_vi, target_en=text_en):
                        failed_in_batch.append(original)
                        continue
                    row["summary_en"] = summary_en
                    row["text_en"] = text_en
                else:
                    summary_vi = str(res.get("summary", "")).strip()
                    title_vi = str(res.get("title", "")).strip() or str(original.get("title", "")).strip()
                    if not summary_vi:
                        failed_in_batch.append(original)
                        continue
                    row["summary_vi"] = summary_vi
                    row["title"] = title_vi
                    row["text_vi"] = f"Tiêu đề: {title_vi}\nNội dung: {str(row.get('content_vi', '')).strip()}"
                success_rows.append(row)

            append_jsonl(success_rows, output_file)
            append_jsonl(failed_in_batch, failed_file)

            success_count += len(success_rows)
            failed_rows.extend(failed_in_batch)
            print(f"✅ Hoan thanh lo. Them {len(success_rows)} thanh cong | tong={success_count}")

            if i + max(1, args.batch) < len(to_process):
                pause = random.randint(3, 7)
                print(f"Nghi {pause}s...")
                time.sleep(pause)

    except KeyboardInterrupt:
        print("\n🛑 Nguoi dung dung chuong trinh.")
    finally:
        print("\n--- TONG KET ---")
        print(f"Thanh cong: {success_count}")
        print(f"That bai: {len(failed_rows)}")
        print(f"Output: {output_file}")
        if failed_rows:
            print(f"Failed: {failed_file}")


if __name__ == "__main__":
    main()
