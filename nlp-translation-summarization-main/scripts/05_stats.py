import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FILE = "data/processed/train.jsonl"

def main():
    if not os.path.exists(FILE):
        print(f"Chưa có file {FILE}. Hãy chạy các bước trước!")
        return

    text_lens = []
    sum_lens = []

    with open(FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("text_vi"):
                text_lens.append(len(item["text_vi"].split()))
            if item.get("summary_vi"):
                sum_lens.append(len(item["summary_vi"].split()))

    if not text_lens:
        print("Dữ liệu trống hoặc không đúng định dạng.")
        return

    print("="*30)
    print("THỐNG KÊ DỮ LIỆU HUẤN LUYỆN")
    print("="*30)
    print(f"Số mẫu: {len(text_lens)}")
    print(f"{'Văn bản gốc':<15} | Trung bình: {sum(text_lens)//len(text_lens):<4} | Max: {max(text_lens):<4} | Min: {min(text_lens)}")
    if sum_lens:
        print(f"{'Bản tóm tắt':<15} | Trung bình: {sum(sum_lens)//len(sum_lens):<4} | Max: {max(sum_lens):<4} | Min: {min(sum_lens)}")
    print("="*30)

if __name__ == "__main__":
    main()