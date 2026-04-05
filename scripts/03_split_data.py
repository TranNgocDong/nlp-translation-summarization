import argparse
import json
import os
import random

INPUT_FILE = "data/translated_data.jsonl"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"


def load_data():
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # lọc data lỗi
            if all([
                item.get("text_vi"),
                item.get("summary_vi"),
                item.get("text_en"),
                item.get("summary_en")
            ]):
                data.append(item)

    return data


def split_data(data, ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * ratio)

    return data[:split_idx], data[split_idx:]


def save_jsonl(data, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Xao tron toan bo translated_data va ghi de train/val",
    )
    args = ap.parse_args()

    data = load_data()
    if args.replace:
        train, val = split_data(data)
        print(f"Train: {len(train)} | Val: {len(val)} (replace)")
    else:
        train = load_jsonl(TRAIN_FILE)
        val = load_jsonl(VAL_FILE)
        seen = {r["text_vi"] for r in train + val if r.get("text_vi")}
        new_rows = [x for x in data if x.get("text_vi") and x["text_vi"] not in seen]
        if not new_rows:
            print("Khong co mau moi so voi train/val hien tai.")
            return
        tr, va = split_data(new_rows)
        train = train + tr
        val = val + va
        print(f"Them {len(new_rows)} mau: train +{len(tr)}, val +{len(va)} | Tong train {len(train)}, val {len(val)}")

    save_jsonl(train, TRAIN_FILE)
    save_jsonl(val, VAL_FILE)


if __name__ == "__main__":
    main()