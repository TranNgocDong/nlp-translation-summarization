import json
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
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    data = load_data()
    train, val = split_data(data)

    save_jsonl(train, TRAIN_FILE)
    save_jsonl(val, VAL_FILE)

    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")


if __name__ == "__main__":
    main()