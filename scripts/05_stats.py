import json

FILE = "data/processed/train.jsonl"

def main():
    lengths = []

    with open(FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lengths.append(len(item["text_vi"].split()))

    print("Số mẫu:", len(lengths))
    print("Trung bình độ dài:", sum(lengths)//len(lengths))
    print("Max:", max(lengths))
    print("Min:", min(lengths))


if __name__ == "__main__":
    main()