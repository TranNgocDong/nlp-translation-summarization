import json
from pathlib import Path
from torch.utils.data import Dataset

class JsonlSummarizeDataset(Dataset):
    """
    Dataset dùng chung cho cả huấn luyện Tiếng Việt và Tiếng Anh.
    Đọc từ file .jsonl và tokenize dữ liệu.
    """
    def __init__(
        self,
        path: Path,
        tokenizer,
        text_key: str,
        summary_key: str,
        max_input: int = 512,
        max_target: int = 128,
        prefix: str = "summarize: ",
    ):
        self.rows = []
        self.path = path
        if not path.exists():
            print(f"Warning: File {path} không tồn tại.")
        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.rows.append(json.loads(line))
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                        
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.summary_key = summary_key
        self.max_input = max_input
        self.max_target = max_target
        self.prefix = prefix

        raw_count = len(self.rows)
        self.rows = [
            row for row in self.rows
            if str(row.get(self.text_key, "")).strip() and str(row.get(self.summary_key, "")).strip()
        ]
        dropped = raw_count - len(self.rows)
        print(
            f"[Dataset] {path.name}: raw={raw_count}, valid({self.text_key},{self.summary_key})={len(self.rows)}, dropped={dropped}"
        )
        if not self.rows:
            raise ValueError(
                "Khong co du lieu hop le de train.\n"
                f"- File: {self.path}\n"
                f"- Can co truong: {self.text_key}, {self.summary_key}\n"
                "Hay tao bo du lieu dung ngon ngu truoc khi train."
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = self.prefix + str(row.get(self.text_key, ""))
        summary = str(row.get(self.summary_key, ""))
        
        enc = self.tokenizer(
            text,
            max_length=self.max_input,
            truncation=True,
            padding="max_length", # Đảm bảo batch đồng nhất
        )
        
        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(
                summary,
                max_length=self.max_target,
                truncation=True,
                padding="max_length",
            )
            
        enc["labels"] = tgt["input_ids"]
        return enc
