# NLP Translation & Summarization

Hệ thống NLP song ngữ Việt–Anh sử dụng `ViT5`, gồm:

- **Tóm tắt tiếng Việt:** `text_vi → summary_vi` (VietAI/vit5-base)
- **Tóm tắt tiếng Anh:** `text_en → summary_en` (google/mt5-base)
- **Dịch thuật:** `VI → EN` (Helsinki-NLP/opus-mt-vi-en)
- **Trích xuất quan hệ:** Character Relation Graph

Giao diện được viết bằng `Streamlit`, phần xử lý phía sau dùng `FastAPI`.

---

## Mục lục

1. [Tổng quan dự án](#tổng-quan-dự-án)
2. [Cài đặt](#cài-đặt)
3. [Quick Start](#quick-start)
4. [Pipeline dữ liệu](#pipeline-dữ-liệu)
5. [API Usage](#api-usage)
6. [UI Usage](#ui-usage)
7. [Cấu trúc dự án](#cấu-trúc-dự-án)
8. [Models Reference](#models-reference)
9. [Troubleshooting](#troubleshooting)
10. [Cải tiến tương lai](#cải-tiến-tương-lai)

---

## Tổng quan dự án

Dự án xây dựng một pipeline NLP đầy đủ cho tiếng Việt và tiếng Anh:

| Module | Mô tả |
|--------|-------|
| **Crawler** | Cào báo tự động từ VnExpress, TuoiTre, DanTri, v.v. |
| **Summarization** | Tóm tắt phân cấp (Hierarchical) với ViT5 |
| **Translation** | Dịch thuật song ngữ Local MarianMT (VI ↔ EN) |
| **Relation Graph** | Trích xuất thực thể và quan hệ từ văn bản |
| **API** | FastAPI backend phục vụ inference |
| **UI** | Streamlit frontend tương tác |

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- GPU NVIDIA (khuyến nghị, tối thiểu 4GB VRAM) hoặc CPU
- Kết nối Internet (để tải model từ Hugging Face lần đầu)

### Bước 1: Clone hoặc vào thư mục dự án

```bash
# Linux/macOS
cd nlp-translation-summarization

# Windows PowerShell
Set-Location "d:\AI\DA\nlp-translation-summarization"
```

### Bước 2: Cài PyTorch

**GPU NVIDIA (khuyến nghị):**

```bash
# Linux/macOS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Windows PowerShell
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Kiểm tra GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.__version__)"
```

**CPU only:**

```bash
pip install torch torchvision torchaudio
```

### Bước 3: Cài dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra imports

```bash
python -c "from models.summarization import VIT5Summarizer, HierarchicalSummarizer; print('✅ OK')"
```

---

## Quick Start

### Chạy toàn bộ pipeline (từ đầu)

```bash
# Bước 1: Cào dữ liệu
python scripts/01_prepare_data.py --max-articles 20

# Bước 2: Tạo tóm tắt
python scripts/02_generate_summary.py

# Bước 3: Chia train/val
python scripts/03_split_data.py

# Bước 4: Dịch sang tiếng Anh
python scripts/04_translate_data.py

# Bước 5: Xem thống kê
python scripts/05_stats.py
```

### Chạy API server

```bash
python server.py
```

Server khởi động tại `http://localhost:8000`.

### Chạy UI

```bash
streamlit run UI/app.py
# hoặc
python -m streamlit run UI/app.py
```

UI mở tại `http://localhost:8501`.

---

## Pipeline dữ liệu

### Script 01 – Cào dữ liệu

```bash
python scripts/01_prepare_data.py --max-articles 50
```

| Tùy chọn | Mặc định | Mô tả |
|----------|---------|-------|
| `--max-articles` | `20` | Số bài tối đa mỗi nguồn |

Output: `data/raw_data.jsonl`

### Script 02 – Tạo tóm tắt

```bash
python scripts/02_generate_summary.py [--replace] [--batch-size N]
```

| Tùy chọn | Mặc định | Mô tả |
|----------|---------|-------|
| `--replace` | `False` | Xóa file cũ và tóm tắt lại từ đầu |
| `--batch-size` | `2` | Số bài xử lý mỗi batch |

Output: `data/summary_data.jsonl`

### Script 03 – Chia dữ liệu

```bash
python scripts/03_split_data.py [--replace] [--train_ratio 0.8]
```

Output: `data/processed/train.jsonl`, `data/processed/val.jsonl`

### Script 04 – Dịch thuật

```bash
python scripts/04_translate_data.py [--replace]
```

Output: `data/translated_data.jsonl`

### Script 05 – Thống kê

```bash
python scripts/05_stats.py
```

---

## API Usage

### Khởi động server

```bash
python server.py
```

### Endpoints

#### `GET /health`

Kiểm tra trạng thái server.

```bash
curl http://localhost:8000/health
```

#### `POST /api/process`

Xử lý văn bản: tóm tắt, dịch thuật, trích xuất quan hệ.

**Request body:**

```json
{
  "text": "Nội dung văn bản cần xử lý...",
  "source_lang": "vi",
  "target_lang": "en",
  "summary_mode": "original",
  "max_input_length": 1024
}
```

| Trường | Kiểu | Mặc định | Mô tả |
|--------|------|---------|-------|
| `text` | string | bắt buộc | Văn bản đầu vào |
| `source_lang` | `"vi"` \| `"en"` | `"vi"` | Ngôn ngữ nguồn |
| `target_lang` | `"vi"` \| `"en"` | `"en"` | Ngôn ngữ đích |
| `summary_mode` | `"original"` \| `"dpo"` \| `"both"` | `"original"` | Chế độ tóm tắt |
| `max_input_length` | integer | `1024` | Độ dài token đầu vào tối đa |

**Response:**

```json
{
  "summary_vi": "Tóm tắt tiếng Việt...",
  "summary_en": "English summary...",
  "summary_vi_original": "Bản gốc tiếng Việt...",
  "summary_vi_dpo": "Bản DPO tiếng Việt...",
  "summary_en_original": "Original English summary...",
  "translated_text_en": "Translated text...",
  "entities": [],
  "metadata": {
    "auto_hierarchical": false,
    "input_tokens": 256
  }
}
```

**Ví dụ với curl:**

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hội nghị thượng đỉnh G7 năm nay diễn ra tại Italy với sự tham dự của lãnh đạo các nước...",
    "source_lang": "vi",
    "target_lang": "en",
    "summary_mode": "original"
  }'
```

**Ví dụ với Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/process",
    json={
        "text": "Văn bản cần xử lý...",
        "source_lang": "vi",
        "summary_mode": "original"
    }
)
result = response.json()
print(result["summary_vi"])
```

---

## UI Usage

### Khởi động

```bash
streamlit run UI/app.py
```

### Quy trình sử dụng

1. Mở trình duyệt tại `http://localhost:8501`
2. Nhập văn bản vào ô nội dung
3. Chọn ngôn ngữ đầu vào (`Tiếng Việt` hoặc `English`)
4. Chọn chế độ tóm tắt (`Original`, `DPO`, hoặc `Both`)
5. Bấm **Process**
6. Xem kết quả: tóm tắt, bản dịch, và đồ thị quan hệ

---

## Cấu trúc dự án

```
nlp-translation-summarization/
├── data/                           # Dữ liệu & Kết quả
│   ├── raw_data.jsonl              # Output Crawler (script 01)
│   ├── summary_data.jsonl          # Output Summarizer (script 02)
│   ├── translated_data.jsonl       # Output Translator (script 04)
│   └── processed/                  # Dữ liệu đã xử lý
│       ├── train.jsonl             # Tập huấn luyện (80%)
│       └── val.jsonl               # Tập kiểm tra (20%)
├── models/                         # Models & Logic
│   ├── summarization/              # Module tóm tắt
│   │   ├── __init__.py             # Public API: VIT5Summarizer, HierarchicalSummarizer
│   │   ├── vit5_wrapper.py         # VIT5Summarizer class
│   │   ├── hierarchical.py         # HierarchicalSummarizer class
│   │   ├── prompting.py            # wrap_vi_like_training helper
│   │   └── config.py               # Cấu hình hyperparameters
│   └── translation/                # Module dịch thuật
│       └── inference.py            # TranslationModel class
├── scripts/                        # Data Pipeline
│   ├── 01_prepare_data.py          # Crawler
│   ├── 02_generate_summary.py      # Tạo nhãn tóm tắt
│   ├── 03_split_data.py            # Chia train/val
│   ├── 04_translate_data.py        # Dịch sang tiếng Anh
│   └── 05_stats.py                 # Thống kê dữ liệu
├── relation_graph/                 # Trích xuất quan hệ
│   └── extractor.py                # Regex-based NER & Relation Extraction
├── UI/                             # Frontend Streamlit
│   └── app.py
├── tests/                          # Unit tests
│   ├── test_hierarchical.py
│   ├── test_ner.py
│   ├── test_relation_extraction.py
│   └── test_translation.py
├── server.py                       # Backend FastAPI
├── requirements.txt                # Python dependencies
└── PROJECT_STRUCTURE.md            # Tài liệu kỹ thuật chi tiết
```

---

## Models Reference

### VIT5Summarizer

Tóm tắt văn bản tiếng Việt với `VietAI/vit5-base`.

```python
from models.summarization import VIT5Summarizer

summarizer = VIT5Summarizer(model_name="VietAI/vit5-base", lang_label="vi")

result = summarizer.summarize(
    text="Văn bản cần tóm tắt...",
    max_input_length=1024,
    max_new_tokens=150,
    min_new_tokens=20,
    num_beams=3,
    length_penalty=1.5,
)
print(result["summary"])
```

### HierarchicalSummarizer

Tóm tắt văn bản dài bằng cách chia nhỏ thành các chunk.

```python
from models.summarization import HierarchicalSummarizer

summarizer = HierarchicalSummarizer()
result = summarizer.summarize("Văn bản rất dài...")
print(result["summary"])
```

### wrap_vi_like_training

Chuẩn bị đầu vào theo định dạng giống lúc huấn luyện.

```python
from models.summarization import wrap_vi_like_training

formatted = wrap_vi_like_training(
    text="Nội dung bài báo...",
    title="Tiêu đề bài báo",
    sapo="Tóm tắt nhanh..."
)
```

### Tóm tắt tiếng Anh

```python
from models.summarization import VIT5Summarizer

summarizer_en = VIT5Summarizer(model_name="google/mt5-base", lang_label="en")
result = summarizer_en.summarize("Text to summarize...")
print(result["summary"])
```

---

## Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'models'`

Thêm project root vào `PYTHONPATH`:

```bash
# Linux/macOS
export PYTHONPATH=$(pwd)
python scripts/02_generate_summary.py

# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
py -3 scripts/02_generate_summary.py
```

Hoặc chạy script từ thư mục gốc dự án (các script đã tự xử lý điều này nội bộ).

### Lỗi `ImportError: cannot import name 'DualVIT5Summarizer'`

`DualVIT5Summarizer` đã bị loại bỏ. Thay bằng `VIT5Summarizer`:

```python
# ❌ Cũ
from models.summarization import DualVIT5Summarizer

# ✅ Mới
from models.summarization import VIT5Summarizer
summarizer = VIT5Summarizer("VietAI/vit5-base", lang_label="vi")
```

### Lỗi CUDA Out of Memory

Giảm batch size:

```bash
python scripts/02_generate_summary.py --batch-size 1
```

### Lỗi UnicodeEncodeError trên Windows

Đã được xử lý tự động trong các script. Nếu vẫn gặp lỗi:

```powershell
$env:PYTHONIOENCODING = "utf-8"
py -3 scripts/02_generate_summary.py
```

### Summary bị corrupt (`extra_id_*` tokens)

Đây là lỗi của mô hình T5 khi dùng sai chế độ. Đã được xử lý bằng `_clean_output()` trong `VIT5Summarizer`. Nếu vẫn xuất hiện, chạy lại với `--replace`:

```bash
python scripts/02_generate_summary.py --replace
```

### Tải model chậm

Lần đầu tải model từ Hugging Face có thể mất 5–15 phút tùy tốc độ mạng. Model được cache tại:

- Linux/macOS: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<user>\.cache\huggingface\hub\`

---

## Cải tiến tương lai

- [ ] **Fine-tuning VIT5** trên dữ liệu báo chí tiếng Việt nội bộ
- [ ] **DPO (Direct Preference Optimization)** để cải thiện chất lượng tóm tắt
- [ ] **NER tích hợp** cho trường `entities` trong API response
- [ ] **Dịch thuật EN→VI** bổ sung (hiện chỉ hỗ trợ VI→EN)
- [ ] **Đánh giá tự động** với ROUGE, BERTScore
- [ ] **Docker deployment** để đơn giản hóa cài đặt
- [ ] **Batch API endpoint** cho xử lý nhiều văn bản cùng lúc

---

## Lưu ý

- `requirements.txt` ghim phiên bản `transformers==4.57.6` để tránh lỗi tokenizer với `VietAI/vit5-base`.
- GPU 4GB đủ cho inference; khi train ViT5-base nếu hết VRAM thì giảm `batch_size`.
- Streamlit phải chạy bằng `python -m streamlit run UI/app.py`.
- Trường `entities` trong API hiện trả về danh sách rỗng (NER chưa tích hợp).
- Trường `translated_text_en` hiện trả lại văn bản đầu vào (chưa nối model dịch riêng).