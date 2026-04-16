# 🗺️ PROJECT MASTER CONTEXT: NLP Translation & Summarization

## 🎯 1. Mục tiêu dự án
- **Mô tả:** Hệ thống chuyên sâu về xử lý ngôn ngữ tự nhiên (NLP) cho tiếng Việt và tiếng Anh, tập trung vào:
    - Cào dữ liệu báo chí tự động (VnExpress, Vneconomy, v.v.).
    - Tóm tắt văn bản chất lượng cao (Sử dụng ViT5 và mT5-XLSum).
    - Dịch thuật song ngữ (Local Marian Translator).
    - Trích xuất graph quan hệ nhân vật từ văn bản.
- **Trạng thái hiện tại:** 
    - ✅ Đã tối ưu hóa Crawler với bộ lọc noise và chống trùng lặp.
    - ✅ Summarizer đã nâng cấp lên **GPU Batch Processing** (Tốc độ ~4.5s cho batch 2).
Hệ thống chuyên sâu về xử lý ngôn ngữ tự nhiên (NLP) đa nhiệm:
- **Crawler:** Cào báo tự động (VnExpress, Vneconomy, v.v.), lọc rác bằng Regex.
- **Summarization:** Tóm tắt phân cấp (Hierarchical) và tối ưu hóa DPO (Direct Preference Optimization).
- **Translation:** Dịch thuật song ngữ Local MarianMT (EN <-> VI).
- **Graph Extraction:** Trích xuất thực thể (Character) và quan hệ (Relation) dựa trên quy luật ngôn ngữ.

## 🏗️ 2. Sơ đồ cây thư mục (File Tree)
```text
nlp-translation-summarization/
├── data/                       # Dữ liệu & Kết quả
│   ├── raw_data.jsonl          # Output Crawler (01)
│   ├── summary_data.jsonl      # Output Summarizer (02) - Dataset gốc cho Training
│   └── processed/              # Data đã qua xử lý (Train/Val/Test)
├── scripts/                    # Pipeline xử lý dữ liệu (Data Pipeline)
│   ├── 01_prepare_data.py      # Crawler: BeautifulSoup + Noise Filter
│   ├── 01_1.generate_preference_data.py # DPO: Tạo dữ liệu Chosen/Rejected
│   ├── 02_generate_summary.py  # Labeler: Tạo nhãn bằng mT5 (GPU Batching)
│   ├── 03_split_data.py        # Dataset Splitter: Chia Train/Val (80/20)
│   ├── 04_translate_data.py    # Translator: Tạo dữ liệu song ngữ
│   └── 05_stats.py             # Statistics: Thống kê tập dữ liệu
├── models/                     # Checkpoints & Training/Inference Logic
│   ├── trainModelsAI/          # Các script huấn luyện mô hình (Training)
│   │   ├── train_vi.py          # Huấn luyện SFT Tiếng Việt
│   │   ├── train_en.py          # Huấn luyện SFT Tiếng Anh
│   │   ├── train_dpo_vi.py      # Huấn luyện DPO Tiếng Việt
│   │   └── train_vit5_summarize.py # Script huấn luyện ViT5 tổng quát
│   ├── summarization/          # Logic Inference & Cấu hình
│   │   ├── config.py            # Cấu hình Hyperparameters & Paths
│   │   ├── dataset.py           # DataLoader & Dataset format
│   │   ├── evaluate.py          # Script đánh giá model (ROUGE...)
│   │   ├── hierarchical.py      # Logic tóm tắt văn bản dài (Chunking)
│   │   ├── inference.py         # Logic suy luận core
│   │   ├── inference_vi.py      # Suy luận Tiếng Việt
│   │   ├── inference_en.py      # Suy luận Tiếng Anh
│   │   └── inference_vit5_summarize.py # Helper suy luận ViT5
│   ├── vit5-summarize-vi/      # Checkpoint ViT5 Gốc
│   └── vit5-summarize-vi-dpo/  # Checkpoint ViT5 DPO
├── summarization/              # Module wrapper (Legacy/Wrapper)
│   └── vit5_wrapper.py         # Wrapper chính cho ViT5 Inference
├── translation/                # Module dịch thuật (Local MarianMT)
├── relation_graph/             # Module trích xuất thực thể
│   └── extractor.py            # Regex-based NER & Relationship Extraction
├── UI/                         # Frontend Streamlit (Việt hóa)
├── server.py                   # Backend FastAPI
└── PROJECT_STRUCTURE.md        # File Master Context này
```

## 🧠 3. Logic Kỹ thuật & Luồng Dữ liệu (Deep Tech)

### 3.1. Cấu trúc Dataset (JSONL Schema)
- **raw_data.jsonl**: `{"url":..., "title":..., "sapo_vi":..., "content_vi":...}`.
- **summary_data.jsonl**: Thêm trường `summary_vi` (nhãn AI) và `text_vi` (đã format prefix).
- **train.jsonl / val.jsonl**: Dữ liệu cuối cùng dùng cho `Trainer`.

### 3.2. Summarization Strategy
- **Prompt Structure**: Prefix bắt buộc: `Tiêu đề: {title}\nTóm tắt nhanh: {sapo}\nNội dung:\n{content}`.
- **Hierarchical Summarization**: Nếu văn bản >1024 tokens -> Chia nhỏ (Chunk 400 tokens) -> Tóm tắt từng phần -> Tóm tắt lại Level 2.
- **Quality Control (QC)**:
    - Loại bỏ bài < 50 từ. Summary phải 15-200 từ.
    - Chống trùng (Overlap check) với Title/Sapo.

### 3.3. Character Relation Graph
- **Extraction**: Dùng Regex tìm `[Danh xưng] + Tên Riêng`.
- **Edges**: Có 9 loại quan hệ (`danh_bai`, `tan_cong`, `bao_ve`, `phan_boi`, `yeu_thuong`, v.v.).

### 3.4. Training Config (SFT & DPO)
- **Base Model**: `VietAI/vit5-base`.
- **DPO Params**: Epochs 0.1, LR 2e-6, Beta 0.3. So sánh output Gốc vs DPO để chọn nhãn tốt nhất.

## 🚀 4. Hướng dẫn vận hành (Deployment)

### 4.1. API Endpoints (server.py)
- **GET /health**: Kiểm tra trạng thái server và sự tồn tại của model checkpoints.
- **POST /api/process**: Xử lý text. Body: `{"text":str, "source_lang":"vi/en", "target_lang":"vi/en", "summary_mode":"original/dpo/both"}`.

### 4.2. Chạy dự án
- **Backend Server**: `python server.py` (Lắng nghe tại Cổng 8000).
- **Frontend UI**: `streamlit run UI/app.py` (Mặc định Cổng 8501).

## ⚠️ 5. Lưu ý quan trọng cho AI Models
- **Encoding Fix:** Luôn cấu trúc lại `sys.stdout` về UTF-8 trên Windows để tránh lỗi `UnicodeEncodeError` khi in tiếng Việt.
- **VRAM limit:** `BATCH_SIZE` trong `02_generate_summary.py` mặc định là `2`. Nếu GPU > 8GB, có thể tăng lên `4-8`.
- **DPO Checkpoints:** Nếu `models/vit5-summarize-vi-dpo` chưa có `best_checkpoint`, server sẽ tự động fallback về bản Gốc.

##WRAP_VI_LIKE_TRAINING dùng để giảm lệch train/infer.

- Lúc train, model thấy input dạng prompt báo chí: Tiêu đề... / Tóm tắt nhanh... / Nội dung... (kèm prefix summarize:).
- Nếu infer đưa text thô không theo format này, phân phối input khác hẳn lúc train, output dễ “lạ”/kém ổn định.
- Khi WRAP_VI_LIKE_TRAINING=1 (mặc định), server tự bọc text thô thành dạng gần giống train (tối thiểu Nội dung:\n...).
- Khi WRAP_VI_LIKE_TRAINING=0, server không bọc, tức model nhận raw text trực tiếp.