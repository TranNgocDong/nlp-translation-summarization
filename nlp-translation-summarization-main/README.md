# 🚀 NLP Translation & Summarization

Vietnamese NLP Pipeline with Translation, Summarization, NER, and Relation Extraction.

## Features

- **📰 Web Scraping**: Crawl Vietnamese news articles from multiple sources
- **📝 Summarization**: VIT5-based Vietnamese text summarization (with hierarchical support)
- **🌐 Translation**: Opus-MT Vietnamese ↔ English translation
- **🏷️ NER**: Named Entity Recognition for Vietnamese text
- **🔗 Relation Extraction**: Build knowledge graphs from text
- **🔧 REST API**: FastAPI backend with integrated NLP models
- **💻 Web UI**: Streamlit interface for easy interaction

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/TranNgocDong/nlp-translation-summarization.git
cd nlp-translation-summarization

# 2. Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# 3. Install PyTorch with CUDA (recommended for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## Quick Start

Run these steps from the **project root** directory:

### Step 1 — Scrape Data

```bash
python scripts/01_prepare_data.py --max-articles 20
```

Crawls Vietnamese news articles and saves them to `data/raw_data.jsonl`.

### Step 2 — Generate Summaries

```bash
python scripts/02_generate_summary.py --replace
```

Uses VIT5 model to summarize articles; saves to `data/summary_data.jsonl`.

### Step 3 — Split Dataset

```bash
python scripts/03_split_data.py
```

Splits data into train/val sets under `data/processed/`.

### Step 4 — Translate Data

```bash
python scripts/04_translate_data.py --replace
```

Translates Vietnamese summaries to English using Helsinki-NLP/Opus-MT.

### Step 5 — View Statistics

```bash
python scripts/05_stats.py
```

Prints word-count statistics for the processed dataset.

## API Endpoints

### Start Server

```bash
python server.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### `GET /health`

Returns server health status.

```bash
curl http://localhost:8000/health
```

### `POST /api/process`

Process text through the full NLP pipeline.

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nhân vật A đánh bại nhân vật B và lấy được bảo vật cổ.",
    "source_lang": "vi",
    "target_lang": "en",
    "quick_mode": false
  }'
```

**Response fields:**

| Field | Description |
|-------|-------------|
| `summary_vi` | Vietnamese summary |
| `summary_en` | English summary |
| `translated_text_en` | Translated text |
| `entities` | Named entities found |
| `relation_graph` | Knowledge graph (nodes & edges) |
| `metadata` | Processing metadata |

**`summary_mode` options** (pass in request body):

| Value | Behavior |
|-------|----------|
| `original` | Run original model only |
| `dpo` | Run DPO-tuned model only |
| `both` | Run both models (Vietnamese) |

> The API automatically applies **hierarchical summarization** when the input exceeds the model's `max_input_length`. This is reflected in the `metadata.auto_hierarchical` field.

## UI Usage

### Start Streamlit App

```bash
streamlit run UI/app.py
# Opens at http://localhost:8501
```

### Workflow

1. Enter Vietnamese text in the input box
2. Select the source language
3. Click **Process**
4. View summaries, translations, entity list, and relation graph
5. Download results as JSON or CSV

## Project Structure

```
nlp-translation-summarization/
├── data/                        # Data directory (git-ignored)
│   ├── raw_data.jsonl           # Scraped articles (step 01)
│   ├── summary_data.jsonl       # Summarized articles (step 02)
│   ├── translated_data.jsonl    # Translated articles (step 04)
│   └── processed/
│       ├── train.jsonl          # Training split (step 03)
│       └── val.jsonl            # Validation split (step 03)
├── models/                      # NLP model modules
│   ├── summarization/           # VIT5 summarization
│   │   ├── __init__.py
│   │   ├── vit5_wrapper.py
│   │   ├── hierarchical.py
│   │   └── prompting.py
│   ├── translation/             # Opus-MT translation
│   ├── ner/                     # Named Entity Recognition
│   └── relation_extraction/     # Relation extraction + graph
│       ├── graph_builder.py
│       └── inference.py
├── scripts/                     # Data pipeline scripts
│   ├── 01_prepare_data.py       # Web scraping
│   ├── 02_generate_summary.py   # Summarization
│   ├── 03_split_data.py         # Train/val split
│   ├── 04_translate_data.py     # Translation
│   └── 05_stats.py              # Dataset statistics
├── UI/                          # Streamlit frontend
│   └── app.py
├── server.py                    # FastAPI backend
├── relation_graph.py            # Relation graph pipeline
├── requirements.txt
└── README.md
```

## Models Reference

| Task | Model | Source | Notes |
|------|-------|--------|-------|
| Summarization (VI) | `VietAI/vit5-base` | Hugging Face | Vietnamese summarization |
| Summarization (EN) | `google/mt5-base` | Hugging Face | English summarization |
| Translation (VI→EN) | `Helsinki-NLP/opus-mt-vi-en` | Hugging Face | Auto-downloaded |
| NER | Vietnamese NER | underthesea | Rule + model based |
| Relation Extraction | Custom rule-based | This repo | Pattern matching |

Models are auto-downloaded from Hugging Face on first run. An internet connection is required.

## Troubleshooting

### `ModuleNotFoundError: No module named 'models'`

Always run scripts from the **project root**, not from inside the `scripts/` folder:

```bash
# ✅ Correct
python scripts/02_generate_summary.py

# ❌ Wrong
cd scripts && python 02_generate_summary.py
```

### CUDA out of memory

Reduce the batch size in the relevant script, or run in CPU mode by removing CUDA device selection.

### Model download fails

Check your internet connection. Models are cached in `~/.cache/huggingface/` after the first download.

### Unicode errors on Windows

The scripts automatically reconfigure `stdout` to UTF-8. If issues persist, set the environment variable:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## Performance

| Operation | CPU | GPU (RTX 3050) |
|-----------|-----|----------------|
| Summarization (per article) | ~30–60 s | ~5–10 s |
| Translation (per 100 words) | ~10–15 s | ~2–3 s |
| Full pipeline (per article) | ~60–90 s | ~15–20 s |

GPU acceleration requires PyTorch with CUDA (see [Installation](#installation)).

## Future Improvements

- [ ] Vietnamese → Chinese translation support
- [ ] Fine-tuning on domain-specific datasets
- [ ] Semantic search across the document corpus
- [ ] Batch API endpoint for multiple documents
- [ ] WebSocket support for streaming responses
- [ ] Integrate dedicated NER model for `entities` field

## Notes

- `requirements.txt` intentionally omits `torch` to avoid installing the CPU-only build. Install PyTorch manually as shown in [Installation](#installation).
- `transformers==4.57.6` is pinned to avoid tokenizer issues with `VietAI/vit5-base`.
- The `entities` field in API responses currently returns an empty list; NER integration is planned.
- Run `python server.py` from the project root so that module imports resolve correctly.