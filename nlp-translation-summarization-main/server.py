from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from relation_graph import build_relation_graph
from models.summarization import HierarchicalSummarizer, VIT5Summarizer, wrap_vi_like_training
from models.translation.inference import TranslationModel  # ✅ FIX: Import đúng

PROJECT_ROOT = Path(__file__).resolve().parent

# ✅ FIX: Use pre-trained models instead of checkpoint paths
SUMMARY_MODELS = {
    "vi": "VietAI/vit5-base",
    "en": "google/mt5-base",
}

_SPACE_RE = re.compile(r"\s+")
_TEXT_NOISE_RE = re.compile(r"[<>\[\]{}|~`^]+")
_REPEAT_PUNCT_RE = re.compile(r"([,.;:!?])\1+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

app = FastAPI(
    title="NLP Translation & Summarization API",
    version="1.0.0",
    description="Vietnamese NLP: Translation, Summarization, Relation Extraction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    """Request model for text processing"""
    text: str = Field(..., min_length=1, description="Input text to summarize or analyze")
    source_lang: Literal["vi", "en"] = Field(default="vi", description="Source language")
    target_lang: Literal["vi", "en"] = Field(default="en", description="Target language")
    max_input_length: int = Field(default=1024, ge=128, le=2048, description="Max input length")
    max_new_tokens: int = Field(default=512, ge=32, le=1024, description="Max output tokens")
    min_new_tokens: int = Field(default=-1, ge=-1, le=512, description="Min output tokens (-1 for auto)")
    num_beams: int = Field(default=4, ge=1, le=8, description="Number of beams")
    length_penalty: float = Field(default=1.15, ge=0.8, le=2.0, description="Length penalty")
    chunk_size_words: int = Field(default=240, ge=120, le=400, description="Chunk size")
    chunk_overlap_words: int = Field(default=40, ge=0, le=120, description="Chunk overlap")


# ==================== UTILITY FUNCTIONS ====================

def _normalize_text(text: str) -> str:
    """Normalize text by removing noise and extra whitespace"""
    text = text.replace("\u00a0", " ")
    text = _TEXT_NOISE_RE.sub(" ", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def _tokens_lower(text: str) -> list[str]:
    """Extract and lowercase tokens from text"""
    return [t.lower() for t in _WORD_RE.findall(text)]


def _lexical_overlap(summary: str, source: str) -> float:
    """Calculate lexical overlap between summary and source"""
    s_tokens = {t for t in _tokens_lower(summary) if len(t) > 2}
    src_tokens = {t for t in _tokens_lower(source) if len(t) > 2}
    if not s_tokens:
        return 0.0
    return len(s_tokens & src_tokens) / max(1, len(s_tokens))


# ==================== CACHED MODELS ====================

@lru_cache(maxsize=2)
def get_summarization_pipeline(lang: str) -> VIT5Summarizer:
    """Get cached summarization model"""
    model_name = SUMMARY_MODELS.get(lang)
    
    if not model_name:
        raise ValueError(f"Unsupported language: {lang}")
    
    print(f"[Model] Loading {lang} model: {model_name}")
    
    try:
        return VIT5Summarizer(model_name, lang_label=lang)
    except Exception as e:
        print(f"[Model] ❌ Error: {e}")
        raise


@lru_cache(maxsize=1)
def get_translation_model() -> TranslationModel:
    """Get cached translation model (singleton)"""
    print("[Model] Initializing TranslationModel...")
    return TranslationModel()


# ==================== TEXT PROCESSING ====================

def _summarize_text(
    lang: str,
    text: str,
    request: ProcessRequest,
) -> dict[str, object]:
    """Summarize text in specified language"""
    
    try:
        summarizer = get_summarization_pipeline(lang)
    except Exception as e:
        print(f"[Summary] ❌ Error loading model: {e}")
        return {
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "chunk_count": 0,
            "chunk_summaries": [],
            "levels": 0,
            "auto_hierarchical": False,
            "input_tokens": len(text.split()),
            "error": str(e)
        }

    # Calculate token limits
    if request.min_new_tokens == -1:
        input_tokens = len(summarizer.tokenizer.encode(text))
        min_tokens = max(30, int(input_tokens * 0.15))
        max_tokens = min(512, int(input_tokens * 0.4))
        max_tokens = max(min_tokens + 1, max_tokens)
    else:
        min_tokens = request.min_new_tokens
        max_tokens = request.max_new_tokens

    # Decide between hierarchical and direct summarization
    input_tokens = len(summarizer.tokenizer.encode(text))
    hierarchy_trigger = max(64, request.max_input_length - 32)
    use_hierarchical = input_tokens > hierarchy_trigger

    try:
        if use_hierarchical:
            print(f"[Summary] Hierarchical ({input_tokens} > {hierarchy_trigger})")
            hierarchical = HierarchicalSummarizer(summarizer)
            summary_text = hierarchical.summarize(
                text,
                max_input_length=request.max_input_length,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_beams=request.num_beams,
                length_penalty=request.length_penalty,
            )
            chunk_count = -1
            chunk_summaries = []
            levels = -1
        else:
            print(f"[Summary] Direct ({input_tokens} ≤ {hierarchy_trigger})")
            result = summarizer.summarize(
                text,
                max_input_length=request.max_input_length,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_beams=request.num_beams,
                length_penalty=request.length_penalty,
            )
            summary_text = str(result.get("summary", ""))
            chunk_count = 1
            chunk_summaries = [summary_text]
            levels = 1

        return {
            "summary": summary_text,
            "chunk_count": chunk_count,
            "chunk_summaries": chunk_summaries,
            "levels": levels,
            "auto_hierarchical": use_hierarchical,
            "input_tokens": input_tokens,
        }
    
    except Exception as e:
        print(f"[Summary] ❌ Error: {e}")
        return {
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "chunk_count": 0,
            "chunk_summaries": [],
            "levels": 0,
            "auto_hierarchical": False,
            "input_tokens": input_tokens,
            "error": str(e)
        }


def _translate_text(source_lang: str, target_lang: str, text: str) -> tuple[str, str]:
    """Translate text between languages"""
    
    if source_lang == target_lang:
        return text, "Không cần dịch (ngôn ngữ giống nhau)."
    
    if not ((source_lang == "vi" and target_lang == "en") or 
            (source_lang == "en" and target_lang == "vi")):
        return text, f"Không hỗ trợ: {source_lang} → {target_lang}"
    
    try:
        translator = get_translation_model()
        
        print(f"[Translation] {source_lang} → {target_lang}")
        
        if source_lang == "vi" and target_lang == "en":
            translated_text = translator.translate_vi_to_en(text)
        else:
            translated_text = translator.translate_en_to_vi(text)
        
        if not translated_text:
            return text, "Lỗi dịch: kết quả rỗng"
        
        return translated_text, f"✓ Dịch thành công"
    
    except Exception as e:
        print(f"[Translation] ❌ Error: {e}")
        return text, f"Lỗi dịch: {str(e)}"


def _entities_from_graph(graph: dict[str, object]) -> list[dict[str, object]]:
    """Extract entity information from relation graph"""
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    
    return [
        {
            "text": node.get("label", ""),
            "type": node.get("type", "UNKNOWN"),
            "id": node.get("id", ""),
        }
        for node in nodes
    ]


# ==================== API ENDPOINTS ====================

@app.get("/health")
def health() -> dict[str, object]:
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "API is running",
        "project_root": str(PROJECT_ROOT),
        "supported_languages": list(SUMMARY_MODELS.keys()),
    }


@app.get("/")
def root() -> dict[str, object]:
    """Root endpoint with API information"""
    return {
        "api": "NLP Translation & Summarization",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "process": "POST /api/process",
            "docs": "GET /docs",
        },
        "supported_languages": ["vi", "en"],
    }


@app.post("/api/process")
def process_text(request: ProcessRequest) -> dict[str, object]:
    """
    Main text processing endpoint
    
    Process:
    1. Text normalization
    2. Summarization in source language
    3. Translation to target language
    4. Summarization in target language
    5. Relation graph extraction
    """
    
    print(f"\n{'='*80}")
    print(f"[Request] {request.source_lang} → {request.target_lang}")
    print(f"[Request] Length: {len(request.text)} characters")
    print(f"{'='*80}")
    
    # Normalize text
    text = _normalize_text(request.text)
    display_text = text
    wrap_vi_enabled = os.getenv("WRAP_VI_LIKE_TRAINING", "1").strip() not in {"0", "false", "False"}

    summary_vi = ""
    summary_en = ""
    summary_vi_original = ""
    summary_en_original = ""

    source_summary_meta: dict[str, object] = {"chunk_count": 0, "chunk_summaries": [], "levels": 0}

    # Step 1: Summarize in source language
    print("\n[Step 1/4] Summarizing in source language...")
    summary_input_text = text
    if request.source_lang == "vi":
        if wrap_vi_enabled:
            summary_input_text = wrap_vi_like_training(text)
        source_summary_meta = _summarize_text("vi", summary_input_text, request)
        summary_vi_original = str(source_summary_meta.get("summary", ""))
        summary_vi = summary_vi_original
    else:
        source_summary_meta = _summarize_text("en", summary_input_text, request)
        summary_en_original = str(source_summary_meta.get("summary", ""))
        summary_en = summary_en_original

    # Step 2: Translate text
    print("\n[Step 2/4] Translating...")
    translated_text, translation_note = _translate_text(
        request.source_lang,
        request.target_lang,
        display_text
    )

    # Step 3: Summarize in target language if different
    target_summary_meta: dict[str, object] | None = None
    if request.target_lang != request.source_lang and translated_text and translated_text != text:
        print("\n[Step 3/4] Summarizing in target language...")
        if request.target_lang == "vi":
            if wrap_vi_enabled:
                translated_text = wrap_vi_like_training(translated_text)
            target_summary_meta = _summarize_text("vi", translated_text, request)
            summary_vi_original = str(target_summary_meta.get("summary", ""))
            summary_vi = summary_vi_original
        else:
            target_summary_meta = _summarize_text("en", translated_text, request)
            summary_en_original = str(target_summary_meta.get("summary", ""))
            summary_en = summary_en_original
    else:
        print("\n[Step 3/4] Skipping target language summarization")

    # Step 4: Build relation graph
    print("\n[Step 4/4] Building relation graph...")
    graph_source_text = text
    relation_graph = build_relation_graph(graph_source_text)

    # Build response
    response = {
        "original_text": display_text,
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
        "original_text_vi": display_text if request.source_lang == "vi" else "",
        "original_text_en": display_text if request.source_lang == "en" else "",
        "translated_text": translated_text,
        "translated_text_vi": translated_text if request.target_lang == "vi" else "",
        "translated_text_en": translated_text if request.target_lang == "en" else "",
        "summary_vi": summary_vi,
        "summary_en": summary_en,
        "summary_vi_original": summary_vi_original,
        "summary_en_original": summary_en_original,
        "relation_graph": relation_graph,
        "entities": _entities_from_graph(relation_graph),
        "metadata": {
            "summary_mode": "original",
            "source_summary": source_summary_meta,
            "target_summary": target_summary_meta,
            "graph_source_text": graph_source_text,
            "translation_note": translation_note,
        },
    }
    
    print(f"\n[Done] ✅ Request processed")
    print(f"{'='*80}\n")
    
    return response


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print("NLP TRANSLATION & SUMMARIZATION API")
    print("="*80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Starting at http://localhost:8000")
    print(f"API Docs at http://localhost:8000/docs")
    print("="*80 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)