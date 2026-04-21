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
from summarization import HierarchicalSummarizer, VIT5Summarizer, wrap_vi_like_training
from translation import LocalMarianTranslator, TranslationUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_CHECKPOINTS = {
    "vi": PROJECT_ROOT / "models" / "vit5-summarize-vi" / "best_checkpoint",
    "en": PROJECT_ROOT / "models" / "vit5-summarize-en" / "best_checkpoint",
}
_SPACE_RE = re.compile(r"\s+")
_TEXT_NOISE_RE = re.compile(r"[<>\[\]{}|~`^]+")
_REPEAT_PUNCT_RE = re.compile(r"([,.;:!?])\1+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

app = FastAPI(title="NLP Translation & Summarization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to summarize or analyze")
    source_lang: Literal["vi", "en"] = "vi"
    target_lang: Literal["vi", "en"] = "en"
    max_input_length: int = Field(default=1024, ge=128, le=2048)
    max_new_tokens: int = Field(default=512, ge=32, le=1024)
    min_new_tokens: int = Field(default=-1, ge=-1, le=512, description="Set to -1 for dynamic calculation (15-40% rule)")
    num_beams: int = Field(default=4, ge=1, le=8)
    length_penalty: float = Field(default=1.15, ge=0.8, le=2.0)
    chunk_size_words: int = Field(default=240, ge=120, le=400)
    chunk_overlap_words: int = Field(default=40, ge=0, le=120)


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _TEXT_NOISE_RE.sub(" ", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def _tokens_lower(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _lexical_overlap(summary: str, source: str) -> float:
    s_tokens = {t for t in _tokens_lower(summary) if len(t) > 2}
    src_tokens = {t for t in _tokens_lower(source) if len(t) > 2}
    if not s_tokens:
        return 0.0
    return len(s_tokens & src_tokens) / max(1, len(s_tokens))


@lru_cache(maxsize=2)
def get_summarization_pipeline(lang: str) -> VIT5Summarizer:
    checkpoint_dir = SUMMARY_CHECKPOINTS[lang]
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint for language '{lang}': {checkpoint_dir}")
    return VIT5Summarizer(checkpoint_dir, lang_label=lang)


@lru_cache(maxsize=2)
def get_translator(source_lang: str, target_lang: str) -> LocalMarianTranslator | None:
    allow_remote_models = os.getenv("ALLOW_REMOTE_TRANSLATION_MODELS", "").strip() == "1"
    try:
        return LocalMarianTranslator(
            source_lang=source_lang,
            target_lang=target_lang,
            local_files_only=not allow_remote_models,
        )
    except TranslationUnavailableError:
        return None


def _summarize_text(
    lang: str,
    text: str,
    request: ProcessRequest,
) -> dict[str, object]:
    model_handle = lang

    summarizer = get_summarization_pipeline(model_handle)

    if request.min_new_tokens == -1:
        input_tokens = len(summarizer.tokenizer.encode(text))
        min_tokens = max(30, int(input_tokens * 0.15))
        max_tokens = min(512, int(input_tokens * 0.4))
        max_tokens = max(min_tokens + 1, max_tokens)
    else:
        min_tokens = request.min_new_tokens
        max_tokens = request.max_new_tokens

    input_tokens = len(summarizer.tokenizer.encode(text))
    hierarchy_trigger = max(64, request.max_input_length - 32)
    use_hierarchical = input_tokens > hierarchy_trigger

    if use_hierarchical:
        hierarchical = HierarchicalSummarizer(summarizer)
        summary_text = hierarchical.summarize(
            text,
            max_input_length=request.max_input_length,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty,
        )
        chunk_count = -1  # nội bộ hierarchical tự chia chunk
        chunk_summaries = []
        levels = -1
    else:
        result = summarizer.summarize(
            text,
            max_input_length=request.max_input_length,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty,
        )
        summary_text = str(result["summary"])
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


def _translate_text(source_lang: str, target_lang: str, text: str) -> tuple[str, str]:
    if source_lang == target_lang:
        return text, "Không cần dịch vì ngôn ngữ nguồn và đích giống nhau."

    translator = get_translator(source_lang, target_lang)
    if translator is None:
        return text, "Không tìm thấy model dịch cục bộ, hệ thống trả lại văn bản gốc."

    translated = translator.translate(text)
    return str(translated["translated_text"]), f"Dịch bằng model {translated['model_name']}."


def _entities_from_graph(graph: dict[str, object]) -> list[dict[str, object]]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    return [
        {
            "text": node.get("label", ""),
            "type": "CHARACTER",
            "mentions": node.get("mentions", 0),
        }
        for node in nodes
    ]


def _factual_check(source_text: str, summary_text: str) -> tuple[bool, list[str]]:
    source_graph = build_relation_graph(source_text)
    summary_graph = build_relation_graph(summary_text)

    source_entities = {str(node["id"]) for node in source_graph.get("nodes", [])}
    summary_entities = {str(node["id"]) for node in summary_graph.get("nodes", [])}

    hallucinated = list(summary_entities - source_entities)
    return len(hallucinated) == 0, hallucinated


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "available_summaries": {lang: path.exists() for lang, path in SUMMARY_CHECKPOINTS.items()},
    }


@app.post("/api/process")
def process_text(request: ProcessRequest) -> dict[str, object]:
    text = _normalize_text(request.text)
    display_text = text
    wrap_vi_enabled = os.getenv("WRAP_VI_LIKE_TRAINING", "1").strip() not in {"0", "false", "False"}

    summary_vi = ""
    summary_en = ""
    summary_vi_original = ""
    summary_en_original = ""

    source_summary_meta: dict[str, object] = {"chunk_count": 0, "chunk_summaries": [], "levels": 0}

    summary_input_text = text
    if request.source_lang == "vi":
        if wrap_vi_enabled:
            summary_input_text = wrap_vi_like_training(text)
        source_summary_meta = _summarize_text("vi", summary_input_text, request)
        summary_vi_original = str(source_summary_meta["summary"])
        summary_vi = summary_vi_original

        is_factual, hallucinated = _factual_check(text, summary_vi)
        if not is_factual:
            warning_msg = f"[⚠️ CẢNH BÁO: Phát hiện thông tin/thực thể không có trong bản gốc: {', '.join(hallucinated)}]"
            summary_vi = f"{warning_msg}\n\n{summary_vi}"
            source_summary_meta["factual_status"] = "FAILED"
            source_summary_meta["hallucinated_entities"] = hallucinated
        else:
            source_summary_meta["factual_status"] = "PASSED"

    else:
        source_summary_meta = _summarize_text("en", summary_input_text, request)
        summary_en_original = str(source_summary_meta["summary"])
        summary_en = summary_en_original

    translated_text, translation_note = _translate_text(request.source_lang, request.target_lang, display_text)

    target_summary_meta: dict[str, object] | None = None
    if request.target_lang != request.source_lang and translated_text and translated_text != text:
        if request.target_lang == "vi":
            if wrap_vi_enabled:
                translated_text = wrap_vi_like_training(translated_text)
            target_summary_meta = _summarize_text("vi", translated_text, request)
            summary_vi_original = str(target_summary_meta["summary"])
            summary_vi = summary_vi_original
        else:
            target_summary_meta = _summarize_text("en", translated_text, request)
            summary_en_original = str(target_summary_meta["summary"])
            summary_en = summary_en_original

    graph_source_text = text
    relation_graph = build_relation_graph(graph_source_text)

    response = {
        "original_text": display_text,
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
        "original_text_vi": display_text if request.source_lang == "vi" else "",
        "original_text_en": display_text if request.source_lang == "en" else "",
        "translated_text": translated_text,
        "translated_text_vi": translated_text if request.target_lang == "vi" else "",
        "translated_text_en": translated_text if request.target_lang == "en" else "",
        # Backward-compatible fields (single active summary)
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
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
