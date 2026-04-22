from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import re
from functools import lru_cache

# Import cấu hình từ file config.py của dự án
from api.config import CORS_ORIGINS, SUMMARY_CHECKPOINTS, PROJECT_ROOT

# Import Pydantic models từ file models.py
from api.models import (
    TextRequest, TranslationRequest, ProcessRequest, 
    SummaryResponse, TranslationResponse, EntitiesResponse, HealthResponse
)

# Import logic AI (Tùy chỉnh lại đường dẫn import nếu nhóm bạn để file ở chỗ khác)
from relation_graph import build_relation_graph
from summarization import HierarchicalSummarizer, VIT5Summarizer
from translation import LocalMarianTranslator, TranslationUnavailableError

SPACE_RE = re.compile(r"\s+")

app = FastAPI(title="Story Summarization API", version="2.0.0")

# Dùng CORS từ config.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.preload_status = {"vi_pipeline": "not_loaded"}

# --- HELPER FUNCTIONS ---
def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()

# Dùng đường dẫn Model từ config.py
@lru_cache(maxsize=1)
def get_vi_story_pipeline() -> HierarchicalSummarizer:
    checkpoint_dir = SUMMARY_CHECKPOINTS.get("vi")
    if not checkpoint_dir or not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing Vietnamese summarization checkpoint at: {checkpoint_dir}")
    return HierarchicalSummarizer(VIT5Summarizer(checkpoint_dir, lang_label="vi"))

@lru_cache(maxsize=1)
def get_en_fallback_pipeline() -> HierarchicalSummarizer | None:
    checkpoint_dir = SUMMARY_CHECKPOINTS.get("en")
    if not checkpoint_dir or not checkpoint_dir.exists():
        return None
    return HierarchicalSummarizer(VIT5Summarizer(checkpoint_dir, lang_label="en"))

@lru_cache(maxsize=4)
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

def summarize_with_pipeline(pipeline: HierarchicalSummarizer, text: str, request: ProcessRequest) -> dict:
    return pipeline.summarize_long_text(
        text,
        max_input_length=request.max_input_length,
        max_new_tokens=request.max_new_tokens,
        min_new_tokens=request.min_new_tokens,
        num_beams=request.num_beams,
        length_penalty=request.length_penalty,
        chunk_size_words=request.chunk_size_words,
        chunk_overlap_words=request.chunk_overlap_words,
    )

def translate_text(source_lang: str, target_lang: str, text: str) -> tuple[str, str, bool]:
    if source_lang == target_lang:
        return text, "Khong can dich vi ngon ngu nguon va dich giong nhau.", True
    translator = get_translator(source_lang, target_lang)
    if translator is None:
        return text, f"Khong co model dich cuc bo cho huong {source_lang}->{target_lang}.", False
    translated = translator.translate(text)
    return str(translated["translated_text"]), f"Dich bang model {translated['model_name']}.", True

def entities_from_graph(graph: dict) -> list:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    return [{"text": node.get("label", ""), "type": "CHARACTER", "mentions": node.get("mentions", 0)} for node in nodes]

# --- APP EVENTS ---
@app.on_event("startup")
def preload_models() -> None:
    try:
        get_vi_story_pipeline()
        app.state.preload_status = {"vi_pipeline": "loaded"}
    except Exception as exc:
        app.state.preload_status = {"vi_pipeline": "failed", "error": str(exc)}

# --- ENDPOINTS ---
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "preload_status": app.state.preload_status}

@app.post("/api/summarize-vi", response_model=SummaryResponse)
def summarize_vi(request: ProcessRequest):
    pipeline = get_vi_story_pipeline()
    text = normalize_text(request.text)
    try:
        result = summarize_with_pipeline(pipeline, text, request)
        return {"summary": str(result["summary"]), "metadata": {"method": "direct_vi"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize-en", response_model=SummaryResponse)
def summarize_en(request: ProcessRequest):
    pipeline = get_en_fallback_pipeline()
    if not pipeline:
        raise HTTPException(status_code=503, detail="English summarization model not available.")
    text = normalize_text(request.text)
    try:
        result = summarize_with_pipeline(pipeline, text, request)
        return {"summary": str(result["summary"]), "metadata": {"method": "direct_en"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate-vi-to-en", response_model=TranslationResponse)
def translate_vi_to_en(request: TranslationRequest):
    translated_text, note, ok = translate_text("vi", "en", request.text)
    if not ok:
        raise HTTPException(status_code=503, detail=note)
    return {"translated_text": translated_text, "note": note}

@app.post("/api/translate-en-to-vi", response_model=TranslationResponse)
def translate_en_to_vi(request: TranslationRequest):
    translated_text, note, ok = translate_text("en", "vi", request.text)
    if not ok:
        raise HTTPException(status_code=503, detail=note)
    return {"translated_text": translated_text, "note": note}

@app.post("/api/extract-entities", response_model=EntitiesResponse)
def extract_entities(request: TextRequest):
    text = normalize_text(request.text)
    graph = build_relation_graph(text)
    entities = entities_from_graph(graph)
    return {"entities": entities}