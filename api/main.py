from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import re
from functools import lru_cache

# Import cấu hình từ file config.py của dự án
from api.config import CORS_ORIGINS, SUMMARY_CHECKPOINTS, PROJECT_ROOT

# Import Pydantic models từ file models.py (Đã bổ sung ProcessResponse)
from api.models import (
    TextRequest, TranslationRequest, ProcessRequest, 
    SummaryResponse, TranslationResponse, EntitiesResponse, HealthResponse, ProcessResponse
)

# Import logic AI 
from relation_graph import build_relation_graph
from summarization import HierarchicalSummarizer, VIT5Summarizer
from translation import (
    CloudflareWorkersTranslator,
    LocalMarianTranslator,
    TranslationUnavailableError,
)

# TODO: Khi nào Thành viên 3 up file NER lên, hãy bỏ dấu # ở dòng dưới để import
# from models.ner.inference import extract_entities

SPACE_RE = re.compile(r"\s+")
VI_QUANTITY = r"(?:\d+|một|hai|ba|bốn|bon|năm|nam|sáu|sau|bảy|bay|tám|tam|chín|chin|mười|muoi)"
DEAD_FACT_RE = re.compile(
    rf"({VI_QUANTITY}\s+người(?:\s+[^\W\d_]+){{0,3}}\s+(?:tử\s+vong|thiệt\s+mạng)[^.;\n]*)",
    flags=re.IGNORECASE,
)
INJURED_FACT_RE = re.compile(
    rf"({VI_QUANTITY}\s+người(?:\s+[^\W\d_]+){{0,3}}\s+bị\s+thương[^.;\n]*)",
    flags=re.IGNORECASE,
)
CAUSE_FACT_RE = re.compile(
    r"(nguyên\s+nhân[^.;\n]{0,180})",
    flags=re.IGNORECASE,
)

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


def _append_unique_fact(parts: list[str], candidate: str, current_summary: str) -> None:
    c = normalize_text(candidate).strip(" .;,:")
    if not c:
        return
    if c[:1].isalpha():
        c = c[:1].upper() + c[1:]
    c_lower = c.lower()
    if c_lower in current_summary.lower():
        return
    if any(c_lower == p.lower() for p in parts):
        return
    parts.append(c)


def enrich_vi_summary_with_key_facts(source_text: str, summary: str | None) -> str | None:
    if not summary:
        return summary

    summary_clean = normalize_text(summary)
    if not summary_clean:
        return summary

    additions: list[str] = []

    dead_match = DEAD_FACT_RE.search(source_text)
    if dead_match and "tử vong" not in summary_clean.lower() and "thiệt mạng" not in summary_clean.lower():
        _append_unique_fact(additions, dead_match.group(1), summary_clean)

    injured_match = INJURED_FACT_RE.search(source_text)
    if injured_match and "bị thương" not in summary_clean.lower():
        _append_unique_fact(additions, injured_match.group(1), summary_clean)

    cause_match = CAUSE_FACT_RE.search(source_text)
    if cause_match and "nguyên nhân" not in summary_clean.lower():
        _append_unique_fact(additions, cause_match.group(1), summary_clean)

    if not additions:
        return summary_clean

    suffix = "; ".join(additions).strip()
    if not suffix.endswith("."):
        suffix += "."
    return f"{summary_clean} {suffix}"

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

@lru_cache(maxsize=8)
def get_translator(source_lang: str, target_lang: str, backend: str | None = None):
    backend = (backend or os.getenv("TRANSLATION_BACKEND", "auto")).strip().lower()
    allow_remote_models = os.getenv("ALLOW_REMOTE_TRANSLATION_MODELS", "").strip() == "1"
    cloudflare_model = os.getenv("CLOUDFLARE_TRANSLATION_MODEL", "").strip() or "@cf/meta/m2m100-1.2b"

    def _build_local() -> LocalMarianTranslator | None:
        try:
            return LocalMarianTranslator(
                source_lang=source_lang,
                target_lang=target_lang,
                local_files_only=not allow_remote_models,
            )
        except TranslationUnavailableError:
            return None

    def _build_cloudflare() -> CloudflareWorkersTranslator | None:
        try:
            return CloudflareWorkersTranslator(
                source_lang=source_lang,
                target_lang=target_lang,
                model_name=cloudflare_model,
            )
        except TranslationUnavailableError:
            return None

    if backend == "local":
        return _build_local()
    if backend == "cloudflare":
        return _build_cloudflare()

    # auto: uu tien cloudflare neu co credentials; neu khong thi fallback local.
    return _build_cloudflare() or _build_local()

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
        carry_prev_summary=request.carry_prev_summary,
    )

def get_pipeline_for_lang(lang: str) -> HierarchicalSummarizer | None:
    if lang == "vi":
        return get_vi_story_pipeline()
    if lang == "en":
        return get_en_fallback_pipeline()
    return None

def translate_text(source_lang: str, target_lang: str, text: str) -> tuple[str, str, bool]:
    if source_lang == target_lang:
        return text, "Khong can dich vi ngon ngu nguon va dich giong nhau.", True
    backend = os.getenv("TRANSLATION_BACKEND", "auto").strip().lower()
    translator = get_translator(source_lang, target_lang, backend)
    if translator is None:
        return text, (
            f"Khong khoi tao duoc backend dich ({backend}) cho huong {source_lang}->{target_lang}. "
            "Hay kiem tra env (CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN) hoac model local."
        ), False
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

# CỔNG SỐ 6 MỚI THÊM VÀO: FULL PROCESSING (TÓM TẮT + DỊCH + THỰC THỂ)
@app.post("/api/process", response_model=ProcessResponse)
def process_full_workflow(request: ProcessRequest):
    text = normalize_text(request.text)
    
    try:
        source_pipeline = get_pipeline_for_lang(request.source_lang)
        if source_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=f"Summarization model not available for source_lang={request.source_lang}.",
            )

        source_summary_result = summarize_with_pipeline(source_pipeline, text, request)

        summary_vi = None
        summary_en = None
        if request.source_lang == "vi":
            summary_vi = enrich_vi_summary_with_key_facts(text, str(source_summary_result["summary"]))
        else:
            summary_en = str(source_summary_result["summary"])

        translated_text, translation_note, translation_ok = translate_text(
            request.source_lang,
            request.target_lang,
            text,
        )

        target_summary_result = None
        if request.target_lang != request.source_lang and translation_ok and translated_text.strip():
            target_pipeline = get_pipeline_for_lang(request.target_lang)
            if target_pipeline is not None:
                target_summary_result = summarize_with_pipeline(target_pipeline, translated_text, request)
                if request.target_lang == "vi":
                    summary_vi = enrich_vi_summary_with_key_facts(
                        translated_text,
                        str(target_summary_result["summary"]),
                    )
                else:
                    summary_en = str(target_summary_result["summary"])
            else:
                translation_note += f" Summarization model not available for target_lang={request.target_lang}."

        # TODO: Đổi thành `entities = extract_entities(text)` khi TV3 làm xong
        graph = build_relation_graph(text)
        entities = entities_from_graph(graph)
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "summary_vi": summary_vi,
            "summary_en": summary_en,
            "entities": entities,
            "relation_graph": graph,
            "metadata": {
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "translation_ok": translation_ok,
                "translation_note": translation_note,
                "source_summary_metadata": source_summary_result.get("metadata"),
                "target_summary_metadata": target_summary_result.get("metadata") if target_summary_result else None,
            },
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

@app.post("/api/extract-entities", response_model=EntitiesResponse)
def extract_entities_endpoint(request: TextRequest):
    text = normalize_text(request.text)
    try:
        # TODO: Đổi thành `entities = extract_entities(text)` khi TV3 làm xong
        graph = build_relation_graph(text)
        entities = entities_from_graph(graph)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
