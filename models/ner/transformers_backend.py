from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import torch
from transformers import pipeline


class TransformersNERUnavailableError(RuntimeError):
    """Raised when Transformers NER cannot be loaded/used."""


_SPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").replace("\u00a0", " ")).strip()


def _map_group(label: str) -> str | None:
    s = (label or "").strip().upper()
    if not s:
        return None
    if "-" in s:
        s = s.split("-", 1)[1]  # B-PER -> PER
    return {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "MISC": "MISC",
        "PERSON": "PERSON",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
    }.get(s)


def _get_device_index() -> int:
    """
    Env:
      - NER_DEVICE=cpu -> -1
      - NER_DEVICE=cuda or cuda:0 -> 0
      - NER_DEVICE=cuda:1 -> 1
    Default: auto (cuda if available else cpu)
    """
    dev = (os.getenv("NER_DEVICE", "") or "").strip().lower()

    if dev == "cpu":
        return -1

    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            return -1
        if ":" in dev:
            try:
                return int(dev.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    return 0 if torch.cuda.is_available() else -1


def _max_chars() -> int:
    """
    Limit NER input length to reduce OOM/latency for very long docs.
    Override by env NER_MAX_CHARS.
    """
    try:
        v = int((os.getenv("NER_MAX_CHARS", "") or "").strip() or "4000")
    except ValueError:
        v = 4000
    return max(200, v)


@lru_cache(maxsize=1)
def _get_pipeline():
    model_name = os.getenv("NER_TRANSFORMERS_MODEL", "").strip()
    if not model_name:
        # Multilingual model (works out of the box). You can replace later with a Vietnamese NER model.
        model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"

    try:
        device_index = _get_device_index()
        print(f"[Transformers NER] model={model_name} device={device_index}")
        return pipeline(
            task="token-classification",
            model=model_name,
            aggregation_strategy="simple",
            device=_get_device_index(),  # -1 CPU, 0..N GPU index
        )
    except Exception as e:
        raise TransformersNERUnavailableError(str(e))


def extract_entities_transformers(text: str) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    normalized = normalized[: _max_chars()]

    ner = _get_pipeline()
    preds = ner(normalized)

    out: list[dict[str, Any]] = []
    for p in preds:
        group = _map_group(p.get("entity_group") or p.get("entity") or "")
        if not group:
            continue

        start = int(p.get("start", 0))
        end = int(p.get("end", 0))
        span = normalized[start:end].strip()
        if not span:
            continue

        out.append({"text": span, "type": group, "score": float(p.get("score", 0.0))})
    return out