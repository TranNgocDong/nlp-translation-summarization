from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable

try:
    # pip install underthesea
    from underthesea import ner as _ud_ner
except Exception:  # pragma: no cover
    _ud_ner = None


# ----------------------------
# Normalization helpers
# ----------------------------
_SPACE_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"\s*[-–—]\s*")

_VI_UPPER = "A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
_WORD_RE = re.compile(
    rf"[{_VI_UPPER}a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9]+"
)


class NERUnavailableError(RuntimeError):
    """Raised when Underthesea NER is not installed/available."""


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    text = _DASH_RE.sub("-", text)
    return text


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _titleish(text: str) -> str:
    """Normalize casing lightly to reduce duplicates (công an vs Công an)."""
    t = _SPACE_RE.sub(" ", text.strip())
    if not t:
        return t
    parts = t.split(" ")
    out = []
    for p in parts:
        if p.isupper():  # keep acronyms
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)


# ----------------------------
# Generic noise filters (for any entity)
# ----------------------------
_STOP_START_WORDS = {
    "về",
    "tại",
    "trước",
    "sau",
    "khi",
    "nhưng",
    "và",
    "của",
    "với",
    "do",
    "để",
    "trong",
    "trên",
    "dưới",
    "theo",
    "from",
    "to",
    "in",
    "on",
    "at",
    "about",
    "of",
    "with",
    "and",
    "but",
    "before",
    "after",
    "when",
}


def _is_all_caps_phrase(s: str) -> bool:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    # Consider "all caps" if there is no lowercase letter
    return not any(ch.islower() for ch in letters)


def _should_drop_entity(text: str, typ: str) -> bool:
    t = _SPACE_RE.sub(" ", (text or "").strip()).strip(" .,:;")
    if not t:
        return True
    ws = _words(t)
    if not ws:
        return True

    first = ws[0].lower()
    if first in _STOP_START_WORDS:
        return True

    # drop headline-like ALL CAPS long phrases
    if _is_all_caps_phrase(t) and len(ws) >= 4:
        return True

    # drop overly long clause fragments
    if len(ws) >= 12:
        return True

    return False


# ----------------------------
# Underthesea tag mapping + merge
# ----------------------------
def _map_underthesea_tag(tag: str) -> str | None:
    if not tag or tag == "O":
        return None

    if "-" in tag:
        _, base = tag.split("-", 1)
    else:
        base = tag

    base = base.upper()
    return {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "MISC": "MISC",
    }.get(base, base)


_ORG_KEYWORDS = (
    "bệnh viện",
    "bv",
    "công an",
    "cảnh sát",
    "pccc",
    "phòng cháy",
    "cứu nạn",
    "cứu hộ",
    "ủy ban",
    "ubnd",
    "sở ",
    "bộ ",
    "trường",
    "đại học",
    "học viện",
    "tập đoàn",
    "công ty",
    "cty",
    "ngân hàng",
    "viện",
    "trung tâm",
    "đài ",
    "báo ",
)

_GENERIC_PREFIX_ONLY = {"Bộ", "Sở", "Ủy ban", "UBND", "Công an", "Cảnh sát"}


def _postprocess_ner_entity(text: str, typ: str) -> tuple[str, str] | None:
    t = _SPACE_RE.sub(" ", (text or "").strip())
    t = t.lstrip("-").strip()
    if not t or t in {"-", "–", "—"}:
        return None

    t_lower = t.lower()
    if any(k in t_lower for k in _ORG_KEYWORDS):
        typ = "ORGANIZATION"

    # drop too generic single-token org-prefix entities (e.g. "Bộ")
    ws = _words(t)
    if typ == "ORGANIZATION" and len(ws) == 1 and _titleish(t) in _GENERIC_PREFIX_ONLY:
        return None

    t = _titleish(t)
    if _should_drop_entity(t, typ):
        return None

    return (t, typ)


def _merge_entities(tokens: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    current_words: list[str] = []
    current_type: str | None = None

    def flush():
        nonlocal current_words, current_type
        if current_words and current_type:
            merged = " ".join(current_words).strip()
            if merged:
                pp = _postprocess_ner_entity(merged, current_type)
                if pp is not None:
                    t2, ty2 = pp
                    out.append({"text": t2, "type": ty2})
        current_words = []
        current_type = None

    for tok in tokens:
        if isinstance(tok, dict):
            word = str(tok.get("word", "")).strip()
            tag = str(tok.get("ner", "")).strip()
        elif isinstance(tok, (tuple, list)) and len(tok) >= 4:
            word = str(tok[0]).strip()
            tag = str(tok[3]).strip()
        else:
            continue

        mapped = _map_underthesea_tag(tag)
        if mapped is None:
            flush()
            continue

        is_begin = tag.startswith("B-")
        is_inside = tag.startswith("I-")

        if is_begin or (current_type is not None and mapped != current_type):
            flush()
            current_type = mapped
            if word:
                current_words.append(word)
            continue

        if is_inside and current_type == mapped:
            if word:
                current_words.append(word)
            continue

        flush()
        current_type = mapped
        if word:
            current_words.append(word)

    flush()
    return out


@lru_cache(maxsize=128)
def _cached_ner(normalized_text: str) -> list[dict[str, str]]:
    if _ud_ner is None:
        raise NERUnavailableError("Underthesea is not available. Install with: pip install underthesea")
    tokens = _ud_ner(normalized_text)
    if not isinstance(tokens, list):
        return []
    return _merge_entities(tokens)


class UndertheseaNER:
    def extract(self, text: str) -> list[dict[str, str]]:
        normalized = _normalize_text(text)
        if not normalized:
            return []
        return _cached_ner(normalized)


# ----------------------------
# Role extraction (broad, bounded, includes fans/dư luận/KOL)
# ----------------------------
ROLE_PERSON_PATTERNS: list[re.Pattern[str]] = [
    # accidents/news roles
    re.compile(
        r"\b(nam|nữ)\s+([^\W\d_]{2,20}\s+)?(tài xế|lái xe|hành khách|nạn nhân|người dân|cư dân|nhân chứng)\b(?:\s+\d{1,3}\s*tuổi)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b(tài xế|lái xe|hành khách|nạn nhân|nhân chứng)\b(?:\s+\d{1,3}\s*tuổi)?", re.IGNORECASE),
    re.compile(r"\b(lãnh đạo|người dân|cư dân)\s+địa phương\b", re.IGNORECASE),

    # entertainment / social
    re.compile(r"\b(fan|fans)\s+[A-Z0-9][A-Z0-9\-]*\b", re.IGNORECASE),  # fans EXO, fan BLACKPINK
    re.compile(r"\b(fandom|fanclub)\s+[A-Z0-9][A-Z0-9\-]*\b", re.IGNORECASE),
    re.compile(r"\b(cư dân mạng|netizen|dư luận|khán giả)\b", re.IGNORECASE),
    re.compile(r"\b(KOL|influencer|creator|streamer)\b", re.IGNORECASE),

    # forces
    re.compile(
        r"\b(cảnh sát giao thông|cảnh sát|công an|phòng cháy chữa cháy|cứu nạn cứu hộ|lực lượng chức năng)\b",
        re.IGNORECASE,
    ),

    # quantified people mentions
    re.compile(r"\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+người(?:\s+khác)?\b", re.IGNORECASE),
]

_ORG_PREFIXES = r"(Bộ|Sở|UBND|Ủy ban|Công an|Tòa án|Viện kiểm sát|Bệnh viện|Trường|Đại học|Học viện|Công ty|Ngân hàng|Tập đoàn|Viện|Trung tâm)"
_ADMIN_HINT = r"(tỉnh|thành phố|tp\.?|huyện|quận|thị xã|xã|phường)"
ROLE_ORG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        rf"\b{_ORG_PREFIXES}\s+(?:{_ADMIN_HINT}\s+)?[{_VI_UPPER}][^,.;\n]*",
        re.IGNORECASE,
    )
]

_CLAUSE_CUTTERS = (
    "cho biết",
    "nhận định",
    "đánh giá",
    "có mặt",
    "nhanh chóng",
    "tổ chức",
    "phân luồng",
    "hỗ trợ",
    "cấp cứu",
    "đưa đi",
    "giải thích",
    "nói rằng",
    "khi",
    "do",
    "vì",
    "để",
)


def _cut_at_clause(text: str) -> str:
    t = text
    lower = t.lower()
    idxs = []
    for c in _CLAUSE_CUTTERS:
        i = lower.find(" " + c)
        if i > 0:
            idxs.append(i)
    if idxs:
        t = t[: min(idxs)].strip()
    return t


def _limit_words(text: str, max_words: int) -> str:
    ws = _words(text)
    if len(ws) <= max_words:
        return text
    return " ".join(ws[:max_words])


def _extract_roles(text: str) -> list[dict[str, str]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    found: list[dict[str, str]] = []

    for pat in ROLE_PERSON_PATTERNS:
        for m in pat.finditer(normalized):
            phrase = (m.group(0) or "").strip(" .,:;")
            if not phrase:
                continue
            phrase = _titleish(_SPACE_RE.sub(" ", phrase))
            if _should_drop_entity(phrase, "ROLE_PERSON"):
                continue
            found.append({"text": phrase, "type": "ROLE_PERSON"})

    for pat in ROLE_ORG_PATTERNS:
        for m in pat.finditer(normalized):
            phrase = (m.group(0) or "").strip(" .,:;")
            if not phrase:
                continue
            phrase = _SPACE_RE.sub(" ", phrase)
            phrase = _cut_at_clause(phrase)
            phrase = _limit_words(phrase, max_words=8)
            phrase = _titleish(phrase)

            if len(_words(phrase)) == 1 and phrase in _GENERIC_PREFIX_ONLY:
                continue
            if _should_drop_entity(phrase, "ROLE_ORGANIZATION"):
                continue

            found.append({"text": phrase, "type": "ROLE_ORGANIZATION"})

    return found


# ----------------------------
# Dedup + keep-longest pruning
# ----------------------------
def _dedup_keep_order(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for it in items:
        t = str(it.get("text", "")).strip()
        ty = str(it.get("type", "")).strip()
        if not t or not ty:
            continue
        key = (t, ty)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": t, "type": ty})
    return out


def _prune_substrings(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_type: dict[str, list[str]] = {}
    for it in items:
        by_type.setdefault(it["type"], []).append(it["text"])

    drop: set[tuple[str, str]] = set()
    for ty, texts in by_type.items():
        texts_sorted = sorted(texts, key=lambda s: len(s), reverse=True)
        kept_lower: list[str] = []
        for t in texts_sorted:
            tl = t.lower()
            if any(tl in k for k in kept_lower):
                drop.add((t, ty))
            else:
                kept_lower.append(tl)

    return [it for it in items if (it["text"], it["type"]) not in drop]


# ----------------------------
# Public API
# ----------------------------
def extract_entities(text: str) -> list[dict[str, str]]:
    """
    Combined extractor:
      - Proper NER: PERSON/LOCATION/ORGANIZATION/MISC
      - Role extraction: ROLE_PERSON/ROLE_ORGANIZATION (includes fans/dư luận/KOL)
    """
    ner_ents = UndertheseaNER().extract(text)
    roles = _extract_roles(text)

    merged = _dedup_keep_order(ner_ents + roles)
    merged = _prune_substrings(merged)
    return merged