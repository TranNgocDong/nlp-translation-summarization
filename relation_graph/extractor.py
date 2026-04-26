from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

UPPER = "A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
LOWER = "a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
ENTITY_WORD = rf"[{UPPER}][{UPPER}{LOWER}0-9]*(?:['’\-][{UPPER}{LOWER}0-9]+)?"
ENTITY_PATTERN = re.compile(
    rf"(?:(?:[Nn]hân vật|[Nn]han vat|[Ôô]ng|[Bb]à|[Aa]nh|[Cc]hị|[Cc]ô|[Cc]ậu|"
    rf"[Tt]ướng|[Vv]ua|[Hh]oàng tử|[Hh]oang tu|[Cc]ông chúa|[Cc]ong chua|[Kk]ing|[Qq]ueen|[Pp]rince|"
    rf"[Pp]rincess|[Ll]ord|[Ll]ady)\s+)?{ENTITY_WORD}(?:\s+{ENTITY_WORD}){{0,3}}"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:\n])\s+")
SPACE_RE = re.compile(r"\s+")
TRAILING_PUNCT_RE = re.compile(rf"^[^{UPPER}{LOWER}0-9]+|[^{UPPER}{LOWER}0-9]+$")
LETTERS_ONLY_RE = re.compile(rf"[^{UPPER}{LOWER}]")
SHARED_SUBJECT_RE = re.compile(r"^(?:,?\s*(và|va|and|rồi|roi|then|sau đó|sau do))+$", re.IGNORECASE)
GENERIC_STARTERS = {
    "Trong",
    "Sau",
    "Khi",
    "Tuy",
    "Nhung",
    "Nhưng",
    "Va",
    "Và",
    "Mot",
    "Một",
    "Cac",
    "Các",
    "Nhung",
    "Những",
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Meanwhile",
    "Later",
    "Then",
}


@dataclass(frozen=True)
class RelationRule:
    relation: str
    pattern: re.Pattern[str]


RELATION_RULES = [
    RelationRule("danh_bai", re.compile(r"\b(đánh bại|danh bai|hạ gục|ha guc|defeat(?:s|ed)?|beat)\b", re.IGNORECASE)),
    RelationRule("tan_cong", re.compile(r"\b(tấn công|tan cong|truy sát|truy sat|attack(?:s|ed)?)\b", re.IGNORECASE)),
    RelationRule("bao_ve", re.compile(r"\b(bảo vệ|bao ve|che chở|che cho|cứu|cuu|protect(?:s|ed)?|save(?:s|d)?)\b", re.IGNORECASE)),
    RelationRule("giup_do", re.compile(r"\b(giúp|giup|hỗ trợ|ho tro|assist(?:s|ed)?|help(?:s|ed)?)\b", re.IGNORECASE)),
    RelationRule("gap_go", re.compile(r"\b(gặp|gap|encounter(?:s|ed)?|meet(?:s|ing|met)?)\b", re.IGNORECASE)),
    RelationRule("lien_minh", re.compile(r"\b(hợp tác với|hop tac voi|liên minh với|lien minh voi|chiến đấu cùng|chien dau cung|ally(?:ied)? with)\b", re.IGNORECASE)),
    RelationRule("phan_boi", re.compile(r"\b(phản bội|phan boi|betray(?:s|ed)?)\b", re.IGNORECASE)),
    RelationRule("yeu_thuong", re.compile(r"\b(yêu|yeu|thương|thuong|love(?:s|d)?)\b", re.IGNORECASE)),
    RelationRule("thu_ghet", re.compile(r"\b(ghét|ghet|căm thù|cam thu|hate(?:s|d)?)\b", re.IGNORECASE)),
]


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def clean_entity_name(raw: str) -> str | None:
    value = TRAILING_PUNCT_RE.sub("", raw).strip()
    value = SPACE_RE.sub(" ", value)
    if not value:
        return None

    lowered = value.lower()
    if lowered.startswith("nhân vật "):
        value = "Nhân vật " + value.split(" ", 2)[2]
    elif lowered.startswith("nhan vat "):
        value = "Nhan vat " + value.split(" ", 2)[2]

    first_word = value.split(" ", 1)[0]
    if first_word in GENERIC_STARTERS:
        return None

    letters_only = LETTERS_ONLY_RE.sub("", value)
    if len(letters_only) < 2:
        return None
    if len(value) <= 2 and value.upper() != value:
        return None
    return value


def iter_entities(sentence: str) -> list[dict[str, int | str]]:
    entities: list[dict[str, int | str]] = []
    for match in ENTITY_PATTERN.finditer(sentence):
        entity = clean_entity_name(match.group(0))
        if not entity:
            continue
        entities.append(
            {
                "name": entity,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return entities


def nearest_left_entity(entities: Iterable[dict[str, int | str]], index: int) -> str | None:
    eligible = [entity for entity in entities if int(entity["end"]) <= index]
    if not eligible:
        return None
    return str(max(eligible, key=lambda item: int(item["end"]))["name"])


def nearest_left_entity_record(entities: Iterable[dict[str, int | str]], index: int) -> dict[str, int | str] | None:
    eligible = [entity for entity in entities if int(entity["end"]) <= index]
    if not eligible:
        return None
    return max(eligible, key=lambda item: int(item["end"]))


def nearest_right_entity(entities: Iterable[dict[str, int | str]], index: int) -> str | None:
    eligible = [entity for entity in entities if int(entity["start"]) >= index]
    if not eligible:
        return None
    return str(min(eligible, key=lambda item: int(item["start"]))["name"])


def choose_source_entity(sentence: str, entities: list[dict[str, int | str]], relation_start: int) -> str | None:
    left_entity = nearest_left_entity_record(entities, relation_start)
    if left_entity is None:
        return None

    first_entity = entities[0]
    connector = sentence[int(left_entity["end"]) : relation_start].strip()
    if (
        str(left_entity["name"]) != str(first_entity["name"])
        and connector
        and SHARED_SUBJECT_RE.fullmatch(connector)
    ):
        return str(first_entity["name"])
    return str(left_entity["name"])


class RelationGraphExtractor:
    def extract(self, text: str) -> dict[str, list[dict[str, str | int]] | int]:
        normalized = normalize_text(text)
        if not normalized:
            return {"nodes": [], "edges": [], "sentence_count": 0}

        sentences = split_sentences(normalized)
        node_counter: Counter[str] = Counter()
        edge_map: dict[tuple[str, str, str], dict[str, str | int]] = {}

        for sentence in sentences:
            entities = iter_entities(sentence)
            if not entities:
                continue

            unique_entities: list[str] = []
            for entity in entities:
                name = str(entity["name"])
                node_counter[name] += 1
                if name not in unique_entities:
                    unique_entities.append(name)

            found_relation = False
            for rule in RELATION_RULES:
                for match in rule.pattern.finditer(sentence):
                    source = choose_source_entity(sentence, entities, match.start())
                    target = nearest_right_entity(entities, match.end())
                    if not source or not target or source == target:
                        continue
                    key = (source, target, rule.relation)
                    if key not in edge_map:
                        edge_map[key] = {
                            "source": source,
                            "target": target,
                            "relation": rule.relation,
                            "evidence": sentence,
                            "count": 0,
                        }
                    edge_map[key]["count"] = int(edge_map[key]["count"]) + 1
                    found_relation = True

            if not found_relation and len(unique_entities) >= 2:
                source, target = unique_entities[0], unique_entities[1]
                key = (source, target, "cung_xuat_hien")
                if key not in edge_map:
                    edge_map[key] = {
                        "source": source,
                        "target": target,
                        "relation": "cung_xuat_hien",
                        "evidence": sentence,
                        "count": 0,
                    }
                edge_map[key]["count"] = int(edge_map[key]["count"]) + 1

        nodes = [
            {
                "id": name,
                "label": name,
                "mentions": mentions,
                "type": "character",
            }
            for name, mentions in node_counter.most_common()
        ]
        edges = sorted(
            edge_map.values(),
            key=lambda item: (-int(item["count"]), str(item["source"]), str(item["target"]), str(item["relation"])),
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "sentence_count": len(sentences),
        }


def build_relation_graph(text: str) -> dict[str, list[dict[str, str | int]] | int]:
    return RelationGraphExtractor().extract(text)
