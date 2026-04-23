import pytest

pytest.importorskip("underthesea")

from models.ner.inference import extract_entities


def test_empty_text_returns_empty_list():
    assert extract_entities("") == []
    assert extract_entities("   \n\t  ") == []


def test_output_format_is_list_of_dicts():
    text = "Ông Nguyễn Văn A sống ở Hà Nội và làm việc tại VNPT."
    ents = extract_entities(text)

    assert isinstance(ents, list)
    for e in ents:
        assert isinstance(e, dict)
        assert "text" in e and isinstance(e["text"], str)
        assert "type" in e and isinstance(e["type"], str)
        assert e["text"].strip() != ""
        assert e["type"].strip() != ""


def test_contains_some_entities():
    text = "Ông Nguyễn Văn A sống ở Hà Nội."
    ents = extract_entities(text)
    assert len(ents) >= 1