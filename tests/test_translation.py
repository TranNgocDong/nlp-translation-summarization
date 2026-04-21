import pytest

from models.translation.inference import TranslationModel

def test_translation_vi_en_or_skip():
    model = TranslationModel("vi", "en", local_files_only=True)

    # Nếu model không có local thì translate sẽ ok=False -> skip để tránh fail môi trường
    out = model.translate("Xin chào thế giới.")
    if not out.get("ok", False):
        pytest.skip(out.get("error", "translation model unavailable locally"))

    assert isinstance(out["translated_text"], str)
    assert out["translated_text"].strip() != ""
    assert out["source_lang"] == "vi"
    assert out["target_lang"] == "en"

def test_translation_en_vi_or_skip():
    model = TranslationModel("en", "vi", local_files_only=True)
    out = model.translate("Hello world.")
    if not out.get("ok", False):
        pytest.skip(out.get("error", "translation model unavailable locally"))

    assert isinstance(out["translated_text"], str)
    assert out["translated_text"].strip() != ""
    assert out["source_lang"] == "en"
    assert out["target_lang"] == "vi"