import pytest

from translation import LocalMarianTranslator, TranslationUnavailableError

def test_translation_vi_en_or_skip():
    try:
        model = LocalMarianTranslator("vi", "en", local_files_only=True)
    except TranslationUnavailableError as exc:
        pytest.skip(str(exc))

    out = model.translate("Xin chao the gioi.")

    assert isinstance(out["translated_text"], str)
    assert out["translated_text"].strip() != ""
    assert out["source_lang"] == "vi"
    assert out["target_lang"] == "en"

def test_translation_en_vi_or_skip():
    try:
        model = LocalMarianTranslator("en", "vi", local_files_only=True)
    except TranslationUnavailableError as exc:
        pytest.skip(str(exc))

    out = model.translate("Hello world.")

    assert isinstance(out["translated_text"], str)
    assert out["translated_text"].strip() != ""
    assert out["source_lang"] == "en"
    assert out["target_lang"] == "vi"
