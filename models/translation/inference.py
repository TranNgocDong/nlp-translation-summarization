from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from translation.marian_wrapper import LocalMarianTranslator, TranslationUnavailableError


Lang = Literal["vi", "en"]


@dataclass
class TranslationResult:
    translated_text: str
    source_lang: Lang
    target_lang: Lang
    model_name: str
    ok: bool
    error: str = ""


class TranslationModel:
    """
    Thin adapter to reuse existing translation/LocalMarianTranslator,
    placed under models/translation for assignment/rubric compatibility.
    """

    def __init__(self, source_lang: Lang, target_lang: Lang, *, device: str | None = None, local_files_only: bool = True):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.local_files_only = local_files_only

        self._translator: LocalMarianTranslator | None = None
        self._init_error: str = ""

        try:
            self._translator = LocalMarianTranslator(
                source_lang=source_lang,
                target_lang=target_lang,
                device=device,
                local_files_only=local_files_only,
            )
        except TranslationUnavailableError as exc:
            self._translator = None
            self._init_error = str(exc)

    def translate(
        self,
        text: str,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> dict[str, Any]:
        # consistent output format
        if self._translator is None:
            return TranslationResult(
                translated_text=text or "",
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                model_name="",
                ok=False,
                error=self._init_error or "Translation model unavailable",
            ).__dict__

        payload = self._translator.translate(
            text=text,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return TranslationResult(
            translated_text=str(payload.get("translated_text", "")),
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            model_name=str(payload.get("model_name", "")),
            ok=True,
            error="",
        ).__dict__