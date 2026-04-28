from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAMES = {
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
}


class TranslationUnavailableError(RuntimeError):
    pass


class LocalMarianTranslator:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        device: str | None = None,
        local_files_only: bool = True,
    ):
        key = (source_lang, target_lang)
        if key not in MODEL_NAMES:
            raise TranslationUnavailableError(f"Unsupported translation direction: {source_lang}->{target_lang}")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_name = MODEL_NAMES[key]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise TranslationUnavailableError(
                f"Translation model {self.model_name} is not available locally."
            ) from exc

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> dict[str, Any]:
        src = text.strip()
        if not src:
            return {
                "translated_text": "",
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "model_name": self.model_name,
            }

        import re
        # Phân rã văn bản thành các câu nhỏ dựa trên dấu chấm, chấm hỏi, chấm than
        sentences = re.split(r'(?<=[.!?])\s+', src)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Nhóm câu thành các chunk an toàn (tối đa ~150 từ, khoảng 200 tokens)
        # để vừa giữ được ngữ cảnh, vừa không bao giờ bị cắt cụt bởi giới hạn 512 tokens.
        chunks = []
        current_chunk = ""
        for s in sentences:
            if len((current_chunk + " " + s).split()) < 150:
                current_chunk += " " + s
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = s
        if current_chunk:
            chunks.append(current_chunk.strip())

        translated_paragraphs = []
        for chunk in chunks:
            enc = self.tokenizer(
                chunk,
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            translated_para = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            translated_paragraphs.append(translated_para)

        full_translation = " ".join(translated_paragraphs)
        return {
            "translated_text": full_translation,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "model_name": self.model_name,
        }

