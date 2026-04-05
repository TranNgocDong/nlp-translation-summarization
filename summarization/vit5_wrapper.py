import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_PREFIX = "summarize: "

_LEADING_NOISE = re.compile(r"^[\s*\-•·–—]+")


def _clean(text: str) -> str:
    text = _LEADING_NOISE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


class VIT5Summarizer:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | None = None,
        prefix: str = DEFAULT_PREFIX,
        lang_label: str | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.lang_label = lang_label
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.checkpoint_dir))
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def summarize(
        self,
        text: str,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
        min_new_tokens: int = 48,
        num_beams: int = 4,
        length_penalty: float = 1.15,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
    ) -> dict[str, Any]:
        src = self.prefix + text
        enc = self.tokenizer(
            src,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        if min_new_tokens > 0:
            gen_kw["min_new_tokens"] = min_new_tokens
        out_ids = self.model.generate(**enc, **gen_kw)
        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {
            "summary": _clean(decoded),
            "language": self.lang_label,
        }


class DualVIT5Summarizer:
    def __init__(
        self,
        vi_checkpoint: str | Path,
        en_checkpoint: str | Path,
        device: str | None = None,
    ):
        self.vi = VIT5Summarizer(vi_checkpoint, device=device, lang_label="vi")
        self.en = VIT5Summarizer(en_checkpoint, device=device, lang_label="en")

    def summarize_pair(
        self,
        text_vi: str,
        text_en: str,
        num_beams: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
        min_new_tokens: int = 48,
        length_penalty: float = 1.15,
    ) -> dict[str, str]:
        r_vi = self.vi.summarize(
            text_vi,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        r_en = self.en.summarize(
            text_en,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        return {
            "summary_vi": r_vi["summary"],
            "summary_en": r_en["summary"],
        }
