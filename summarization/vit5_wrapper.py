import sys
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    import config
except ImportError:
    config = None


import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_PREFIX = "summarize: "

_LEADING_NOISE = re.compile(r"^[\s*\-•·–—+]+")
_MID_NOISE = re.compile(r"[<>\[\]{}|~`^]+")
_REPEAT_PUNCT = re.compile(r"([,.;:!?])\1+")
_REPEAT_UNIGRAM = re.compile(r"\b([^\W\d_]+)\b(?:\s+\1\b)+", flags=re.IGNORECASE | re.UNICODE)
_REPEAT_BIGRAM = re.compile(
    r"\b([^\W\d_]+\s+[^\W\d_]+)\b(?:\s+\1\b)+",
    flags=re.IGNORECASE | re.UNICODE,
)


def _normalize_input(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _MID_NOISE.sub(" ", text)
    # Không dùng \s+ vì nó sẽ xóa mất dấu xuống dòng \n
    text = re.sub(r"[ \t]+", " ", text) 
    return text.strip()


def _clean(text: str) -> str:
    text = text.replace(" + ", ". ")
    text = _LEADING_NOISE.sub("", text).strip()
    text = _MID_NOISE.sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    # Loại bỏ hoàn toàn dấu chấm phẩy theo yêu cầu (thay bằng dấu chấm phẩy thành dấu chấm hoặc xóa)
    text = text.replace(";", ".")
    # Khử lặp từ/cụm liền kề
    text = _REPEAT_BIGRAM.sub(r"\1", text)
    text = _REPEAT_UNIGRAM.sub(r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



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
        # Dat legacy=True de tranh warning tokenizer T5 va giu dung hanh vi da train truoc day.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir),
            use_fast=False,
            legacy=True,
        )
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
        max_input_length: int = config.MAX_INPUT_LENGTH_INFERENCE if config else 1024,
        max_new_tokens: int = config.MAX_NEW_TOKENS if config else 160,
        min_new_tokens: int = config.MIN_NEW_TOKENS if config else 24,
        num_beams: int = config.NUM_BEAMS if config else 5,
        length_penalty: float = config.LENGTH_PENALTY if config else 1.0,
        early_stopping: bool = config.EARLY_STOPPING if config else True,
        no_repeat_ngram_size: int = config.NO_REPEAT_NGRAM_SIZE if config else 2,
        **kwargs
    ) -> dict[str, Any]:

        res = self.summarize_batch(
            [text],
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return res[0]

    @torch.inference_mode()
    def summarize_batch(
        self,
        texts: list[str],
        max_input_length: int = config.MAX_INPUT_LENGTH_INFERENCE if config else 1024,
        max_new_tokens: int = config.MAX_NEW_TOKENS if config else 160,
        min_new_tokens: int = config.MIN_NEW_TOKENS if config else 24,
        num_beams: int = config.NUM_BEAMS if config else 5,
        length_penalty: float = config.LENGTH_PENALTY if config else 1.0,
        early_stopping: bool = config.EARLY_STOPPING if config else True,
        no_repeat_ngram_size: int = config.NO_REPEAT_NGRAM_SIZE if config else 2,
    ) -> list[dict[str, Any]]:

        if not texts:
            return []
            
        normalized_texts = [_normalize_input(t) for t in texts]
        
        srcs = []
        for t in normalized_texts:
            # logic đồng bộ với JsonlSummarizeDataset trong train_vit5_summarize.py
            if "Từ khóa bắt buộc:" in t or "Nội dung:" in t:
                srcs.append(t) # Không thêm gánh nặng prefix
            else:
                srcs.append(self.prefix + t)
        
        enc = self.tokenizer(
            srcs,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": config.REPETITION_PENALTY if config else 1.25,
            "renormalize_logits": True,
            "do_sample": False,
        }

        
        out_ids = self.model.generate(**enc, **gen_kw)
        decoded_list = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        
        results = []
        for decoded in decoded_list:
            results.append({
                "summary": _clean(decoded),
                "language": self.lang_label,
            })
        return results


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
