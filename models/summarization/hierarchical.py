import re
from typing import List

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import config
HIERARCHICAL = config.HIERARCHICAL



class HierarchicalSummarizer:
    """
    Logic tóm tắt phân cấp cho văn bản dài.
    Chia bài viết thành các đoạn nhỏ, tóm tắt từng đoạn rồi tổng hợp lại.

    Bản này dùng *token-aware chunking* để giảm nguy cơ vượt max token.
    """

    def __init__(self, base_summarizer):
        self.base = base_summarizer
        self.chunk_tokens = int(HIERARCHICAL.get("chunk_tokens", 400))
        self.overlap_sents = int(HIERARCHICAL.get("overlap_sents", 2))
        self.max_levels = int(HIERARCHICAL.get("max_levels", 2))

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _token_len(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self.base.tokenizer.encode(text, add_special_tokens=False))

    def _resolve_chunk_budget(
        self,
        max_input_length: int,
        chunk_size_words: int | None,
    ) -> int:
        # Giữ vùng đệm cho prefix/special tokens để tránh cắt cụt quá mạnh
        hard_cap = max(64, max_input_length - 32)
        safe_budget = max(64, min(self.chunk_tokens, hard_cap))

        # Nếu client truyền chunk_size_words thì ưu tiên cấu hình theo request
        if chunk_size_words is not None and chunk_size_words > 0:
            # Heuristic cho tiếng Việt/Anh: ~1.5-1.8 token/word
            approx_tokens = max(64, int(chunk_size_words * 1.6))
            safe_budget = max(64, min(approx_tokens, hard_cap))
        return safe_budget

    def _resolve_overlap_sents(
        self,
        sentences: List[str],
        chunk_overlap_words: int | None,
    ) -> int:
        overlap_sents = max(0, self.overlap_sents)
        if chunk_overlap_words is None or chunk_overlap_words <= 0 or not sentences:
            return overlap_sents

        avg_words_per_sent = max(
            1,
            int(sum(len(s.split()) for s in sentences) / len(sentences)),
        )
        derived_overlap = int(round(chunk_overlap_words / avg_words_per_sent))
        return max(0, min(8, derived_overlap))

    def _create_chunks(
        self,
        text: str,
        max_input_length: int,
        chunk_size_words: int | None = None,
        chunk_overlap_words: int | None = None,
    ) -> tuple[List[str], int, int]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [], 0, 0

        safe_budget = self._resolve_chunk_budget(
            max_input_length=max_input_length,
            chunk_size_words=chunk_size_words,
        )
        overlap_sents = self._resolve_overlap_sents(
            sentences=sentences,
            chunk_overlap_words=chunk_overlap_words,
        )

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._token_len(sent)

            # Câu quá dài: đóng chunk hiện tại và xử lý câu riêng
            if sent_tokens > safe_budget:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                words = sent.split()
                tmp: list[str] = []
                for w in words:
                    candidate = " ".join(tmp + [w])
                    if self._token_len(candidate) > safe_budget and tmp:
                        chunks.append(" ".join(tmp))
                        tmp = [w]
                    else:
                        tmp.append(w)
                if tmp:
                    chunks.append(" ".join(tmp))
                continue

            if current_tokens + sent_tokens > safe_budget and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap = current_chunk[-overlap_sents:] if overlap_sents > 0 and len(current_chunk) >= overlap_sents else []
                current_chunk = overlap + [sent]
                current_tokens = self._token_len(" ".join(current_chunk))
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks, safe_budget, overlap_sents

    def summarize(self, text: str, **kwargs):
        """Hàm bọc cho API gọi, tự động bóc tách text tóm tắt."""
        res = self.summarize_long_text(text, **kwargs)
        if isinstance(res, dict):
            return res.get("summary", "")
        return str(res)

    def summarize_long_text(self, text: str, **kwargs) -> dict:

        generation_kwargs = dict(kwargs)
        
        # Lấy các tham số phân cấp riêng nếu có
        chunk_size_words = generation_kwargs.pop("chunk_size_words", None)
        chunk_overlap_words = generation_kwargs.pop("chunk_overlap_words", None)
        carry_prev_summary = bool(generation_kwargs.pop("carry_prev_summary", False))
        
        model_limit = generation_kwargs.pop("max_input_length", 1024)
        if model_limit is None or model_limit <= 0:
            model_limit = 1024

        # Model tokenizer đôi khi báo số cực lớn không thực tế
        if model_limit > 4096:
            model_limit = 1024

        if self._token_len(text) <= model_limit:
            res = self.base.summarize(text, max_input_length=model_limit, **generation_kwargs)
            return {"summary": res["summary"], "metadata": {"hierarchical": False}}

        current_text = text
        latest_chunk_count = 0
        effective_budget = 0
        effective_overlap = 0
        for level in range(1, self.max_levels + 1):
            chunks, effective_budget, effective_overlap = self._create_chunks(
                current_text,
                max_input_length=model_limit,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
            )
            if not chunks:
                break

            latest_chunk_count = len(chunks)
            print(f"[Hierarchical] Level {level}: {len(chunks)} chunks")

            intermediate_summaries: list[str] = []
            prev_summary = ""
            for chunk in chunks:
                if carry_prev_summary and prev_summary:
                    chunk_to_process = f"Tóm tắt phần trước: {prev_summary}\nTiếp tục nội dung:\n{chunk}"
                else:
                    chunk_to_process = chunk
                res = self.base.summarize(chunk_to_process, max_input_length=model_limit, **generation_kwargs)
                prev_summary = res["summary"]
                intermediate_summaries.append(prev_summary)

            combined_text = " ".join(intermediate_summaries).strip()
            if not combined_text:
                break

            if self._token_len(combined_text) <= model_limit or level == self.max_levels:
                final_res = self.base.summarize(combined_text, max_input_length=model_limit, **generation_kwargs)
                return {
                    "summary": final_res["summary"],
                    "metadata": {
                        "hierarchical": True,
                        "level": level,
                        "chunk_count": latest_chunk_count,
                        "chunk_token_budget": effective_budget,
                        "overlap_sents": effective_overlap,
                        "carry_prev_summary": carry_prev_summary,
                    },
                }

            current_text = combined_text

        final_res = self.base.summarize(current_text, max_input_length=model_limit, **generation_kwargs)
        return {
            "summary": final_res["summary"],
            "metadata": {
                "hierarchical": True,
                "level": "max",
                "chunk_count": latest_chunk_count,
                "chunk_token_budget": effective_budget,
                "overlap_sents": effective_overlap,
                "carry_prev_summary": carry_prev_summary,
            },
        }
