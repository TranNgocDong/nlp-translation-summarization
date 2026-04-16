from __future__ import annotations


def wrap_vi_like_training(text_vi: str) -> str:
    t = (text_vi or "").strip()
    if not t:
        return t
    if ("Tiêu đề:" in t) or ("Nội dung:" in t) or ("Tóm tắt nhanh:" in t):
        return t
    return f"Nội dung:\n{t}"

