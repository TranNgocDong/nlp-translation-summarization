"""
Summarization Models Package

Provides:
- VIT5Summarizer: Single model for Vietnamese summarization
- HierarchicalSummarizer: Multi-level summarization for long texts
- wrap_vi_like_training: Vietnamese text preprocessing
"""

# ✅ FIX: Remove DualVIT5Summarizer (doesn't exist)
from .vit5_wrapper import VIT5Summarizer
from .prompting import wrap_vi_like_training
from .hierarchical import HierarchicalSummarizer

__all__ = [
    "VIT5Summarizer",
    "HierarchicalSummarizer",
    "wrap_vi_like_training",
]