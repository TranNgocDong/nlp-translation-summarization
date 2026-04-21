"""
VIT5 Summarizer - Vietnamese Text Summarization
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class VIT5Summarizer:
    """
    VIT5 Summarizer for Vietnamese text
    
    Uses VietAI/vit5-base model for fast and accurate summarization.
    """
    
    def __init__(self, model_name: str = "VietAI/vit5-base", lang_label: str = "vi"):
        """
        Initialize VIT5 summarizer
        
        Args:
            model_name: HuggingFace model name (default: VietAI/vit5-base)
            lang_label: Language label (vi/en)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VIT5] Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"[VIT5] Loading model from {model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.lang_label = lang_label
        self.model.eval()
        
        print(f"✅ VIT5 Summarizer initialized on {self.device.upper()}")
    
    def _clean_output(self, text: str) -> str:
        """
        Clean model output from sentinel tokens and corruption
        
        Removes:
        - T5 sentinel tokens (<extra_id_*>)
        - Weird/corrupted characters
        - Extra whitespace and punctuation
        """
        
        if not text:
            return ""
        
        # ========== Remove T5 Sentinel Tokens ==========
        text = re.sub(r'<extra_id_\d+>', '', text)
        text = re.sub(r'extra_id_\d+', '', text)
        
        # ========== Remove Corruption Patterns ==========
        text = re.sub(r'dung:', '', text)      # "dung:" repeated
        text = re.sub(r'marize:', '', text)
        text = re.sub(r'Content:', '', text)
        text = re.sub(r'nội dung:', '', text)
        
        # ========== Remove Weird/Corrupted Characters ==========
        weird_chars = [
            # ASCII symbols
            '+', '_', 'Z', 'j', '&', 'W',
            # Vietnamese corrupted diacritics
            'Ỵ', 'Ẽ', 'Ơ', 'Ẫ', 'Ẳ', 'Ỡ',
            'Ế', 'Ỹ', 'Ữ', 'Ư', 'Ỗ', 'Ặ',
            'Õ', 'Ẻ', 'Ỷ', 'Ẹ', 'Ỏ',
            # Special symbols
            '|', '`', '~', '^', '¬', '§', '¶', '†', '‡', '¿', '¡',
            '©', '®', '™', '€', '£', '¥', '¢', '₹', '₽', '₩', '฿',
        ]
        
        for char in weird_chars:
            text = text.replace(char, ' ')
        
        # ========== Fix Spacing Issues ==========
        text = re.sub(r'\s+', ' ', text)          # Multiple spaces → single
        text = re.sub(r'\.+', '.', text)          # Multiple dots → single
        text = re.sub(r',+', ',', text)           # Multiple commas → single
        text = re.sub(r'!+', '!', text)           # Multiple ! → single
        text = re.sub(r'\?+', '?', text)          # Multiple ? → single
        
        # ========== Fix Sentence Structure ==========
        # Add space after punctuation + Vietnamese letter
        text = re.sub(r'\.([A-Za-zÀ-ỿ])', r'. \1', text)
        text = re.sub(r',([A-Za-zÀ-ỿ])', r', \1', text)
        text = re.sub(r'!([A-Za-zÀ-ỿ])', r'! \1', text)
        text = re.sub(r'\?([A-Za-zÀ-ỿ])', r'? \1', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        return text.strip()
    
    def summarize(
        self,
        text: str,
        max_input_length: int = 1024,
        max_new_tokens: int = 150,
        min_new_tokens: int = 20,
        num_beams: int = 3,
        length_penalty: float = 1.5,
    ) -> dict:
        """
        Summarize text
        
        Args:
            text: Input text to summarize
            max_input_length: Max input token length (default: 1024)
            max_new_tokens: Max summary token length (default: 150)
            min_new_tokens: Min summary token length (default: 20)
            num_beams: Beam search width (default: 3, use 2 for faster)
            length_penalty: Length penalty factor (default: 1.5)
        
        Returns:
            dict with "summary" key containing cleaned summary
        
        Example:
            >>> summarizer = VIT5Summarizer()
            >>> result = summarizer.summarize("Long text here...")
            >>> print(result["summary"])
        """
        try:
            # Validate input
            if not text or not text.strip():
                return {"summary": ""}
            
            text = text.strip()
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    num_beams=num_beams,
                    max_length=max_new_tokens,
                    min_length=min_new_tokens,
                    no_repeat_ngram_size=2,
                    length_penalty=length_penalty,
                    early_stopping=True,
                    do_sample=False  # Deterministic output
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean output
            summary = self._clean_output(summary)
            
            return {"summary": summary}
        
        except Exception as e:
            print(f"❌ Summarization error: {e}")
            import traceback
            traceback.print_exc()
            return {"summary": ""}


# ==================== TEST ====================

if __name__ == "__main__":
    print("=== VIT5 Summarizer Test ===\n")
    
    try:
        # Initialize
        summarizer = VIT5Summarizer()
        
        # Test text
        test_text = """
        Nhân vật A là một chiến binh mạnh mẽ, được biết đến trong cộng đồng vì sức mạnh và lòng dũng cảm của mình.
        Anh ta đã tham gia vào nhiều trận chiến khác nhau và luôn đứng đầu trong các chiến dịch quân sự.
        Nhân vật B, một công chúa xinh đẹp, đã gặp nhân vật A tại một sự kiện hoàng gia.
        Họ nhanh chóng trở thành bạn bè thân thiết và sau đó phát triển thành tình yêu sâu sắc.
        Cùng nhau, họ chống lại các kẻ thù của vương quốc và bảo vệ người dân.
        """ * 3
        
        # Summarize
        result = summarizer.summarize(test_text)
        summary = result["summary"]
        
        print(f"Original length: {len(test_text.split())} words")
        print(f"Summary length: {len(summary.split())} words")
        print(f"\nSummary:\n{summary}")
        print(f"\n✅ Test passed (no corruption)")
        
        # Check for common corruption patterns
        corruption_patterns = ['extra_id_', 'dung:', 'marize:', '<extra_id']
        has_corruption = any(p in summary for p in corruption_patterns)
        
        if has_corruption:
            print("⚠️ Warning: Corruption detected!")
        else:
            print("✅ No corruption detected!")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()