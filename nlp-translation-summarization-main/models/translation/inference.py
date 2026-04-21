import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ FIX: Absolute import thay vì relative import
try:
    from models.translation.config import MODEL_VI_TO_EN, MODEL_EN_TO_VI, MAX_LENGTH, DEVICE, NUM_BEAMS
except ImportError:
    # Fallback for direct execution
    from config import MODEL_VI_TO_EN, MODEL_EN_TO_VI, MAX_LENGTH, DEVICE, NUM_BEAMS


class TranslationModel:
    """
    Translation model for Vietnamese ↔ English translation
    
    Uses Helsinki-NLP/Opus-MT pre-trained models for fast, accurate translation.
    
    Example:
        >>> model = TranslationModel()
        >>> en_text = model.translate_vi_to_en("Xin chào")
        >>> print(en_text)  # "Hello"
    """
    
    def __init__(self):
        """Initialize translation models (VI→EN and EN→VI)"""
        print("Loading translation models...")
        
        try:
            # VI → EN Model
            print("  Loading VI→EN model...")
            self.tokenizer_vi_en = AutoTokenizer.from_pretrained(MODEL_VI_TO_EN)
            self.model_vi_en = AutoModelForSeq2SeqLM.from_pretrained(MODEL_VI_TO_EN)
            
            # Try to move to device, fallback to CPU if error
            try:
                self.model_vi_en.to(DEVICE)
                self.device = DEVICE
            except Exception as e:
                print(f"  ⚠️ Cannot use {DEVICE}, falling back to CPU: {e}")
                self.model_vi_en.to("cpu")
                self.device = "cpu"
            
            self.model_vi_en.eval()
            print(f"  ✅ VI→EN model loaded (device: {self.device})")
            
            # EN → VI Model
            print("  Loading EN→VI model...")
            self.tokenizer_en_vi = AutoTokenizer.from_pretrained(MODEL_EN_TO_VI)
            self.model_en_vi = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_VI)
            
            # Move to same device as first model
            self.model_en_vi.to(self.device)
            self.model_en_vi.eval()
            print(f"  ✅ EN→VI model loaded (device: {self.device})")
            
            print("✅ Translation models loaded successfully")
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    # ✅ Text cleaning method
    def _clean_text(self, text: str) -> str:
        """
        Clean corrupted translation output
        
        Removes:
        - Weird characters (+, _, Z, j, &, W, Ỵ, Ẽ, Ơ, Ẫ, Ẳ, etc.)
        - Extra spaces
        - Broken punctuation
        """
        
        if not text:
            return ""
        
        # Remove weird/corrupted characters
        weird_chars = [
            '+', '_', 'Z', 'j', '&', 'W', 'Ỵ', 'Ẽ', 'Ơ', 'Ẫ', 'Ẳ',
            '|', '`', '~', '^', '¬', '§', '¶', '†', '‡', '¿', '¡',
        ]
        
        for char in weird_chars:
            text = text.replace(char, ' ')
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)      # Multiple spaces → single space
        text = re.sub(r'\.+', '.', text)      # Multiple dots → single dot
        text = re.sub(r',+', ',', text)       # Multiple commas → single comma
        text = re.sub(r'!+', '!', text)       # Multiple ! → single !
        text = re.sub(r'\?+', '?', text)      # Multiple ? → single ?
        
        # Fix sentence structure
        text = re.sub(r'\.([A-Za-zÀ-ỿ])', r'. \1', text)  # Add space after period
        text = re.sub(r',([A-Za-zÀ-ỿ])', r', \1', text)   # Add space after comma
        text = re.sub(r'!([A-Za-zÀ-ỿ])', r'! \1', text)   # Add space after !
        text = re.sub(r'\?([A-Za-zÀ-ỿ])', r'? \1', text)  # Add space after ?
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def translate_vi_to_en(self, text: str, max_length: int = 256) -> str:
        """
        Translate Vietnamese to English
        
        Args:
            text (str): Vietnamese text to translate
            max_length (int): Maximum length of output (default: 256)
        
        Returns:
            str: English translation (empty string on error)
        
        Example:
            >>> model = TranslationModel()
            >>> result = model.translate_vi_to_en("Nhân vật A là chiến binh mạnh mẽ")
            >>> print(result)
            "Character A is a strong warrior"
        """
        # Input validation
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer_vi_en(
                text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation with no gradient
            with torch.no_grad():
                translated_ids = self.model_vi_en.generate(
                    inputs["input_ids"],
                    num_beams=NUM_BEAMS,
                    max_length=max_length,
                    early_stopping=True,
                    do_sample=False
                )
            
            # ✅ Use clean_up_tokenization_spaces
            translation = self.tokenizer_vi_en.batch_decode(
                translated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            result = translation[0].strip() if translation else ""
            
            # ✅ Clean the output
            result = self._clean_text(result)
            
            return result
        
        except Exception as e:
            print(f"⚠️ Error in translate_vi_to_en: {e}")
            return ""
    
    def translate_en_to_vi(self, text: str, max_length: int = 256) -> str:
        """
        Translate English to Vietnamese
        
        Args:
            text (str): English text to translate
            max_length (int): Maximum length of output (default: 256)
        
        Returns:
            str: Vietnamese translation (empty string on error)
        
        Example:
            >>> model = TranslationModel()
            >>> result = model.translate_en_to_vi("Character A is a strong warrior")
            >>> print(result)
            "Nhân vật A là chiến binh mạnh mẽ"
        """
        # Input validation
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer_en_vi(
                text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation with no gradient
            with torch.no_grad():
                translated_ids = self.model_en_vi.generate(
                    inputs["input_ids"],
                    num_beams=NUM_BEAMS,
                    max_length=max_length,
                    early_stopping=True,
                    do_sample=False
                )
            
            # ✅ Use clean_up_tokenization_spaces
            translation = self.tokenizer_en_vi.batch_decode(
                translated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            result = translation[0].strip() if translation else ""
            
            # ✅ Clean the output
            result = self._clean_text(result)
            
            return result
        
        except Exception as e:
            print(f"⚠️ Error in translate_en_to_vi: {e}")
            return ""
    
    def translate(
        self,
        text: str,
        source_lang: str = "vi",
        target_lang: str = "en",
        max_length: int = 256
    ) -> str:
        """
        Translate between languages (VI ↔ EN)
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language ("vi" or "en")
            target_lang (str): Target language ("vi" or "en")
            max_length (int): Maximum length of output (default: 256)
        
        Returns:
            str: Translated text (empty string on error)
        
        Raises:
            None (returns empty string for unsupported language pairs)
        
        Example:
            >>> model = TranslationModel()
            >>> result = model.translate("Xin chào", source_lang="vi", target_lang="en")
            >>> print(result)  # "Hello"
        """
        # Validate input
        if not text or not isinstance(text, str):
            return ""
        
        # Validate languages
        valid_langs = {"vi", "en"}
        if source_lang not in valid_langs or target_lang not in valid_langs:
            print(f"⚠️ Unsupported language: {source_lang} → {target_lang}")
            return ""
        
        if source_lang == target_lang:
            return text  # No translation needed
        
        # Translate
        if source_lang == "vi" and target_lang == "en":
            return self.translate_vi_to_en(text, max_length)
        elif source_lang == "en" and target_lang == "vi":
            return self.translate_en_to_vi(text, max_length)
        else:
            print(f"⚠️ Unsupported language pair: {source_lang} → {target_lang}")
            return ""
    
    def get_device(self) -> str:
        """Get current device being used"""
        return self.device


# Test code
if __name__ == "__main__":
    try:
        print("=== Translation Model Test ===\n")
        
        # Initialize model
        model = TranslationModel()
        
        # Test 1: VI → EN
        print("\nTest 1: Vietnamese → English")
        vi_text = "Nhân vật A là chiến binh mạnh mẽ"
        en_text = model.translate_vi_to_en(vi_text)
        print(f"  VI: {vi_text}")
        print(f"  EN: {en_text}")
        
        # Test 2: EN → VI
        print("\nTest 2: English → Vietnamese")
        en_text = "Character A is a strong warrior"
        vi_text = model.translate_en_to_vi(en_text)
        print(f"  EN: {en_text}")
        print(f"  VI: {vi_text}")
        
        # Test 3: Using translate() method
        print("\nTest 3: Using translate() method")
        result = model.translate("Xin chào", source_lang="vi", target_lang="en")
        print(f"  VI: Xin chào")
        print(f"  EN: {result}")
        
        # Test 4: Device info
        print(f"\nDevice: {model.get_device()}")
        
        print("\n✅ All tests completed!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()