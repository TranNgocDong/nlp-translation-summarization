import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.translation.inference import TranslationModel


class TestTranslationModel:
    """Comprehensive test suite for TranslationModel class"""
    
    @pytest.fixture
    def model(self):
        """Initialize translation model for each test"""
        try:
            return TranslationModel()
        except Exception as e:
            pytest.skip(f"Model initialization failed: {e}")
    
    @pytest.fixture
    def sample_texts(self):
        """Provide sample texts for testing"""
        return {
            "short_vi": "Xin chào",
            "short_en": "Hello",
            "medium_vi": "Nhân vật A là chiến binh mạnh mẽ.",
            "medium_en": "Character A is a strong warrior.",
            "long_vi": "Nhân vật A là một chiến binh mạnh mẽ sống trong một làng nhỏ ở núi cao. "
                       "Từ nhỏ, anh ta được dạy dỗ bởi sư phụ về nghệ thuật kiếm pháp. "
                       "Anh ta có một tấm lòng vàng và luôn sẵn sàng giúp đỡ những người yếu thế.",
            "long_en": "Character A is a strong warrior living in a small village in the high mountains. "
                      "From childhood, he was taught by his master about the art of swordsmanship. "
                      "He has a kind heart and is always willing to help the weak.",
            "special_chars_vi": "Xin chào! Đây là kiểm tra. Bạn khỏe không?",
            "numbers_vi": "Năm 2024, tôi học lập trình.",
            "mixed_vi": "Hello world! Xin chào thế giới.",
        }
    
    # ==================== TEST translate_vi_to_en() ====================
    
    def test_translate_vi_to_en_short(self, model, sample_texts):
        """Test VI→EN translation on short text"""
        vi_text = sample_texts["short_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        assert result != vi_text, "Result should be different from input"
        print(f"\nVI: {vi_text}")
        print(f"EN: {result}")
    
    def test_translate_vi_to_en_medium(self, model, sample_texts):
        """Test VI→EN translation on medium text"""
        vi_text = sample_texts["medium_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        assert "warrior" in result.lower() or "strong" in result.lower(), \
            "Translation should contain key words"
        print(f"\nVI: {vi_text}")
        print(f"EN: {result}")
    
    def test_translate_vi_to_en_long(self, model, sample_texts):
        """Test VI→EN translation on long text"""
        vi_text = sample_texts["long_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        assert len(result.split()) > 5, "Translation should have multiple words"
        print(f"\nVI (long): {vi_text[:100]}...")
        print(f"EN: {result[:100]}...")
    
    def test_translate_vi_to_en_with_special_chars(self, model, sample_texts):
        """Test VI→EN translation with special characters"""
        vi_text = sample_texts["special_chars_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert isinstance(result, str), "Should handle special characters"
        assert len(result) > 0, "Translation should not be empty"
        print(f"\nVI: {vi_text}")
        print(f"EN: {result}")
    
    def test_translate_vi_to_en_with_numbers(self, model, sample_texts):
        """Test VI→EN translation with numbers"""
        vi_text = sample_texts["numbers_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert isinstance(result, str), "Should handle numbers"
        assert len(result) > 0, "Translation should not be empty"
        print(f"\nVI: {vi_text}")
        print(f"EN: {result}")
    
    # ==================== TEST translate_en_to_vi() ====================
    
    def test_translate_en_to_vi_short(self, model, sample_texts):
        """Test EN→VI translation on short text"""
        en_text = sample_texts["short_en"]
        result = model.translate_en_to_vi(en_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        assert result != en_text, "Result should be different from input"
        print(f"\nEN: {en_text}")
        print(f"VI: {result}")
    
    def test_translate_en_to_vi_medium(self, model, sample_texts):
        """Test EN→VI translation on medium text"""
        en_text = sample_texts["medium_en"]
        result = model.translate_en_to_vi(en_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        print(f"\nEN: {en_text}")
        print(f"VI: {result}")
    
    def test_translate_en_to_vi_long(self, model, sample_texts):
        """Test EN→VI translation on long text"""
        en_text = sample_texts["long_en"]
        result = model.translate_en_to_vi(en_text)
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        assert len(result.split()) > 5, "Translation should have multiple words"
        print(f"\nEN (long): {en_text[:100]}...")
        print(f"VI: {result[:100]}...")
    
    # ==================== TEST translate() ====================
    
    def test_translate_vi_to_en(self, model, sample_texts):
        """Test translate() method VI→EN"""
        vi_text = sample_texts["medium_vi"]
        result = model.translate(vi_text, source_lang="vi", target_lang="en")
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        print(f"\nVI: {vi_text}")
        print(f"EN: {result}")
    
    def test_translate_en_to_vi(self, model, sample_texts):
        """Test translate() method EN→VI"""
        en_text = sample_texts["medium_en"]
        result = model.translate(en_text, source_lang="en", target_lang="vi")
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Translation should not be empty"
        print(f"\nEN: {en_text}")
        print(f"VI: {result}")
    
    def test_translate_same_language(self, model, sample_texts):
        """Test translate() when source and target are same"""
        text = sample_texts["medium_vi"]
        result = model.translate(text, source_lang="vi", target_lang="vi")
        
        assert result == text, "Should return same text when source=target"
        print(f"\nSame language: {result}")
    
    def test_translate_invalid_language_pair(self, model, sample_texts):
        """Test translate() with invalid language pair"""
        text = sample_texts["medium_vi"]
        result = model.translate(text, source_lang="fr", target_lang="es")
        
        assert result == "", "Should return empty string for unsupported language"
        print("\n✓ Invalid language pair handled correctly")
    
    def test_translate_with_custom_max_length(self, model, sample_texts):
        """Test translate() with custom max_length"""
        vi_text = sample_texts["long_vi"]
        
        result_short = model.translate(vi_text, max_length=100)
        result_long = model.translate(vi_text, max_length=256)
        
        assert isinstance(result_short, str), "Should work with custom max_length"
        assert isinstance(result_long, str), "Should work with custom max_length"
        # Shorter max_length might result in shorter output
        assert len(result_short) > 0, "Even short max_length should produce output"
        print(f"\nmax_length=100: {len(result_short)} chars")
        print(f"max_length=256: {len(result_long)} chars")
    
    # ==================== TEST Edge Cases ====================
    
    def test_translate_empty_string(self, model):
        """Test translation with empty string"""
        result = model.translate_vi_to_en("")
        assert result == "", "Empty string should return empty result"
        
        result = model.translate_en_to_vi("")
        assert result == "", "Empty string should return empty result"
        
        result = model.translate("")
        assert result == "", "Empty string should return empty result"
        print("\n✓ Empty string handled correctly")
    
    def test_translate_whitespace_only(self, model):
        """Test translation with whitespace only"""
        result = model.translate_vi_to_en("   \n\t  ")
        assert result == "", "Whitespace-only should return empty result"
        
        result = model.translate_en_to_vi("   \n\t  ")
        assert result == "", "Whitespace-only should return empty result"
        print("\n✓ Whitespace-only handled correctly")
    
    def test_translate_none_input(self, model):
        """Test translation with None input"""
        result = model.translate_vi_to_en(None)
        assert result == "", "None input should return empty result"
        
        result = model.translate_en_to_vi(None)
        assert result == "", "None input should return empty result"
        
        result = model.translate(None)
        assert result == "", "None input should return empty result"
        print("\n✓ None input handled correctly")
    
    def test_translate_very_long_text(self, model):
        """Test translation with very long text (truncation test)"""
        long_text = "Đây là một câu dài. " * 100  # Very long text
        
        result = model.translate_vi_to_en(long_text)
        assert isinstance(result, str), "Should handle very long text"
        assert len(result) > 0, "Should produce output even for very long text"
        print(f"\nVery long text (input: {len(long_text)} chars)")
        print(f"Output: {len(result)} chars")
    
    def test_translate_repeated_words(self, model):
        """Test translation with repeated words"""
        text = "hello hello hello hello hello"
        result = model.translate_en_to_vi(text)
        
        assert isinstance(result, str), "Should handle repeated words"
        assert len(result) > 0, "Should produce output for repeated words"
        print(f"\nRepeated words: {text}")
        print(f"Translation: {result}")
    
    def test_translate_punctuation_only(self, model):
        """Test translation with punctuation only"""
        text = "... !!! ???"
        result = model.translate_en_to_vi(text)
        
        assert isinstance(result, str), "Should handle punctuation"
        print(f"\nPunctuation: {text}")
        print(f"Translation: {result}")
    
    # ==================== TEST Return Types & Format ====================
    
    def test_return_type_is_string(self, model, sample_texts):
        """Test that all methods return strings"""
        vi_text = sample_texts["medium_vi"]
        en_text = sample_texts["medium_en"]
        
        result1 = model.translate_vi_to_en(vi_text)
        result2 = model.translate_en_to_vi(en_text)
        result3 = model.translate(vi_text, source_lang="vi", target_lang="en")
        
        assert isinstance(result1, str), "translate_vi_to_en should return string"
        assert isinstance(result2, str), "translate_en_to_vi should return string"
        assert isinstance(result3, str), "translate should return string"
        print("\n✓ All methods return strings")
    
    def test_result_is_stripped(self, model, sample_texts):
        """Test that results are stripped of leading/trailing whitespace"""
        vi_text = sample_texts["medium_vi"]
        result = model.translate_vi_to_en(vi_text)
        
        assert result == result.strip(), "Result should be stripped"
        print(f"\n✓ Result properly stripped: '{result}'")
    
    def test_result_not_empty_for_valid_input(self, model, sample_texts):
        """Test that valid inputs always produce non-empty output"""
        test_cases = [
            ("translate_vi_to_en", sample_texts["short_vi"]),
            ("translate_vi_to_en", sample_texts["medium_vi"]),
            ("translate_en_to_vi", sample_texts["short_en"]),
            ("translate_en_to_vi", sample_texts["medium_en"]),
        ]
        
        for method_name, text in test_cases:
            if method_name == "translate_vi_to_en":
                result = model.translate_vi_to_en(text)
            else:
                result = model.translate_en_to_vi(text)
            
            assert len(result) > 0, f"{method_name} should produce non-empty output"
        
        print("\n✓ All valid inputs produce non-empty output")
    
    # ==================== TEST Device & Initialization ====================
    
    def test_model_device(self, model):
        """Test that model is on correct device"""
        device = model.get_device()
        
        assert device in ["cpu", "cuda"], "Device should be 'cpu' or 'cuda'"
        print(f"\n✓ Model on device: {device}")
    
    def test_model_is_initialized(self, model):
        """Test that model has required attributes"""
        assert hasattr(model, 'model_vi_en'), "Should have model_vi_en"
        assert hasattr(model, 'model_en_vi'), "Should have model_en_vi"
        assert hasattr(model, 'tokenizer_vi_en'), "Should have tokenizer_vi_en"
        assert hasattr(model, 'tokenizer_en_vi'), "Should have tokenizer_en_vi"
        assert hasattr(model, 'device'), "Should have device attribute"
        print("\n✓ Model properly initialized with all attributes")
    
    def test_models_in_eval_mode(self, model):
        """Test that models are in evaluation mode"""
        assert not model.model_vi_en.training, "VI→EN model should be in eval mode"
        assert not model.model_en_vi.training, "EN→VI model should be in eval mode"
        print("\n✓ Both models in evaluation mode")
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_round_trip_translation(self, model, sample_texts):
        """Test round-trip translation (VI→EN→VI)"""
        original_vi = sample_texts["medium_vi"]
        
        # Translate VI → EN
        en_text = model.translate_vi_to_en(original_vi)
        assert len(en_text) > 0, "First translation should not be empty"
        
        # Translate EN → VI
        back_to_vi = model.translate_en_to_vi(en_text)
        assert len(back_to_vi) > 0, "Round-trip translation should not be empty"
        
        print(f"\nOriginal VI: {original_vi}")
        print(f"Translated EN: {en_text}")
        print(f"Back to VI: {back_to_vi}")
    
    def test_multiple_translations_consistency(self, model, sample_texts):
        """Test that same input produces same output multiple times"""
        vi_text = sample_texts["medium_vi"]
        
        result1 = model.translate_vi_to_en(vi_text)
        result2 = model.translate_vi_to_en(vi_text)
        
        assert result1 == result2, "Same input should produce same output"
        print(f"\n✓ Consistent results: '{result1}'")
    
    def test_batch_like_translations(self, model, sample_texts):
        """Test translating multiple texts in sequence"""
        texts = [
            sample_texts["short_vi"],
            sample_texts["medium_vi"],
            sample_texts["special_chars_vi"],
        ]
        
        results = [model.translate_vi_to_en(text) for text in texts]
        
        assert len(results) == len(texts), "Should have result for each input"
        assert all(isinstance(r, str) for r in results), "All results should be strings"
        assert all(len(r) > 0 for r in results), "All results should be non-empty"
        
        print(f"\n✓ Batch translation: {len(results)} texts processed")


if __name__ == "__main__":
    """
    Run tests with:
        pytest tests/test_translation.py -v
    
    Run specific test:
        pytest tests/test_translation.py::TestTranslationModel::test_translate_vi_to_en_short -v
    
    Run with coverage:
        pytest tests/test_translation.py -v --cov=models.translation
    """
    pytest.main([__file__, "-v", "-s"])