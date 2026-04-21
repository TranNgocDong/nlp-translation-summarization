from typing import List, Dict
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .vit5_wrapper import VIT5Summarizer
    from .config import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, MIN_OUTPUT_LENGTH, NUM_BEAMS, LENGTH_PENALTY
except ImportError:
    # For direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.summarization.vit5_wrapper import VIT5Summarizer
    from models.summarization.config import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, MIN_OUTPUT_LENGTH, NUM_BEAMS, LENGTH_PENALTY


class HierarchicalSummarizer:
    """
    Hierarchical summarization for long documents
    
    Workflow:
    1. Split long document into chunks
    2. Summarize each chunk
    3. Combine chunk summaries
    4. Summarize combined summary
    5. Return final summary with metadata
    
    Example:
        >>> summarizer = HierarchicalSummarizer()
        >>> long_text = "Văn bản dài 5000+ từ..."
        >>> result = summarizer.hierarchical_summarize(long_text)
        >>> print(result['final_summary'])
    """
    
    def __init__(self, checkpoint_dir: str = None, device: str = None):
        """
        Initialize hierarchical summarizer
        
        Args:
            checkpoint_dir (str): Path to model checkpoint. 
                                 Defaults to models/summarization/checkpoints/best_model
            device (str): Device to use ("cpu" or "cuda"). 
                         Defaults to "cuda" if available, else "cpu"
        
        Raises:
            Exception: If both checkpoint and pre-trained model fail to load
        """
        print("Initializing Hierarchical Summarizer...")
        
        if checkpoint_dir is None:
            # Use Path object to handle cross-platform paths
            checkpoint_dir = Path(__file__).parent / "checkpoints" / "best_model"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Convert to string with forward slashes (compatible with Hugging Face)
        checkpoint_dir_str = str(checkpoint_dir).replace("\\", "/")
        
        try:
            self.summarizer = VIT5Summarizer(checkpoint_dir_str, device=device)
            print(f"✅ Loaded model from checkpoint: {checkpoint_dir_str}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("   Falling back to pre-trained VIT5 model (VietAI/vit5-base)...")
            
            try:
                self.summarizer = VIT5Summarizer("VietAI/vit5-base", device=device)
                print("✅ Loaded pre-trained model: VietAI/vit5-base")
            except Exception as e2:
                print(f"❌ Error loading pre-trained model: {e2}")
                raise
        
        self.chunk_size = 1000  # Default chunk size in words
        self.device = self.summarizer.device
        
        print(f"✅ Hierarchical Summarizer initialized")
        print(f"   - Chunk size: {self.chunk_size} words")
        print(f"   - Device: {self.device}")
    
    def split_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Split text into chunks by sentences
        
        Args:
            text (str): Input text to split
            chunk_size (int): Approximate chunk size in words. 
                            Defaults to self.chunk_size
        
        Returns:
            List[str]: List of text chunks (each is a sentence or group of sentences)
        
        Raises:
            None (returns empty list on error)
        
        Example:
            >>> summarizer = HierarchicalSummarizer()
            >>> text = "Câu 1. Câu 2. Câu 3."
            >>> chunks = summarizer.split_text(text, chunk_size=100)
            >>> print(len(chunks))  # Number of chunks
            >>> print(chunks[0])    # First chunk
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        # Input validation
        if not text or not isinstance(text, str):
            return []
        
        # Strip whitespace
        text = text.strip()
        if not text:
            return []
        
        # Normalize punctuation (Chinese/Japanese style to English style)
        text = text.replace('。', '.')  # Chinese period
        text = text.replace('！', '!')  # Chinese exclamation
        text = text.replace('？', '?')  # Chinese question mark
        text = text.replace('；', ';')  # Chinese semicolon
        text = text.replace('，', ',')  # Chinese comma
        
        # Split by sentence markers
        sentences = []
        for delimiter in ['.', '!', '?', ';']:
            if delimiter in text:
                parts = text.split(delimiter)
                sentences = [s.strip() for s in parts if s.strip()]
                if sentences:
                    break
        
        # Fallback: split by newline if no sentence markers found
        if not sentences:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        # If still no sentences, return the whole text as one chunk
        if not sentences:
            return [text]
        
        # Group sentences into chunks based on word count
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                # Join sentences with period
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks if chunks else [text]
    
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Summarize each chunk individually
        
        Args:
            chunks (List[str]): List of text chunks to summarize
        
        Returns:
            List[str]: List of summaries (one per chunk)
        
        Note:
            - Returns empty list if input is empty
            - Skips chunks that fail to summarize (adds empty string)
            - Uses parameters from config
        
        Example:
            >>> chunks = ["Text chunk 1...", "Text chunk 2..."]
            >>> summaries = summarizer.summarize_chunks(chunks)
            >>> print(len(summaries))  # Should equal len(chunks)
        """
        if not chunks or not isinstance(chunks, list):
            return []
        
        summaries = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks, 1):
            if not chunk or not isinstance(chunk, str):
                print(f"  ⚠️ Warning: Skipping invalid chunk {i}/{total_chunks}")
                summaries.append("")
                continue
            
            print(f"  Summarizing chunk {i}/{total_chunks}...")
            
            try:
                # Call VIT5Summarizer.summarize() - returns dict
                result = self.summarizer.summarize(
                    text=chunk,
                    max_input_length=MAX_INPUT_LENGTH,
                    max_new_tokens=MAX_OUTPUT_LENGTH,
                    min_new_tokens=MIN_OUTPUT_LENGTH,
                    num_beams=NUM_BEAMS,
                    length_penalty=LENGTH_PENALTY
                )
                
                # Extract summary from result dict
                # VIT5Summarizer returns: {"summary": "...", "language": "..."}
                if isinstance(result, dict) and 'summary' in result:
                    summary = result['summary']
                else:
                    # Fallback if return format is different
                    summary = str(result) if result else ""
                
                summaries.append(summary)
                print(f"    ✓ Chunk {i} summarized ({len(summary.split())} words)")
            
            except Exception as e:
                print(f"  ⚠️ Error summarizing chunk {i}: {str(e)[:100]}")
                summaries.append("")
        
        return summaries
    
    def combine_summaries(self, summaries: List[str]) -> str:
        """
        Combine multiple chunk summaries into one text
        
        Args:
            summaries (List[str]): List of chunk summaries
        
        Returns:
            str: Combined summary text (empty string if all summaries are empty)
        
        Note:
            - Filters out empty summaries
            - Joins with space
            - Preserves sentence structure
        
        Example:
            >>> summaries = ["Summary 1.", "Summary 2.", "Summary 3."]
            >>> combined = summarizer.combine_summaries(summaries)
            >>> print(combined)
            "Summary 1. Summary 2. Summary 3."
        """
        if not summaries or not isinstance(summaries, list):
            return ""
        
        # Filter empty summaries
        valid_summaries = [s for s in summaries if s and isinstance(s, str) and s.strip()]
        
        if not valid_summaries:
            return ""
        
        # Combine with space (summaries already end with period)
        combined = " ".join(valid_summaries)
        
        # Clean up any double spaces or extra punctuation
        combined = " ".join(combined.split())
        
        return combined
    
    def hierarchical_summarize(
        self,
        text: str,
        chunk_size: int = None,
        final_length: int = MAX_OUTPUT_LENGTH
    ) -> Dict:
        """
        Perform hierarchical summarization on long document
        
        Process:
        1. Check if text fits in one chunk
        2. If yes: summarize directly and return
        3. If no: split into chunks
        4. Summarize each chunk individually
        5. Combine chunk summaries
        6. Recursively summarize if combined is still too long
        7. Return final summary with metadata
        
        Args:
            text (str): Long input text to summarize
            chunk_size (int): Size of each chunk in words. 
                            Defaults to self.chunk_size (1000)
            final_length (int): Desired length of final summary in tokens. 
                              Defaults to MAX_OUTPUT_LENGTH (256)
        
        Returns:
            Dict: Result dictionary containing:
                {
                    "original_length": int,        # Original text length in words
                    "chunks": int,                 # Number of chunks created
                    "chunk_summaries": List[str],  # Summary of each chunk
                    "combined_summary": str,       # Combined chunk summaries
                    "final_summary": str,          # Final hierarchical summary
                    "compression_ratio": float     # Original length / Final length
                }
        
        Raises:
            None (handles all errors internally)
        
        Example:
            >>> summarizer = HierarchicalSummarizer()
            >>> long_text = "Văn bản dài 5000+ từ..."
            >>> result = summarizer.hierarchical_summarize(long_text)
            >>> print(f"Original: {result['original_length']} words")
            >>> print(f"Final: {len(result['final_summary'].split())} words")
            >>> print(f"Compression: {result['compression_ratio']:.2f}x")
            >>> print(result['final_summary'])
        """
        # Input validation
        if not text or not isinstance(text, str):
            return {
                "original_length": 0,
                "chunks": 0,
                "chunk_summaries": [],
                "combined_summary": "",
                "final_summary": "",
                "compression_ratio": 0.0
            }
        
        text = text.strip()
        if not text:
            return {
                "original_length": 0,
                "chunks": 0,
                "chunk_summaries": [],
                "combined_summary": "",
                "final_summary": "",
                "compression_ratio": 0.0
            }
        
        print("\n" + "="*60)
        print("HIERARCHICAL SUMMARIZATION")
        print("="*60)
        
        # Calculate original text length
        text_length = len(text.split())
        print(f"\n📊 Original text length: {text_length} words")
        
        # CASE 1: Text is short enough - summarize directly
        if text_length <= MAX_INPUT_LENGTH:
            print(f"✓ Text fits in one chunk ({text_length} ≤ {MAX_INPUT_LENGTH})")
            print("→ Summarizing directly...")
            
            try:
                result = self.summarizer.summarize(
                    text=text,
                    max_input_length=MAX_INPUT_LENGTH,
                    max_new_tokens=final_length,
                    num_beams=NUM_BEAMS,
                    length_penalty=LENGTH_PENALTY
                )
                
                # Extract summary
                final_summary = result.get('summary', '') if isinstance(result, dict) else str(result)
                final_len = len(final_summary.split()) if final_summary else 1
                compression_ratio = text_length / final_len if final_len > 0 else 0
                
                print(f"✓ Summary generated ({final_len} words)")
                
                return {
                    "original_length": text_length,
                    "chunks": 1,
                    "chunk_summaries": [final_summary],
                    "combined_summary": final_summary,
                    "final_summary": final_summary,
                    "compression_ratio": compression_ratio
                }
            
            except Exception as e:
                print(f"❌ Error summarizing: {e}")
                return {
                    "original_length": text_length,
                    "chunks": 1,
                    "chunk_summaries": [],
                    "combined_summary": "",
                    "final_summary": "",
                    "compression_ratio": 0.0
                }
        
        # CASE 2: Text is long - use hierarchical approach
        print(f"⚠️ Text too long ({text_length} > {MAX_INPUT_LENGTH})")
        print("→ Using hierarchical approach...")
        
        # Step 1: Split into chunks
        print(f"\n📍 Step 1: Splitting into chunks")
        print(f"   Chunk size: {chunk_size or self.chunk_size} words")
        
        chunks = self.split_text(text, chunk_size)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Step 2: Summarize each chunk
        print(f"\n📍 Step 2: Summarizing each chunk")
        chunk_summaries = self.summarize_chunks(chunks)
        valid_summaries = [s for s in chunk_summaries if s and s.strip()]
        print(f"   ✓ Successfully summarized {len(valid_summaries)}/{len(chunks)} chunks")
        
        # Step 3: Combine summaries
        print(f"\n📍 Step 3: Combining chunk summaries")
        combined_summary = self.combine_summaries(chunk_summaries)
        combined_length = len(combined_summary.split()) if combined_summary else 0
        print(f"   ✓ Combined summary: {combined_length} words")
        
        # Step 4: Check if combined summary needs further summarization
        print(f"\n📍 Step 4: Final summarization")
        
        if combined_length > MAX_INPUT_LENGTH:
            print(f"⚠️ Combined summary still too long ({combined_length} > {MAX_INPUT_LENGTH})")
            print("→ Applying recursive hierarchical summarization...")
            
            final_result = self.hierarchical_summarize(
                combined_summary,
                chunk_size=chunk_size,
                final_length=final_length
            )
            final_summary = final_result['final_summary']
        
        else:
            print(f"✓ Combined summary fits in one pass")
            
            try:
                result = self.summarizer.summarize(
                    text=combined_summary,
                    max_input_length=MAX_INPUT_LENGTH,
                    max_new_tokens=final_length,
                    num_beams=NUM_BEAMS,
                    length_penalty=LENGTH_PENALTY
                )
                
                # Extract summary
                final_summary = result.get('summary', '') if isinstance(result, dict) else str(result)
                print(f"   ✓ Final summary generated")
            
            except Exception as e:
                print(f"   ❌ Error in final summarization: {e}")
                final_summary = combined_summary  # Fallback to combined
        
        # Calculate metrics
        final_len = len(final_summary.split()) if final_summary else 1
        compression_ratio = text_length / final_len if final_len > 0 else 0
        
        # Print results
        print(f"\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Original text:    {text_length} words")
        print(f"Chunks created:   {len(chunks)}")
        print(f"Combined summary: {combined_length} words")
        print(f"Final summary:    {final_len} words")
        print(f"Compression:      {compression_ratio:.2f}x")
        print("="*60 + "\n")
        
        return {
            "original_length": text_length,
            "chunks": len(chunks),
            "chunk_summaries": chunk_summaries,
            "combined_summary": combined_summary,
            "final_summary": final_summary,
            "compression_ratio": compression_ratio
        }


# Test code
if __name__ == "__main__":
    """
    Test the HierarchicalSummarizer with sample Vietnamese text
    """
    try:
        print("Starting HierarchicalSummarizer test...\n")
        
        # Initialize
        summarizer = HierarchicalSummarizer()
        
        # Sample long text (Vietnamese story)
        long_text = """
        Nhân vật A là một chiến binh mạnh mẽ sống trong một làng nhỏ ở núi cao. 
        Từ nhỏ, anh ta được dạy dỗ bởi sư phụ về nghệ thuật kiếm pháp. 
        Anh ta có một tấm lòng vàng và luôn sẵn sàng giúp đỡ những người yếu thế. 
        Một hôm, quân thù từ phía đông tấn công làng với số lượng rất lớn. 
        Nhân vật A đã dũng cảm đứng lên chống lại quân thù để bảo vệ làng. 
        Sau một trận chiến đẫm máu, anh ta đã đánh bại tất cả các kẻ thù. 
        Nhân vật A lấy được bảo vật cổ từ tay tướng chỉ huy quân thù. 
        Tất cả mọi người trong làng đều hân hoan khi được cứu sống. 
        Anh ta trở thành anh hùng của làng và được mọi người tôn kính. 
        Sau khi chiến thắng, nhân vật A gặp được công chúa B từ một vương quốc lân cận.
        Công chúa B cũng có một cuộc sống đầy khó khăn vì phải chống lại những kẻ thù của quốc gia.
        Hai người này dần hiểu nhau và phát triển tình cảm sâu sắc.
        Cùng nhau, họ lên kế hoạch để giúp hai vương quốc hòa bình.
        Trận chiến cuối cùng diễn ra trên núi cao nơi có tháp ma thuật cổ xưa.
        Nhân vật A và công chúa B phải đối mặt với những thách thức không tưởng.
        Họ vượt qua tất cả những bài kiểm tra và nhận được sức mạnh từ bảo vật cổ.
        Cuối cùng, họ đã đánh bại tất cả các thế lực thù địch.
        Hai vương quốc kết hợp lại thành một đế chế mạnh mẽ và hòa bình.
        """ * 2  # Repeat 2 times for longer text
        
        # Hierarchical summarization
        result = summarizer.hierarchical_summarize(long_text)
        
        # Display results
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(result['final_summary'])
        print("="*60)
        
        print(f"\n✅ Test completed successfully!")
        print(f"Compression Ratio: {result['compression_ratio']:.2f}x")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()