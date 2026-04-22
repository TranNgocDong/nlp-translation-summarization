from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

# --- REQUEST MODELS (Dữ liệu gửi lên từ UI) ---

class TextRequest(BaseModel):
    """Sử dụng cho các yêu cầu chỉ cần gửi văn bản đơn giản"""
    text: str = Field(..., min_length=1, description="Văn bản cần xử lý")

class TranslationRequest(BaseModel):
    """Sử dụng riêng cho yêu cầu dịch thuật"""
    text: str = Field(..., min_length=1, description="Văn bản cần dịch")

class ProcessRequest(BaseModel):
    """Cấu hình chi tiết cho quá trình tóm tắt và xử lý tổng hợp"""
    text: str = Field(..., min_length=1, description="Văn bản gốc cần tóm tắt hoặc xử lý")
    source_lang: Literal["vi", "en"] = Field(default="vi", description="Ngôn ngữ nguồn")
    target_lang: Literal["vi", "en"] = Field(default="en", description="Ngôn ngữ đích")
    
    # Các tham số điều chỉnh mô hình AI (đã cấu hình theo bài mẫu)
    max_input_length: int = Field(default=512, ge=128, le=1024)
    max_new_tokens: int = Field(default=160, ge=32, le=256)
    min_new_tokens: int = Field(default=48, ge=0, le=128)
    num_beams: int = Field(default=4, ge=1, le=8)
    length_penalty: float = Field(default=1.15, ge=0.8, le=2.0)
    chunk_size_words: int = Field(default=240, ge=120, le=400)
    chunk_overlap_words: int = Field(default=40, ge=0, le=120)

# --- RESPONSE MODELS (Dữ liệu API trả về cho UI) ---

class SummaryResponse(BaseModel):
    """Kết quả trả về của các endpoint tóm tắt"""
    summary: str
    metadata: Optional[Dict[str, Any]] = None

class TranslationResponse(BaseModel):
    """Kết quả trả về của các endpoint dịch thuật"""
    translated_text: str
    note: str

class EntityItem(BaseModel):
    """Thông tin chi tiết của một thực thể (nhân vật, địa điểm...)"""
    text: str
    type: str
    mentions: int

class EntitiesResponse(BaseModel):
    """Danh sách các thực thể trích xuất được"""
    entities: List[EntityItem]

class HealthResponse(BaseModel):
    """Thông tin trạng thái hoạt động của server và model 16GB"""
    status: str
    preload_status: Dict[str, str]

class ProcessResponse(BaseModel):
    """Kết quả tổng hợp khi chạy endpoint /api/process"""
    original_text: str
    translated_text: Optional[str] = None
    summary_vi: Optional[str] = None
    summary_en: Optional[str] = None
    entities: List[EntityItem]
    relation_graph: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None