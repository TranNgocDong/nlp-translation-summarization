# nlp-translation-summarization

Đây là dự án tóm tắt song ngữ dùng `ViT5`, gồm:

- Model tiếng Việt: `text_vi -> summary_vi`
- Model tiếng Anh: `text_en -> summary_en`

Giao diện được viết bằng `Streamlit`, còn phần xử lý phía sau dùng `FastAPI`.

## Cách chạy dự án

### Bước 1: Vào đúng thư mục dự án

Chạy lệnh:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
```

### Bước 2: Cài đặt thư viện

Tùy thuộc vào phần cứng của bạn, hãy chọn file requirements phù hợp:

#### GPU CUDA 12.4
```powershell
python -m pip install -r requirements-cuda.txt
```

#### CPU Only
```powershell
python -m pip install -r requirements-cpu.txt
```

Kiểm tra nhanh (đối với bản CUDA):
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```
Nếu in ra `True` và phiên bản có hậu tố `+cu124` thì GPU đã được PyTorch nhận.


### Bước 3: Huấn luyện model nếu chưa có checkpoint

Nếu trong thư mục `models/` chưa có:

- `models/vit5-summarize-vi/best_checkpoint`
- `models/vit5-summarize-en/best_checkpoint`

thì cần huấn luyện trước.

Cách nhanh (menu chọn sẵn):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/train_menu.py
```

Menu đã có các lựa chọn:
- Train tiếng Việt
- Train tiếng Anh
- Train cả 2
- Resume train (tiếp tục từ checkpoint gần nhất)

Nếu muốn chạy thẳng bằng lệnh:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python models/trainModelsAI/train_vit5_summarize.py --lang vi
python models/trainModelsAI/train_vit5_summarize.py --lang en
```

Nếu đã có sẵn checkpoint thì có thể bỏ qua bước này.

### Bước 4: Chạy server API

Chạy lệnh:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Sau khi chạy, server mặc định lắng nghe tại:

- `http://localhost:8000`
- `http://localhost:8000/health`

### Bước 5: Chạy giao diện Streamlit

Mở một terminal khác rồi chạy:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
python -m streamlit run UI/app.py
```

Hoặc chạy nhanh:

```powershell
python UI/app.py
```

(`UI/app.py` sẽ tự động bootstrap lại bằng `streamlit run` nếu bạn chạy trực tiếp bằng Python.)

Sau đó mở trình duyệt tại:

- `http://localhost:8501`

### Bước 6: Sử dụng trên giao diện

Quy trình sử dụng cơ bản:

1. Nhập văn bản vào ô nội dung.
2. Chọn ngôn ngữ đầu vào.
3. Bấm `Process`.
4. Hệ thống sẽ gọi `POST /api/process` và trả về:

- `summary_vi`
- `summary_en`
- `translated_text`
- `entities`
- `relation_graph`
- `metadata`

### Bước 7: Thử suy luận bằng dòng lệnh

Ví dụ tóm tắt tiếng Việt:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
python models/summarization/inference_vit5_summarize.py --lang vi --text_vi "Nội dung cần tóm tắt"
```

Ví dụ tóm tắt tiếng Anh:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
python models/summarization/inference_vit5_summarize.py --lang en --text_en "Text to summarize"
```

Ví dụ chạy cả hai model:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
python models/summarization/inference_vit5_summarize.py --lang both --text_vi "Nội dung tiếng Việt" --text_en "English content"
```

## Lưu ý

- Dự án cung cấp `requirements-cuda.txt` và `requirements-cpu.txt` để hỗ trợ cả môi trường GPU và CPU. Hãy chọn file phù hợp khi cài đặt.
- GPU 4GB có thể đủ cho inference; khi train `ViT5-base` nếu hết VRAM thì thử giảm `batch_size` trong script train.
- Streamlit cần được chạy bằng lệnh `python -m streamlit run UI/app.py`, không chạy trực tiếp bằng `python UI/app.py`.
- Khi train model hoặc chạy inference, nên thiết lập `PYTHONPATH` bằng lệnh `($env:PYTHONPATH = (Get-Location).Path)`.
- Dự án đang dùng `transformers==4.57.6` để tránh lỗi tokenizer với `VietAI/vit5-base`.
- Lần đầu chạy có thể cần tải model từ Hugging Face, vì vậy cần có kết nối Internet.
- Model tiếng Anh hiện có thể cho kết quả kém ổn định hơn tiếng Việt nếu dữ liệu `text_en` và `summary_en` còn nhiễu hoặc chưa sạch.
- Trường `translated_text` sẽ là bản dịch nếu có model dịch cục bộ cho hướng dịch đã chọn; nếu không, API có thể fallback về văn bản gốc.
- Trường `entities` hiện được suy ra từ `relation_graph` theo luật (rule-based), chưa phải NER học sâu.
