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

### Bước 2: Cài thư viện cần thiết

Chạy lệnh:

```powershell
py -3 -m pip install -r requirements.txt
```

### Bước 3: Huấn luyện model nếu chưa có checkpoint

Nếu trong thư mục `models/` chưa có:

- `models/vit5-summarize-vi/best_checkpoint`
- `models/vit5-summarize-en/best_checkpoint`

thì cần huấn luyện trước bằng các lệnh:

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3 scripts/train_vit5_summarize.py --lang vi
py -3 scripts/train_vit5_summarize.py --lang en
```

Nếu đã có sẵn checkpoint thì có thể bỏ qua bước này.

### Bước 4: Chạy server API

Chạy lệnh:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
py -3 server.py
```

Sau khi chạy, server mặc định lắng nghe tại:

- `http://localhost:8000`
- `http://localhost:8000/health`

### Bước 5: Chạy giao diện Streamlit

Mở một terminal khác rồi chạy:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
py -3 -m streamlit run UI/app.py
```

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
- `translated_text_en`
- `entities`

### Bước 7: Thử suy luận bằng dòng lệnh

Ví dụ tóm tắt tiếng Việt:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
py -3 scripts/inference_vit5_summarize.py --lang vi --text_vi "Nội dung cần tóm tắt"
```

Ví dụ tóm tắt tiếng Anh:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
py -3 scripts/inference_vit5_summarize.py --lang en --text_en "Text to summarize"
```

Ví dụ chạy cả hai model:

```powershell
Set-Location "d:\AI\DA\nlp-translation-summarization"
$env:PYTHONPATH = (Get-Location).Path
py -3 scripts/inference_vit5_summarize.py --lang both --text_vi "Nội dung tiếng Việt" --text_en "English content"
```

## Lưu ý

- Streamlit cần được chạy bằng lệnh `py -3 -m streamlit run UI/app.py`, không chạy trực tiếp bằng `python UI/app.py`.
- Khi train model hoặc chạy inference, nên thiết lập `PYTHONPATH` bằng lệnh `($env:PYTHONPATH = (Get-Location).Path)`.
- Dự án đang dùng `transformers==4.57.6` để tránh lỗi tokenizer với `VietAI/vit5-base`.
- Lần đầu chạy có thể cần tải model từ Hugging Face, vì vậy cần có kết nối Internet.
- Model tiếng Anh hiện có thể cho kết quả kém ổn định hơn tiếng Việt nếu dữ liệu `text_en` và `summary_en` còn nhiễu hoặc chưa sạch.
- Trường `translated_text_en` trong API hiện mới trả lại văn bản đầu vào, chưa nối với model dịch riêng.
- Trường `entities` hiện chưa tích hợp NER, nên đang trả về danh sách rỗng.