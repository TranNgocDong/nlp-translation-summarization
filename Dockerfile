# Sử dụng Python image chính thức
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống tối thiểu cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng cho API và UI
EXPOSE 8000
EXPOSE 8501

# Lệnh khởi chạy mặc định (Docker Compose sẽ ghi đè lệnh này)
CMD ["python", "server.py"]