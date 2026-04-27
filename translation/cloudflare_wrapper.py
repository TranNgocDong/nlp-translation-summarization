# Nội dung file: translation/cloudflare_wrapper.py

class CloudflareWorkersTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate(self, text, *args, **kwargs):
        return {
            "translated_text": "[Cloudflare chưa được cài đặt] " + text,
            "note": "Bạn cần cấu hình Cloudflare API để dùng tính năng này."
        }