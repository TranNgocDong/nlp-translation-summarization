import argparse
import json
import os
import random
import sys
import time
from urllib.parse import urljoin, urlparse, urlunparse
import re
import hashlib

import requests
from bs4 import BeautifulSoup

# Hỗ trợ in tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions if needed, though 3.7+ is standard now
        pass

# Danh sách User-Agents giả lập các trình duyệt khác nhau
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

DATA_FILE = "data/raw_data.jsonl"

# CẤU HÌNH CÁC NGUỒN CÀO DỮ LIỆU
# CẤU HÌNH CÁC NGUỒN CÀO DỮ LIỆU ĐA LĨNH VỰC
SOURCES = {
    # ================= 1. TIN TỨC TỔNG HỢP & CHÍNH THỐNG =================
    "VnExpress": {
        "category_urls": [
            "https://vnexpress.net/thoi-su",          # Chính trị, Xã hội
            "https://vnexpress.net/the-gioi",         # Quốc tế
            "https://vnexpress.net/phap-luat",        # Hình sự, Tòa án
            "https://vnexpress.net/giao-duc"          # Trường học
        ],
        "list_selector": "h3.title-news a",
        "title_selector": "h1.title-detail",
        "desc_selector": "p.description",
        "content_selector": "article.fck_detail p.Normal"
    },
    "TuoiTre": {
        "category_urls": [
            "https://tuoitre.vn/thoi-su.htm",         
            "https://tuoitre.vn/the-gioi.htm",
            "https://tuoitre.vn/phap-luat.htm"
        ],
        "list_selector": "h3.box-title-text a",
        "title_selector": "h1.detail-title",
        "desc_selector": "h2.detail-sapo",
        "content_selector": "div.detail-cmain p"
    },
    "DanTri": {
        "category_urls": [
            "https://dantri.com.vn/xa-hoi.htm",
            "https://dantri.com.vn/the-gioi.htm"
        ],
        "list_selector": "article.article-item h3.article-title a",
        "title_selector": "h1.title-page",
        "desc_selector": "h2.singular-sapo",
        "content_selector": "div.singular-content p"
    },

    # ================= 2. KINH TẾ, TÀI CHÍNH, DOANH NGHIỆP =================
    "CafeF": {
        "category_urls": [
            "https://cafef.vn/tai-chinh-ngan-hang.chn",   # Tiền tệ, Bank
            "https://cafef.vn/chungkhoan.chn",            # Cổ phiếu
            "https://cafef.vn/bat-dong-san.chn"           # Địa ốc
        ],
        "list_selector": "h3.tlitem a, h3.title a",
        "title_selector": "h1.title",
        "desc_selector": "h2.sapo",
        "content_selector": "div.detail-content p"
    },
    "VnEconomy": {
        "category_urls": [
            "https://vneconomy.vn/tai-chinh.htm",
            "https://vneconomy.vn/kinh-te-so.htm"
        ],
        "list_selector": "h3.story__title a",
        "title_selector": "h1.detail__title",
        "desc_selector": "h2.detail__summary",
        "content_selector": "div.detail__content p"
    },

    # ================= 3. CÔNG NGHỆ & KHOA HỌC =================
    "GenK": {
        "category_urls": [
            "https://genk.vn/tin-ict.chn",              # Điện thoại, Laptop
            "https://genk.vn/kham-pha.chn"              # Khoa học, Vũ trụ
        ],
        "list_selector": "h4.knw-title a",
        "title_selector": "h1.kbwc-title",
        "desc_selector": "h2.knc-sapo",
        "content_selector": "div.knc-content p"
    },

    # ================= 4. GIỚI TRẺ, GIẢI TRÍ, VĂN HÓA =================
    "Kenh14": {
        "category_urls": [
            "https://kenh14.vn/star.chn",               # Showbiz
            "https://kenh14.vn/cine.chn"                # Phim ảnh
        ],
        "list_selector": "h3.klwfn-title a",
        "title_selector": "h1.kbwc-title",
        "desc_selector": "h2.knc-sapo",
        "content_selector": "div.knc-content p"
    },

    # ================= 5. ĐỜI SỐNG, SỨC KHỎE =================
    "SucKhoeDoiSong": {
        "category_urls": [
            "https://suckhoedoisong.vn/y-te-c10.htm",
            "https://suckhoedoisong.vn/dinh-duong-c15.htm"
        ],
        "list_selector": "h3.box-title-text a",
        "title_selector": "h1.detail-title",
        "desc_selector": "h2.detail-sapo",
        "content_selector": "div.detail-content p"
    }
}

def get_random_header():
    return {"User-Agent": random.choice(USER_AGENTS)}

def normalize_url(url):
    """Bỏ các query params tracking (ví dụ: ?utm_source=...) để check trùng lặp."""
    parsed = urlparse(url)
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
    return clean_url

def clean_text(text):
    """Xóa khoảng trắng, tab, newline thừa."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_hash(text):
    """Tạo mã băm SHA-256 cho nội dung."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_existing_metadata(filepath):
    """Đọc file cũ để lấy tập hợp các URL và Hash đã cào."""
    existing_urls = set()
    existing_hashes = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_urls.add(obj.get("url"))
                    # Giả sử chúng ta lưu hash vào json, nếu chưa có thì hash lại content
                    content_hash = obj.get("content_hash") or compute_hash(obj.get("content_vi", ""))
                    existing_hashes.add(content_hash)
                except json.JSONDecodeError:
                    continue
    return existing_urls, existing_hashes


def get_article_links(category_url, config):
    try:
        response = requests.get(category_url, headers=get_random_header(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        articles = soup.select(config["list_selector"])
        for a_tag in articles:
            href = a_tag.get("href")
            if href:
                full_url = urljoin(category_url, href)
                links.append(full_url)
        return list(dict.fromkeys(links))
    except Exception as e:
        print(f"Lỗi khi lấy link từ {category_url}: {e}")
        return []

def get_article_content(url, config):
    try:
        response = requests.get(url, headers=get_random_header(), timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, "html.parser")

        # ===== 🧹 XÓA CÁC THÀNH PHẦN GÂY NHIỄU (CẬP NHẬT THÊM) =====
        unwanted_selectors = [
            "div.box-comment", "div.tags", "style", "script",
            "div.tin-lien-quan", "div.relate-news", 
            "div.VCSortableInPreviewMode",
            "figcaption", "div.author",
            "div.social", "div.share", "iframe",
            # Thêm các thẻ thường chứa chú thích ảnh, video, quảng cáo
            "figure", "picture", "div.video", "div.banner", 
            "table.picture", "p.Image", ".hidden", "[style*='display: none']"
        ]

        for unwanted in soup.select(", ".join(unwanted_selectors)):
            unwanted.decompose()

        title = sapo = content = ""

        # ===== 📌 TITLE =====
        if config.get("title_selector"):
            t_tag = soup.select_one(config["title_selector"])
            if t_tag:
                title = clean_text(t_tag.get_text(" ", strip=True))

        # ===== 📌 SAPO =====
        if config.get("desc_selector"):
            d_tag = soup.select_one(config["desc_selector"])
            if d_tag:
                sapo = clean_text(d_tag.get_text(" ", strip=True))

        # ===== 📌 CONTENT =====
        if config.get("content_selector"):
            paragraphs = soup.select(config["content_selector"])
            content_list = []
            seen_texts = set() # 🟢 THÊM: Bộ lặp để check trùng nội dung

            for p in paragraphs:
                txt = clean_text(p.get_text(" ", strip=True))

                # 🚫 Lọc nội dung rác (rỗng)
                if not txt:
                    continue

                blacklist = [
                    "Tối đa:", "ký tự", "bình luận", 
                    "chia sẻ", "quảng cáo", 
                    "xem thêm", "đọc thêm",
                    "Đồ họa:", "Ảnh:", "Video:" # Thêm keyword chặn nguồn ảnh/video
                ]

                if any(b.lower() in txt.lower() for b in blacklist):
                    continue

                # 🚫 Bỏ đoạn quá ngắn (thường là rác hoặc tag thừa)
                if len(txt.split()) < 5:
                    continue
                
                # 🟢 THÊM: Chống lặp đoạn văn (Trị dứt điểm lỗi lặp 2-3 lần)
                if txt in seen_texts:
                    continue
                seen_texts.add(txt)

                # Kiểm tra nếu đoạn văn trùng với sapo thì bỏ qua
                if txt == sapo:
                    continue

                content_list.append(txt)

            content = "\n".join(content_list)

        return {
            "title": title,
            "sapo_vi": sapo,
            "content_vi": content
        }

    except Exception as e:
        print(f"Lỗi khi cào {url}: {e}")
        return None

def append_to_jsonl(item, filename):
    """Ghi ngay lập tức vào file (Append mode)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def crawl_data(max_articles_per_source=20):
    existing_urls, existing_hashes = load_existing_metadata(DATA_FILE)
    print(f"Đã tải {len(existing_urls)} bài viết cũ từ database.")

    total_added = 0
    for source_name, config in SOURCES.items():
        print(f"\n[{source_name}] Đang lấy link bài viết...")
        source_links = []
        for cat_url in config["category_urls"]:
            source_links.extend(get_article_links(cat_url, config))
            time.sleep(random.uniform(0.5, 1.5))
            
        count = 0
        for link in source_links:
            if count >= max_articles_per_source: break
            
            clean_link = normalize_url(link)
            if clean_link in existing_urls:
                print(f"Bỏ qua (URL trùng): {clean_link}")
                continue

            print(f"Đang cào: {clean_link}")
            res = get_article_content(clean_link, config)
            
            if res and len(res["content_vi"].split()) > 100:
                c_hash = compute_hash(res["content_vi"])
                if c_hash in existing_hashes:
                    print("Bỏ qua (Nội dung trùng với bài khác).")
                    continue

                item = {
                    "source": source_name,
                    "url": clean_link,
                    "content_hash": c_hash, # Lưu lại hash để lần sau đọc nhanh hơn
                    "title": res["title"],
                    "sapo_vi": res["sapo_vi"],
                    "content_vi": res["content_vi"]
                }
                append_to_jsonl(item, DATA_FILE)
                existing_urls.add(clean_link)
                existing_hashes.add(c_hash)
                count += 1
                total_added += 1
            time.sleep(random.uniform(1.0, 2.0))
            
    return total_added

def save_jsonl(data, filename="data/raw_data.jsonl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=20)
    args = ap.parse_args()

    total_fresh = crawl_data(max_articles_per_source=args.max_articles)
    print(f"\n✅ Đã lưu thêm {total_fresh} bài báo mới vào {DATA_FILE}")