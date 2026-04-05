import argparse
import json
import os
import random
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://vnexpress.net"
CATEGORY_URL = "https://vnexpress.net/thoi-su"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_article_links():
    """Lấy link bài viết từ trang category"""
    response = requests.get(CATEGORY_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    articles = soup.find_all("h3", class_="title-news")

    for art in articles:
        a_tag = art.find("a")
        if a_tag and a_tag.get("href"):
            links.append(a_tag["href"])

    return links


def get_article_content(url):
    """Lấy nội dung bài viết"""
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1", class_="title-detail")
        description = soup.find("p", class_="description")
        content = soup.find_all("p", class_="Normal")

        text = ""

        if title:
            text += title.text.strip() + "\n\n"

        if description:
            text += description.text.strip() + "\n\n"

        for p in content:
            text += p.text.strip() + "\n"

        return text.strip()

    except Exception as e:
        print(f"Lỗi khi crawl {url}: {e}")
        return None


def crawl_data(max_articles=50):
    """Crawl nhiều bài"""
    links = get_article_links()
    data = []

    for i, link in enumerate(links):
        if i >= max_articles:
            break

        print(f"Đang crawl: {link}")
        text = get_article_content(link)

        if text and len(text.split()) > 200:  # lọc bài quá ngắn
            data.append({
                "text_vi": text
            })

        time.sleep(random.uniform(1, 2))  # tránh bị block

    return data


def save_jsonl(data, filename="data/raw_data.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl_raw(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Chi luu bai crawl lan nay, ghi de len raw_data.jsonl",
    )
    ap.add_argument("--max-articles", type=int, default=50)
    args = ap.parse_args()

    out_path = "data/raw_data.jsonl"
    fresh = crawl_data(max_articles=args.max_articles)

    if args.replace:
        merged = fresh
        before = 0
    else:
        existing = load_jsonl_raw(out_path)
        before = len(existing)
        seen = {r["text_vi"] for r in existing if r.get("text_vi")}
        merged = list(existing)
        for row in fresh:
            tv = row.get("text_vi")
            if tv and tv not in seen:
                seen.add(tv)
                merged.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_jsonl(merged, out_path)

    if args.replace:
        print(f"Ghi de {len(merged)} bai vao {out_path}")
    else:
        print(f"Tong {len(merged)} bai trong {out_path} (truoc {before}, them {len(merged) - before})")