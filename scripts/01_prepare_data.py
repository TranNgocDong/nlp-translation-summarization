import requests
from bs4 import BeautifulSoup
import json
import time
import random

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


if __name__ == "__main__":
    dataset = crawl_data(max_articles=50)
    save_jsonl(dataset)

    print(f"Đã lưu {len(dataset)} bài vào file JSONL")