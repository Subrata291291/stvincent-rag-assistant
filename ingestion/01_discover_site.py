import json
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://stvincentcbseburdwan.org/"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_inventory.json"
)

MAX_PAGES = 500

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    )
}


def normalize_url(url):
    url = urldefrag(url)[0]
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None

    if parsed.netloc.lower() != urlparse(BASE_URL).netloc.lower():
        return None

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    return f"https://{parsed.netloc}{path}"


def is_pdf(url):
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def is_html_page(url):
    path = urlparse(url).path.lower()

    excluded_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".css",
        ".js",
        ".xml",
        ".zip",
        ".rar",
        ".mp4",
        ".mp3",
        ".woff",
        ".woff2",
        ".ttf",
    )

    return not path.endswith(excluded_extensions)


def discover_site():
    queue = deque([BASE_URL])
    visited = set()

    pages = []
    pdfs = []

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(visited) < MAX_PAGES:

        current_url = queue.popleft()
        current_url = normalize_url(current_url)

        if not current_url:
            continue

        if current_url in visited:
            continue

        visited.add(current_url)

        print(f"[{len(visited)}] {current_url}")

        try:
            response = session.get(
                current_url,
                timeout=20
            )

            response.raise_for_status()

        except requests.RequestException as error:
            print(f"  ERROR: {error}")
            continue

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "application/pdf" in content_type or is_pdf(current_url):

            if current_url not in pdfs:
                pdfs.append(current_url)

            continue

        if "text/html" not in content_type:
            continue

        if current_url not in pages:
            pages.append(current_url)

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for link in soup.find_all("a", href=True):

            absolute_url = urljoin(
                current_url,
                link["href"]
            )

            normalized_url = normalize_url(
                absolute_url
            )

            if not normalized_url:
                continue

            if is_pdf(normalized_url):

                if normalized_url not in pdfs:
                    pdfs.append(normalized_url)

                continue

            if is_html_page(normalized_url):
                if normalized_url not in visited:
                    queue.append(normalized_url)

    return pages, pdfs


def save_inventory(pages, pdfs):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    inventory = {
        "base_url": BASE_URL,
        "pages": sorted(pages),
        "pdfs": sorted(pdfs),
        "page_count": len(pages),
        "pdf_count": len(pdfs)
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inventory,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("=" * 60)
    print("ST. VINCENT'S ACADEMY - SITE DISCOVERY")
    print("=" * 60)

    print()
    print(f"Starting URL: {BASE_URL}")
    print()

    pages, pdfs = discover_site()

    save_inventory(
        pages,
        pdfs
    )

    print()
    print("=" * 60)
    print("DISCOVERY COMPLETED")
    print("=" * 60)

    print(f"Pages discovered: {len(pages)}")
    print(f"PDFs discovered: {len(pdfs)}")

    print()
    print(f"Inventory saved to:")
    print(OUTPUT_FILE)

    print()
    print("PDF FILES")
    print("-" * 60)

    for pdf in pdfs:
        print(pdf)


if __name__ == "__main__":
    main()