import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INVENTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_inventory.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pages"
)

TIMEOUT = 30


def load_inventory():
    with open(
        INVENTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def clean_text(text):
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def extract_main_content(soup):
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
            "form",
            "svg"
        ]
    ):
        tag.decompose()

    selectors = [
        "main",
        "article",
        ".entry-content",
        ".post-content",
        ".page-content",
        ".elementor-widget-theme-post-content",
        ".elementor-location-single"
    ]

    for selector in selectors:
        element = soup.select_one(selector)

        if element:
            text = element.get_text(
                "\n",
                strip=True
            )

            text = clean_text(text)

            if len(text) > 200:
                return text

    body = soup.body

    if not body:
        return ""

    text = body.get_text(
        "\n",
        strip=True
    )

    return clean_text(text)


def scrape_page(client, url):
    response = client.get(url)

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    title = ""

    if soup.title:
        title = clean_text(
            soup.title.get_text()
        )

    content = extract_main_content(soup)

    return {
        "url": url,
        "title": title,
        "content": content,
        "source_type": "webpage",
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat()
    }


def save_page(page):
    slug = page["url"].rstrip("/").split("/")[-1]

    if not slug:
        slug = "home"

    output_file = (
        OUTPUT_DIR
        / f"{slug}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            page,
            file,
            ensure_ascii=False,
            indent=4
        )

    return output_file


def main():
    print("=" * 60)
    print("ST. VINCENT - SCRAPE PAGES")
    print("=" * 60)

    inventory = load_inventory()

    pages = inventory.get(
        "pages",
        []
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Pages found: {len(pages)}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    with httpx.Client(
        headers=headers,
        timeout=TIMEOUT,
        follow_redirects=True
    ) as client:

        for number, url in enumerate(
            pages,
            start=1
        ):
            try:
                page = scrape_page(
                    client,
                    url
                )

                output_file = save_page(page)

                print(
                    f"[{number}/{len(pages)}] "
                    f"{url}"
                )

                print(
                    f"  Content length: "
                    f"{len(page['content'])}"
                )

                print(
                    f"  Saved: "
                    f"{output_file.name}"
                )

            except Exception as error:
                print(
                    f"[ERROR] {url}"
                )

                print(
                    f"  {error}"
                )

    print()
    print(
        "PAGE SCRAPING COMPLETED"
    )


if __name__ == "__main__":
    main()