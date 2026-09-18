import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "site_inventory.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pdfs"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    )
}

TIMEOUT = 60


def create_filename(url, index):
    path = urlparse(url).path

    filename = Path(
        unquote(path)
    ).name

    if not filename:
        filename = f"document_{index}.pdf"

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return filename


def download_pdf(client, url, output_file):
    response = client.get(
        url,
        timeout=TIMEOUT,
        follow_redirects=True
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if (
        "application/pdf" not in content_type
        and not url.lower().endswith(".pdf")
    ):
        raise ValueError(
            "URL did not return a PDF"
        )

    output_file.write_bytes(
        response.content
    )


def main():

    print("=" * 60)
    print("ST. VINCENT'S ACADEMY - PDF DOWNLOADER")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        inventory = json.load(file)

    pdfs = inventory.get(
        "pdfs",
        []
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(f"PDFs to download: {len(pdfs)}")
    print()

    successful = 0
    failed = 0

    with httpx.Client(
        headers=HEADERS
    ) as client:

        for index, url in enumerate(
            pdfs,
            start=1
        ):

            filename = create_filename(
                url,
                index
            )

            output_file = (
                OUTPUT_DIR / filename
            )

            print(
                f"[{index}/{len(pdfs)}] "
                f"{filename}"
            )

            try:

                download_pdf(
                    client,
                    url,
                    output_file
                )

                size_kb = (
                    output_file.stat().st_size
                    / 1024
                )

                print(
                    f"  SAVED: {size_kb:.1f} KB"
                )

                successful += 1

            except Exception as error:

                print(
                    f"  ERROR: {error}"
                )

                failed += 1

                if output_file.exists():
                    output_file.unlink()

    print()
    print("=" * 60)
    print("PDF DOWNLOAD COMPLETED")
    print("=" * 60)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()