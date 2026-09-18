import json
from pathlib import Path
from datetime import datetime, timezone

import fitz
import pytesseract
from PIL import Image
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGES_DIR = PROJECT_ROOT / "data" / "pages"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "documents.json"

TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_PATH
    )


def load_page_documents():

    documents = []

    for file in sorted(PAGES_DIR.glob("*.json")):

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:
            page = json.load(f)

        documents.append(
            {
                "title": page.get("title", ""),
                "content": page.get("content", ""),
                "source": page.get("url", ""),
                "source_type": "webpage",
                "processed_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        )

    return documents


def extract_text_with_pypdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text).strip()


def extract_text_with_ocr(pdf_file):

    document = fitz.open(pdf_file)

    text = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [
                pixmap.width,
                pixmap.height
            ],
            pixmap.samples
        )

        page_text = pytesseract.image_to_string(
            image
        )

        if page_text.strip():
            text.append(page_text)

        print(
            f"    OCR page {page_number}: "
            f"{len(page_text)} characters"
        )

    document.close()

    return "\n".join(text).strip()


def extract_pdf_text(pdf_file):

    text = extract_text_with_pypdf(
        pdf_file
    )

    if text:
        print("  Extraction: pypdf")
        return text

    print("  Extraction: OCR")

    return extract_text_with_ocr(
        pdf_file
    )


def load_pdf_documents():

    documents = []

    for pdf in sorted(
        PDF_DIR.glob("*.pdf")
    ):

        print()
        print(f"Processing PDF: {pdf.name}")

        content = extract_pdf_text(pdf)

        if not content:
            print("  WARNING: No text extracted")
            continue

        documents.append(
            {
                "title": pdf.stem,
                "content": content,
                "source": pdf.name,
                "source_type": "pdf",
                "processed_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        )

        print(
            f"  Extracted: "
            f"{len(content)} characters"
        )

    return documents


def main():

    print("=" * 60)
    print("ST. VINCENT - DOCUMENT PROCESSOR")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    page_docs = load_page_documents()
    pdf_docs = load_pdf_documents()

    all_docs = page_docs + pdf_docs

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_docs,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(f"Web pages: {len(page_docs)}")
    print(f"PDFs: {len(pdf_docs)}")
    print(f"Total documents: {len(all_docs)}")
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()