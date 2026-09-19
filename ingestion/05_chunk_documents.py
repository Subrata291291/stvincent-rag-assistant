import json
import hashlib
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_documents():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_content_hash(document):
    content = document.get("content", "").strip()

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def create_document_id(document):
    source = document.get("source", "").strip()
    source_type = document.get(
        "source_type",
        ""
    ).strip().lower()

    prefix = "pdf" if source_type == "pdf" else "web"

    safe_source = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        source
    ).strip("-").lower()

    content_hash = create_content_hash(document)[:12]

    return f"{prefix}-{safe_source}-{content_hash}"


def remove_exact_duplicates(documents):
    unique_documents = []
    seen_hashes = set()
    duplicates = 0

    for document in documents:
        content_hash = create_content_hash(document)

        if content_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(content_hash)
        unique_documents.append(document)

    return unique_documents, duplicates


def extract_contact_block(text):
    if not text:
        return None

    normalized = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    hours_match = re.search(
        r"School\s+Hours\s*:\s*"
        r"(.{1,60}?)"
        r"\s*\|",
        normalized,
        re.IGNORECASE
    )

    phone_match = re.search(
        r"(\(\+91\)\s*[\d\s.-]{8,18})",
        normalized,
        re.IGNORECASE
    )

    email_match = re.search(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        normalized
    )

    address_match = re.search(
        r"St\.?\s*Vincent'?s\s+Academy\s+"
        r"Jotram,\s*Bardhaman,\s*West Bengal\s*713104",
        normalized,
        re.IGNORECASE
    )

    parts = []

    if hours_match:
        hours = hours_match.group(1).strip()
        parts.append(
            f"School Hours: {hours}"
        )

    if phone_match:
        parts.append(
            phone_match.group(1).strip()
        )

    if email_match:
        parts.append(
            email_match.group(0).strip()
        )

    if address_match:
        parts.append(
            address_match.group(0).strip()
        )

    if len(parts) < 2:
        return None

    return "\n".join(parts)


def remove_contact_block(text, contact_block):
    if not contact_block:
        return text

    cleaned = text

    for line in contact_block.splitlines():
        pattern = re.escape(line)

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE
        )

    return cleaned.strip()


def create_chunks_for_document(
    document,
    splitter
):
    text = document.get(
        "content",
        ""
    ).strip()

    if not text:
        return []

    document_id = create_document_id(
        document
    )

    chunks = []

    contact_block = None

    if document.get(
        "source_type",
        ""
    ).lower() == "webpage":
        contact_block = extract_contact_block(
            text
        )

    remaining_text = text

    if contact_block:
        chunks.append({
            "chunk_id": (
                f"{document_id}"
                f"-contact"
            ),
            "document_id": document_id,
            "chunk_index": 0,
            "title": document.get(
                "title",
                ""
            ),
            "content": contact_block,
            "source": document.get(
                "source",
                ""
            ),
            "source_type": document.get(
                "source_type",
                ""
            )
        })

        remaining_text = remove_contact_block(
            text,
            contact_block
        )

    if remaining_text:
        document_chunks = splitter.split_text(
            remaining_text
        )

        start_index = len(chunks)

        for offset, chunk in enumerate(
            document_chunks
        ):
            chunks.append({
                "chunk_id": (
                    f"{document_id}"
                    f"-chunk-"
                    f"{start_index + offset}"
                ),
                "document_id": document_id,
                "chunk_index": (
                    start_index + offset
                ),
                "title": document.get(
                    "title",
                    ""
                ),
                "content": chunk,
                "source": document.get(
                    "source",
                    ""
                ),
                "source_type": document.get(
                    "source_type",
                    ""
                )
            })

    return chunks


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:
        chunks.extend(
            create_chunks_for_document(
                document,
                splitter
            )
        )

    return chunks


def save_chunks(chunks):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():
    print("=" * 60)
    print("ST. VINCENT - DOCUMENT CHUNKING")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    documents = load_documents()

    print(
        f"\nDocuments loaded: "
        f"{len(documents)}"
    )

    documents, duplicate_count = (
        remove_exact_duplicates(
            documents
        )
    )

    print(
        f"Exact duplicates removed: "
        f"{duplicate_count}"
    )

    print(
        f"Unique documents: "
        f"{len(documents)}"
    )

    chunks = create_chunks(
        documents
    )

    save_chunks(chunks)

    contact_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"].endswith(
            "-contact"
        )
    ]

    print(
        f"\nChunks created: "
        f"{len(chunks)}"
    )

    print(
        f"Contact chunks created: "
        f"{len(contact_chunks)}"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_FILE}"
    )

    print("\nCONTACT CHUNKS")
    print("-" * 60)

    for chunk in contact_chunks[:10]:
        print(
            f"\nID: {chunk['chunk_id']}"
        )
        print(
            f"Title: {chunk['title']}"
        )
        print(
            f"Source: {chunk['source']}"
        )
        print(
            f"Text:\n{chunk['content']}"
        )


if __name__ == "__main__":
    main()