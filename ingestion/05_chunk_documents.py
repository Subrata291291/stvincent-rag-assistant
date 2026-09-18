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

    return (
        f"{prefix}-{safe_source}-{content_hash}"
    )


def remove_exact_duplicates(documents):
    unique_documents = []
    seen_hashes = set()

    duplicates = 0

    for document in documents:

        content_hash = create_content_hash(
            document
        )

        if content_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(content_hash)
        unique_documents.append(document)

    return unique_documents, duplicates


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

        text = document.get(
            "content",
            ""
        ).strip()

        if not text:
            continue

        document_id = create_document_id(
            document
        )

        document_chunks = splitter.split_text(
            text
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):

            chunks.append({
                "chunk_id": (
                    f"{document_id}"
                    f"-chunk-{chunk_index}"
                ),
                "document_id": document_id,
                "chunk_index": chunk_index,
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

    print(
        f"\nChunks created: "
        f"{len(chunks)}"
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

    print("\nSAMPLE CHUNKS")
    print("-" * 60)

    for chunk in chunks[:3]:

        print(
            f"\nID: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Document ID: "
            f"{chunk['document_id']}"
        )

        print(
            f"Title: "
            f"{chunk['title']}"
        )

        print(
            f"Source: "
            f"{chunk['source']}"
        )

        print(
            f"Text: "
            f"{chunk['content'][:300]}"
        )


if __name__ == "__main__":
    main()