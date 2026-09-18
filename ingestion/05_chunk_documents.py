import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_documents():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


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

    for document_index, document in enumerate(
        documents
    ):

        text = document.get(
            "content",
            ""
        ).strip()

        if not text:
            continue

        document_chunks = splitter.split_text(
            text
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "chunk_id": (
                        f"doc-{document_index}"
                        f"-chunk-{chunk_index}"
                    ),
                    "document_index": document_index,
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
                }
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
            f"Documents file not found: {INPUT_FILE}"
        )

    documents = load_documents()

    print()
    print(
        f"Documents loaded: {len(documents)}"
    )

    chunks = create_chunks(
        documents
    )

    save_chunks(
        chunks
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print()
    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print("SAMPLE CHUNKS")
    print("-" * 60)

    for chunk in chunks[:3]:

        print()
        print(
            f"ID: {chunk['chunk_id']}"
        )

        print(
            f"Title: {chunk['title']}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            f"Text: {chunk['content'][:300]}"
        )


if __name__ == "__main__":
    main()