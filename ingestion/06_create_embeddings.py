import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.json"
)

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_chunks():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_embeddings(chunks):

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    records = []

    for index, chunk in enumerate(chunks):

        print(
            f"[{index + 1}/{len(chunks)}] "
            f"{chunk['chunk_id']}"
        )

        vector = embeddings_model.embed_query(
            chunk["content"]
        )

        records.append(
            {
                "id": chunk["chunk_id"],
                "values": vector,
                "metadata": {
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "source_type": chunk["source_type"],
                    "document_index": chunk["document_index"],
                    "chunk_index": chunk["chunk_index"]
                }
            }
        )

        print(
            f"  Vector dimension: {len(vector)}"
        )

    return records


def save_embeddings(records):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("=" * 60)
    print("ST. VINCENT - EMBEDDING CREATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_FILE}"
        )

    chunks = load_chunks()

    print()
    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print()
    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print()

    records = create_embeddings(
        chunks
    )

    save_embeddings(
        records
    )

    print()
    print("=" * 60)
    print("EMBEDDING CREATION COMPLETED")
    print("=" * 60)

    print(
        f"Embeddings created: {len(records)}"
    )

    if records:
        print(
            f"Vector dimension: "
            f"{len(records[0]['values'])}"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()