import json
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.json"
)

PINECONE_INDEX_NAME = "stvincent-school-rag"
PINECONE_NAMESPACE = "school"

PINECONE_DIMENSION = 3072
PINECONE_METRIC = "cosine"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

BATCH_SIZE = 50


def load_embeddings():
    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def create_pinecone_client():
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = __import__("os").getenv(
        "PINECONE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing."
        )

    return Pinecone(api_key=api_key)


def create_index_if_needed(pc):
    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME in existing_indexes:
        print(
            f"Index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )
        return

    print(
        f"Creating index: "
        f"{PINECONE_INDEX_NAME}"
    )

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )


def wait_for_index(pc):
    print("\nWaiting for Pinecone index...")

    while True:
        description = pc.describe_index(
            PINECONE_INDEX_NAME
        )

        if description.status.ready:
            break

        print("Index not ready. Waiting...")
        time.sleep(2)

    print("Index is ready.")


def get_index(pc):
    return pc.Index(
        PINECONE_INDEX_NAME
    )


def validate_embeddings(records):
    if not records:
        raise ValueError(
            "No embeddings found."
        )

    print(
        f"\nValidating "
        f"{len(records)} embeddings..."
    )

    chunk_ids = set()

    for record in records:

        chunk_id = record.get("chunk_id")
        vector = record.get("embedding")

        if not chunk_id:
            raise ValueError(
                "Embedding record is missing "
                "'chunk_id'."
            )

        if not vector:
            raise ValueError(
                f"Embedding is missing for "
                f"{chunk_id}"
            )

        if len(vector) != PINECONE_DIMENSION:
            raise ValueError(
                f"{chunk_id} has "
                f"{len(vector)} dimensions. "
                f"Expected "
                f"{PINECONE_DIMENSION}."
            )

        if chunk_id in chunk_ids:
            raise ValueError(
                f"Duplicate chunk_id found: "
                f"{chunk_id}"
            )

        chunk_ids.add(chunk_id)

    print("Embedding validation passed.")


def build_pinecone_vectors(records):
    vectors = []

    for record in records:

        metadata = {
            "document_id": record.get(
                "document_id",
                ""
            ),
            "chunk_index": record.get(
                "chunk_index",
                0
            ),
            "title": record.get(
                "title",
                ""
            ),
            "content": record.get(
                "content",
                ""
            ),
            "source": record.get(
                "source",
                ""
            ),
            "source_type": record.get(
                "source_type",
                ""
            ),
            "embedding_provider": record.get(
                "embedding_provider",
                ""
            ),
            "embedding_model": record.get(
                "embedding_model",
                ""
            )
        }

        vectors.append({
            "id": record["chunk_id"],
            "values": record["embedding"],
            "metadata": metadata
        })

    return vectors


def clear_namespace(index):
    print(
        f"\nClearing namespace: "
        f"{PINECONE_NAMESPACE}"
    )

    try:
        index.delete(
            delete_all=True,
            namespace=PINECONE_NAMESPACE
        )

        print("Namespace cleared.")

    except Exception as error:
        error_text = str(error)

        if "Namespace not found" in error_text:
            print(
                "Namespace does not exist yet. "
                "Skipping clear."
            )
        else:
            raise

def upload_vectors(index, vectors):
    total = len(vectors)

    print(
        f"\nUploading "
        f"{total} vectors..."
    )

    uploaded = 0

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = vectors[start:end]

        index.upsert(
            vectors=batch,
            namespace=PINECONE_NAMESPACE
        )

        uploaded += len(batch)

        print(
            f"Uploaded "
            f"{uploaded}/{total}"
        )

    return uploaded


def verify_sync(index):
    print("\nVerifying Pinecone namespace...")

    stats = index.describe_index_stats()

    namespace_stats = (
        stats.namespaces.get(
            PINECONE_NAMESPACE
        )
    )

    if namespace_stats is None:
        print(
            "Namespace not found in stats."
        )
        return

    vector_count = namespace_stats.vector_count

    print(
        f"Namespace: "
        f"{PINECONE_NAMESPACE}"
    )

    print(
        f"Vectors in Pinecone: "
        f"{vector_count}"
    )


def main():

    print("=" * 60)
    print("ST. VINCENT - PINECONE FULL SYNC")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: "
            f"{INPUT_FILE}"
        )

    records = load_embeddings()

    print(
        f"\nEmbeddings loaded: "
        f"{len(records)}"
    )

    validate_embeddings(records)

    pc = create_pinecone_client()

    create_index_if_needed(pc)

    wait_for_index(pc)

    print(
        f"\nIndex: "
        f"{PINECONE_INDEX_NAME}"
    )

    print(
        f"Namespace: "
        f"{PINECONE_NAMESPACE}"
    )

    index = get_index(pc)

    clear_namespace(index)

    vectors = build_pinecone_vectors(
        records
    )

    print(
        f"\nPinecone vectors prepared: "
        f"{len(vectors)}"
    )

    uploaded = upload_vectors(
        index,
        vectors
    )

    print(
        f"\nUploaded vectors: "
        f"{uploaded}"
    )

    verify_sync(index)

    print("\n" + "=" * 60)
    print("PINECONE SYNC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()