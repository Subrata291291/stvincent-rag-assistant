import json
import sys
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_REGION
)


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.json"
)

BATCH_SIZE = 50


def load_embeddings():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_index_if_missing(pc):

    index_names = pc.list_indexes().names()

    if PINECONE_INDEX_NAME in index_names:

        print(
            f"Index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )

        return

    print()
    print(
        f"Creating Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

    print(
        "Index creation request submitted."
    )


def wait_for_index(pc):

    print()
    print("Waiting for Pinecone index...")

    while True:

        description = (
            pc.describe_index(
                PINECONE_INDEX_NAME
            )
        )

        if description.status["ready"]:

            print("Index is ready.")

            break


def clear_namespace(index):

    print()
    print(
        f"Clearing namespace: "
        f"{PINECONE_NAMESPACE}"
    )

    index.delete(
        delete_all=True,
        namespace=PINECONE_NAMESPACE
    )

    print(
        "Namespace cleared."
    )


def upload_vectors(index, records):

    total = len(records)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = records[
            start:start + BATCH_SIZE
        ]

        index.upsert(
            vectors=batch,
            namespace=PINECONE_NAMESPACE
        )

        uploaded = min(
            start + BATCH_SIZE,
            total
        )

        print(
            f"Uploaded {uploaded}/{total}"
        )


def main():

    print("=" * 60)
    print("ST. VINCENT - PINECONE FULL SYNC")
    print("=" * 60)

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: "
            f"{INPUT_FILE}"
        )

    records = load_embeddings()

    print()
    print(
        f"Embeddings loaded: {len(records)}"
    )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    create_index_if_missing(
        pc
    )

    wait_for_index(
        pc
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    print()
    print(
        f"Index: {PINECONE_INDEX_NAME}"
    )

    print(
        f"Namespace: {PINECONE_NAMESPACE}"
    )

    clear_namespace(
        index
    )

    print()

    upload_vectors(
        index,
        records
    )

    stats = index.describe_index_stats()

    namespace_stats = (
        stats.namespaces.get(
            PINECONE_NAMESPACE
        )
    )

    print()
    print("=" * 60)
    print("PINECONE FULL SYNC COMPLETED")
    print("=" * 60)

    print(
        f"Namespace: "
        f"{PINECONE_NAMESPACE}"
    )

    print(
        f"Vectors uploaded: "
        f"{len(records)}"
    )

    print(
        f"Namespace stats: "
        f"{namespace_stats}"
    )

    print()
    print(
        f"Total index vectors: "
        f"{stats.total_vector_count}"
    )


if __name__ == "__main__":
    main()