import sys
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)


EMBEDDING_MODEL = "models/gemini-embedding-001"

TOP_K = 5


def create_embedding_model():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


def create_pinecone_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    return pc.Index(
        PINECONE_INDEX_NAME
    )


def search(query):

    embedding_model = create_embedding_model()

    query_vector = (
        embedding_model.embed_query(
            query
        )
    )

    index = create_pinecone_index()

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True
    )

    return results


def display_results(query, results):

    print()
    print("=" * 60)
    print("VECTOR SEARCH RESULTS")
    print("=" * 60)

    print()
    print(f"Query: {query}")

    matches = results.get(
        "matches",
        []
    )

    print(
        f"Results: {len(matches)}"
    )

    print()

    for rank, match in enumerate(
        matches,
        start=1
    ):

        metadata = match.get(
            "metadata",
            {}
        )

        print("-" * 60)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {match.get('score')}"
        )

        print(
            f"ID: {match.get('id')}"
        )

        print(
            f"Title: {metadata.get('title', '')}"
        )

        print(
            f"Source: {metadata.get('source', '')}"
        )

        print()

        print(
            metadata.get(
                "content",
                ""
            )
        )

    print()
    print("=" * 60)


def main():

    print("=" * 60)
    print("ST. VINCENT - VECTOR SEARCH")
    print("=" * 60)

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing"
        )

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print(
            "Question cannot be empty."
        )
        return

    results = search(
        query
    )

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()