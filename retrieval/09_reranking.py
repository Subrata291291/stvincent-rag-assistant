import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)

OPENROUTER_MODEL = "google/gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

RETRIEVAL_TOP_K = 30
FINAL_TOP_K = 5

MIN_VECTOR_SCORE = 0.55
MAX_VECTOR_GAP = 0.08

VECTOR_WEIGHT = 0.90
BM25_WEIGHT = 0.10

MAX_CHUNKS_PER_SOURCE = 2


def create_embedding_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def create_pinecone_index():
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    return pc.Index(
        PINECONE_INDEX_NAME
    )


def create_query_embedding(query):
    client = create_embedding_client()

    response = client.embeddings.create(
        model=OPENROUTER_MODEL,
        input=query
    )

    vector = response.data[0].embedding

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query embedding has {len(vector)} dimensions. "
            f"Expected {EMBEDDING_DIMENSION}."
        )

    return vector


def vector_search(query):
    query_vector = create_query_embedding(
        query
    )

    index = create_pinecone_index()

    results = index.query(
        vector=query_vector,
        top_k=RETRIEVAL_TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True
    )

    return results.get(
        "matches",
        []
    )


def tokenize(text):
    return text.lower().split()


def create_bm25_scores(
    query,
    matches
):
    documents = []

    for match in matches:
        metadata = match.metadata or {}

        content = metadata.get(
            "content",
            ""
        )

        documents.append(content)

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(query)

    return bm25.get_scores(
        query_tokens
    )


def normalize_bm25_scores(scores):
    if len(scores) == 0:
        return []

    maximum = max(scores)

    if maximum <= 0:
        return [0.0] * len(scores)

    return [
        float(score) / float(maximum)
        for score in scores
    ]


def apply_semantic_filter(
    vector_results
):
    if not vector_results:
        return []

    best_score = float(
        vector_results[0].score
    )

    threshold = max(
        MIN_VECTOR_SCORE,
        best_score - MAX_VECTOR_GAP
    )

    filtered = []

    for match in vector_results:
        score = float(
            match.score
        )

        if score >= threshold:
            filtered.append(match)

    return filtered


def calculate_hybrid_scores(
    vector_results,
    bm25_scores
):
    if not vector_results:
        return []

    normalized_bm25 = normalize_bm25_scores(
        bm25_scores
    )

    vector_scores = [
        float(match.score)
        for match in vector_results
    ]

    minimum_vector = min(
        vector_scores
    )

    maximum_vector = max(
        vector_scores
    )

    vector_range = (
        maximum_vector -
        minimum_vector
    )

    results = []

    for index, match in enumerate(
        vector_results
    ):
        vector_score = float(
            match.score
        )

        if vector_range > 0:
            normalized_vector = (
                vector_score -
                minimum_vector
            ) / vector_range
        else:
            normalized_vector = 1.0

        bm25_score = normalized_bm25[
            index
        ]

        hybrid_score = (
            VECTOR_WEIGHT *
            normalized_vector
            +
            BM25_WEIGHT *
            bm25_score
        )

        results.append({
            "id": match.id,
            "vector_score": vector_score,
            "bm25_score": float(
                bm25_scores[index]
            ),
            "normalized_vector_score":
                normalized_vector,
            "normalized_bm25_score":
                bm25_score,
            "hybrid_score":
                hybrid_score,
            "metadata":
                match.metadata or {}
        })

    return sorted(
        results,
        key=lambda item: item[
            "hybrid_score"
        ],
        reverse=True
    )


def get_source_key(item):
    metadata = item["metadata"]

    source = metadata.get(
        "source"
    )

    if source:
        return source

    return metadata.get(
        "title",
        "unknown"
    )


def apply_source_diversity(
    ranked_results
):
    selected = []
    source_counts = {}

    for item in ranked_results:
        source = get_source_key(
            item
        )

        count = source_counts.get(
            source,
            0
        )

        if count >= MAX_CHUNKS_PER_SOURCE:
            continue

        selected.append(item)

        source_counts[
            source
        ] = count + 1

        if len(selected) >= FINAL_TOP_K:
            break

    return selected


def display_results(
    vector_results,
    bm25_scores,
    filtered_results,
    final_results
):
    print()
    print("=" * 60)
    print("INITIAL VECTOR SEARCH")
    print("=" * 60)

    for rank, match in enumerate(
        vector_results[:10],
        start=1
    ):
        metadata = match.metadata or {}

        print()
        print(f"Rank: {rank}")
        print(
            f"Vector score: "
            f"{match.score}"
        )
        print(
            f"Title: "
            f"{metadata.get('title', '')}"
        )
        print(
            f"Source: "
            f"{metadata.get('source', '')}"
        )

    print()
    print("=" * 60)
    print("SEMANTIC FILTER")
    print("=" * 60)

    if vector_results:
        best_score = float(
            vector_results[0].score
        )

        threshold = max(
            MIN_VECTOR_SCORE,
            best_score - MAX_VECTOR_GAP
        )

        print(
            f"Best vector score: "
            f"{best_score}"
        )

        print(
            f"Minimum allowed score: "
            f"{threshold}"
        )

    print(
        f"Candidates before filter: "
        f"{len(vector_results)}"
    )

    print(
        f"Candidates after filter: "
        f"{len(filtered_results)}"
    )

    for rank, match in enumerate(
        filtered_results,
        start=1
    ):
        metadata = match.metadata or {}

        print()
        print(f"Rank: {rank}")
        print(
            f"Vector score: "
            f"{match.score}"
        )
        print(
            f"Title: "
            f"{metadata.get('title', '')}"
        )
        print(
            f"Source: "
            f"{metadata.get('source', '')}"
        )

    print()
    print("=" * 60)
    print("BM25 + SEMANTIC RERANKING")
    print("=" * 60)

    for rank, item in enumerate(
        final_results,
        start=1
    ):
        metadata = item[
            "metadata"
        ]

        print()
        print(f"Rank: {rank}")
        print(
            f"Hybrid score: "
            f"{item['hybrid_score']}"
        )
        print(
            f"Vector score: "
            f"{item['vector_score']}"
        )
        print(
            f"BM25 score: "
            f"{item['bm25_score']}"
        )
        print(
            f"Title: "
            f"{metadata.get('title', '')}"
        )
        print(
            f"Source: "
            f"{metadata.get('source', '')}"
        )
        print()
        print(
            metadata.get(
                "content",
                ""
            )[:500]
        )


def main():
    print("=" * 60)
    print("ST. VINCENT - SEMANTIC HYBRID RERANKING")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print(
            "Question cannot be empty."
        )
        return

    print()
    print("Creating query embedding...")
    print(
        f"Embedding model: "
        f"{OPENROUTER_MODEL}"
    )
    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    print()
    print("Searching Pinecone...")

    vector_results = vector_search(
        query
    )

    print(
        f"Candidates retrieved: "
        f"{len(vector_results)}"
    )

    if not vector_results:
        print(
            "No results found."
        )
        return

    filtered_results = apply_semantic_filter(
        vector_results
    )

    if not filtered_results:
        print(
            "No semantically relevant "
            "results found."
        )
        return

    bm25_scores = create_bm25_scores(
        query,
        filtered_results
    )

    ranked_results = calculate_hybrid_scores(
        filtered_results,
        bm25_scores
    )

    final_results = apply_source_diversity(
        ranked_results
    )

    display_results(
        vector_results,
        bm25_scores,
        filtered_results,
        final_results
    )


if __name__ == "__main__":
    main()