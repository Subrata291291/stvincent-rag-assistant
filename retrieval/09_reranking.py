import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)

EMBEDDING_MODEL = "models/gemini-embedding-001"

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60
MAX_CHUNKS_PER_SOURCE = 2


def create_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def vector_search(query):
    embedding_model = create_embedding_model()
    query_vector = embedding_model.embed_query(query)

    index = create_pinecone_index()

    results = index.query(
        vector=query_vector,
        top_k=RETRIEVAL_TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True
    )

    return results.get("matches", [])


def tokenize(text):
    return text.lower().split()


def create_bm25_scores(query, matches):
    documents = []

    for match in matches:
        metadata = match.metadata or {}
        content = metadata.get("content", "")
        documents.append(content)

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)
    query_tokens = tokenize(query)

    return bm25.get_scores(query_tokens)


def create_rrf_scores(vector_results, bm25_scores):
    rrf_results = {}

    for rank, match in enumerate(
        vector_results,
        start=1
    ):
        chunk_id = match.id

        rrf_results[chunk_id] = {
            "id": chunk_id,
            "vector_rank": rank,
            "bm25_rank": None,
            "vector_score": float(match.score),
            "bm25_score": 0.0,
            "rrf_score": 1 / (RRF_K + rank),
            "metadata": match.metadata or {}
        }

    bm25_ranked = sorted(
        enumerate(bm25_scores, start=1),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (index, score) in enumerate(
        bm25_ranked,
        start=1
    ):
        match = vector_results[index - 1]
        chunk_id = match.id

        rrf_results[chunk_id]["bm25_rank"] = rank
        rrf_results[chunk_id]["bm25_score"] = float(score)

        rrf_results[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

    return sorted(
        rrf_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )


def get_source_key(item):
    metadata = item["metadata"]

    source = metadata.get("source")

    if source:
        return source

    return metadata.get("title", "unknown")


def apply_source_diversity(reranked_results):
    selected = []
    source_counts = {}

    for item in reranked_results:
        source = get_source_key(item)

        count = source_counts.get(source, 0)

        if count >= MAX_CHUNKS_PER_SOURCE:
            continue

        selected.append(item)
        source_counts[source] = count + 1

        if len(selected) >= FINAL_TOP_K:
            break

    return selected


def display_results(
    vector_results,
    bm25_scores,
    reranked_results
):
    print()
    print("=" * 60)
    print("INITIAL VECTOR SEARCH")
    print("=" * 60)

    for rank, match in enumerate(
        vector_results,
        start=1
    ):
        metadata = match.metadata or {}

        print()
        print(f"Rank: {rank}")
        print(f"Vector score: {match.score}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source: {metadata.get('source', '')}")

    print()
    print("=" * 60)
    print("BM25 RANKING")
    print("=" * 60)

    bm25_ranked = sorted(
        enumerate(bm25_scores, start=1),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (index, score) in enumerate(
        bm25_ranked,
        start=1
    ):
        match = vector_results[index - 1]
        metadata = match.metadata or {}

        print()
        print(f"Rank: {rank}")
        print(f"BM25 score: {score}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source: {metadata.get('source', '')}")

    print()
    print("=" * 60)
    print("FINAL RERANKED RESULTS")
    print("=" * 60)

    for rank, item in enumerate(
        reranked_results,
        start=1
    ):
        metadata = item["metadata"]

        print()
        print(f"Rank: {rank}")
        print(f"RRF score: {item['rrf_score']}")
        print(f"Vector rank: {item['vector_rank']}")
        print(f"BM25 rank: {item['bm25_rank']}")
        print(f"Vector score: {item['vector_score']}")
        print(f"BM25 score: {item['bm25_score']}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source: {metadata.get('source', '')}")
        print()
        print(metadata.get("content", "")[:500])


def main():
    print("=" * 60)
    print("ST. VINCENT - HYBRID RERANKING")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print("Question cannot be empty.")
        return

    print()
    print("Searching Pinecone...")

    vector_results = vector_search(query)

    print(
        f"Candidates retrieved: "
        f"{len(vector_results)}"
    )

    if not vector_results:
        print("No results found.")
        return

    bm25_scores = create_bm25_scores(
        query,
        vector_results
    )

    rrf_results = create_rrf_scores(
        vector_results,
        bm25_scores
    )

    final_results = apply_source_diversity(
        rrf_results
    )

    display_results(
        vector_results,
        bm25_scores,
        final_results
    )


if __name__ == "__main__":
    main()