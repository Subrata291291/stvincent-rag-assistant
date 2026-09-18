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

RETRIEVAL_TOP_K = 15
FINAL_TOP_K = 5

MIN_VECTOR_SCORE = 0.58

VECTOR_WEIGHT = 0.98
BM25_WEIGHT = 0.02


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


def vector_search(query):

    embedding_model = create_embedding_model()

    query_vector = embedding_model.embed_query(
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

    return [
        word
        for word in text.lower().split()
        if len(word) > 2
    ]


def build_bm25_scores(
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

        documents.append(
            content
        )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    if not tokenized_documents:
        return []

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(query)

    if not query_tokens:
        return [0.0] * len(matches)

    return bm25.get_scores(
        query_tokens
    ).tolist()


def filter_by_vector_score(
    matches
):

    return [
        match
        for match in matches
        if float(match.score) >= MIN_VECTOR_SCORE
    ]


def calculate_scores(
    matches,
    bm25_scores
):

    results = []

    if not matches:
        return []

    max_bm25 = max(
        bm25_scores
    ) if bm25_scores else 0.0

    for index, match in enumerate(matches):

        vector_score = float(
            match.score
        )

        bm25_score = float(
            bm25_scores[index]
        )

        if max_bm25 > 0:

            bm25_normalized = (
                bm25_score / max_bm25
            )

        else:

            bm25_normalized = 0.0

        hybrid_score = (
            VECTOR_WEIGHT * vector_score
            + BM25_WEIGHT * bm25_normalized
        )

        results.append(
            {
                "id": match.id,
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "bm25_normalized": bm25_normalized,
                "hybrid_score": hybrid_score,
                "metadata": match.metadata or {}
            }
        )

    return results


def remove_duplicates(
    results
):

    unique_results = []

    seen = set()

    for result in results:

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            ""
        )

        content = metadata.get(
            "content",
            ""
        ).strip()

        key = (
            source,
            content
        )

        if key in seen:
            continue

        seen.add(key)

        unique_results.append(
            result
        )

    return unique_results


def remove_weak_sources(
    results
):

    if not results:
        return []

    source_groups = {}

    for result in results:

        source = result["metadata"].get(
            "source",
            ""
        )

        source_groups.setdefault(
            source,
            []
        ).append(result)

    strongest_source = max(
        source_groups,
        key=lambda source: max(
            item["vector_score"]
            for item in source_groups[source]
        )
    )

    strongest_score = max(
        item["vector_score"]
        for item in source_groups[strongest_source]
    )

    filtered = []

    for source, items in source_groups.items():

        source_best_score = max(
            item["vector_score"]
            for item in items
        )

        if (
            source == strongest_source
            or
            source_best_score >= strongest_score - 0.03
        ):
            filtered.extend(items)

    return filtered


def retrieve(query):

    matches = vector_search(
        query
    )

    if not matches:
        return []

    matches = filter_by_vector_score(
        matches
    )

    if not matches:
        return []

    bm25_scores = build_bm25_scores(
        query,
        matches
    )

    results = calculate_scores(
        matches,
        bm25_scores
    )

    results.sort(
        key=lambda item: (
            item["vector_score"],
            item["hybrid_score"]
        ),
        reverse=True
    )

    results = remove_duplicates(
        results
    )

    results = remove_weak_sources(
        results
    )

    results.sort(
        key=lambda item: (
            item["vector_score"],
            item["hybrid_score"]
        ),
        reverse=True
    )

    return results[:FINAL_TOP_K]


def display_results(
    query,
    results
):

    print()
    print("=" * 60)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 60)
    print()

    if not results:

        print(
            "NO STRONG EVIDENCE FOUND"
        )

        print()

        print(
            "The available school documents "
            "do not contain sufficiently relevant "
            "information."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(
            f"Rank: {rank}"
        )

        print(
            f"Vector score: "
            f"{result['vector_score']:.4f}"
        )

        print(
            f"BM25 score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Hybrid score: "
            f"{result['hybrid_score']:.4f}"
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

        content = metadata.get(
            "content",
            ""
        )

        print(
            content[:700]
        )

        print()
        print("-" * 60)


def main():

    print("=" * 60)
    print("ST. VINCENT - PRODUCTION RETRIEVAL")
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
    print(
        "Searching Pinecone..."
    )

    results = retrieve(
        query
    )

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()