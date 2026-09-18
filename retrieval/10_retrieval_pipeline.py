import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
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

RETRIEVAL_TOP_K = 10
FINAL_TOP_K = 5

VECTOR_WEIGHT = 0.80
BM25_WEIGHT = 0.20


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

    return text.lower().split()


def normalize_scores(scores):

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:

        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def calculate_bm25_scores(query, matches):

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

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(query)

    return bm25.get_scores(
        query_tokens
    ).tolist()


def hybrid_rerank(query, matches):

    if not matches:
        return []

    vector_scores = [
        float(match.score)
        for match in matches
    ]

    bm25_scores = calculate_bm25_scores(
        query,
        matches
    )

    normalized_vector = normalize_scores(
        vector_scores
    )

    normalized_bm25 = normalize_scores(
        bm25_scores
    )

    results = []

    for index, match in enumerate(matches):

        metadata = match.metadata or {}

        hybrid_score = (
            VECTOR_WEIGHT
            * normalized_vector[index]
            +
            BM25_WEIGHT
            * normalized_bm25[index]
        )

        results.append(
            {
                "id": match.id,
                "vector_score": vector_scores[index],
                "bm25_score": bm25_scores[index],
                "hybrid_score": hybrid_score,
                "title": metadata.get(
                    "title",
                    ""
                ),
                "source": metadata.get(
                    "source",
                    ""
                ),
                "source_type": metadata.get(
                    "source_type",
                    ""
                ),
                "content": metadata.get(
                    "content",
                    ""
                )
            }
        )

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results[:FINAL_TOP_K]


def build_context(results):

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"[Source {index}]\n"
            f"Title: {result['title']}\n"
            f"Source: {result['source']}\n"
            f"Content:\n{result['content']}"
        )

    return "\n\n".join(
        context_parts
    )


def display_results(
    query,
    results
):

    print()
    print("=" * 60)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"Rank: {rank}")
        print(
            f"Hybrid score: "
            f"{result['hybrid_score']:.4f}"
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
            f"Title: "
            f"{result['title']}"
        )
        print(
            f"Source: "
            f"{result['source']}"
        )

        print()
        print(
            result["content"][:600]
        )

    print()
    print("=" * 60)
    print("AI CONTEXT")
    print("=" * 60)
    print()

    print(
        build_context(results)
    )


def main():

    print("=" * 60)
    print("ST. VINCENT - RETRIEVAL PIPELINE")
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

    matches = vector_search(
        query
    )

    print(
        f"Candidates retrieved: "
        f"{len(matches)}"
    )

    results = hybrid_rerank(
        query,
        matches
    )

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()