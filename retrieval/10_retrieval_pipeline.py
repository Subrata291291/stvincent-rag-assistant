import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from dotenv import load_dotenv

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

RETRIEVAL_TOP_K = 50
FINAL_TOP_K = 5

MIN_VECTOR_SCORE = 0.55

VECTOR_WEIGHT = 0.80
BM25_WEIGHT = 0.10
KEYWORD_WEIGHT = 0.10

CONTACT_BOOST = 0.15

MAX_CHUNKS_PER_SOURCE = 2


QUERY_PATTERNS = {
    "address": [
        "address",
        "postal address",
        "location",
        "exact location",
        "where is the school",
        "where is the institution",
        "located"
    ],

    "phone": [
        "phone",
        "phone number",
        "telephone",
        "telephone number",
        "contact number",
        "mobile number",
        "call",
        "contact"
    ],

    "timing": [
        "school timing",
        "school timings",
        "school hours",
        "timing",
        "timings",
        "opening hours",
        "working hours",
        "office hours",
        "school opens",
        "school closes"
    ],

    "admission_fee": [
        "admission fee",
        "admission fees",
        "admission charge",
        "admission charges"
    ],

    "monthly_fee": [
        "monthly fee",
        "monthly fees",
        "school monthly fee",
        "school monthly fees",
        "tuition fee",
        "tuition fees"
    ]
}


QUERY_EXPANSIONS = {
    "address": [
        "postal address",
        "exact location of institution",
        "school address",
        "school location"
    ],

    "phone": [
        "telephone number",
        "contact number",
        "school phone number"
    ],

    "timing": [
        "school hours",
        "school timing",
        "school opening hours",
        "school closing hours",
        "school working hours"
    ],

    "admission_fee": [
        "admission fee",
        "admission fees",
        "admission charges"
    ],

    "monthly_fee": [
        "school monthly fees",
        "monthly fee",
        "tuition fee"
    ]
}


def create_embedding_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


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


def detect_query_intents(query):
    normalized_query = normalize_text(query)

    detected = []

    for intent, patterns in QUERY_PATTERNS.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)

            if normalized_pattern in normalized_query:
                detected.append(intent)
                break

    return detected


def build_search_queries(query):
    intents = detect_query_intents(query)

    queries = [query]

    for intent in intents:
        for expansion in QUERY_EXPANSIONS.get(intent, []):
            if expansion not in queries:
                queries.append(expansion)

    return queries


def vector_search(query, index):
    query_vector = create_query_embedding(query)

    results = index.query(
        vector=query_vector,
        top_k=RETRIEVAL_TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True
    )

    return results.get("matches", [])


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s:/.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return [
        word
        for word in normalize_text(text).split()
        if len(word) > 2
    ]


def keyword_match_score(query, content):
    normalized_content = normalize_text(content)

    if not normalized_content:
        return 0.0

    intents = detect_query_intents(query)

    if not intents:
        return 0.0

    matched_patterns = 0
    total_patterns = 0

    for intent in intents:
        for pattern in QUERY_PATTERNS[intent]:
            normalized_pattern = normalize_text(pattern)

            total_patterns += 1

            if len(normalized_pattern.split()) == 1:
                if re.search(
                    rf"\b{re.escape(normalized_pattern)}\b",
                    normalized_content
                ):
                    matched_patterns += 1
            else:
                if normalized_pattern in normalized_content:
                    matched_patterns += 1

    if total_patterns == 0:
        return 0.0

    return matched_patterns / total_patterns


def build_bm25_scores(query, matches):
    documents = []

    for match in matches:
        metadata = match.metadata or {}

        documents.append(
            metadata.get("content", "")
        )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    if not tokenized_documents:
        return []

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)

    if not query_tokens:
        return [0.0] * len(matches)

    return bm25.get_scores(
        query_tokens
    ).tolist()


def normalize_bm25_scores(scores):
    if not scores:
        return []

    positive_scores = [
        float(score)
        for score in scores
        if float(score) > 0
    ]

    if not positive_scores:
        return [0.0] * len(scores)

    max_score = max(positive_scores)

    if max_score <= 0:
        return [0.0] * len(scores)

    return [
        max(float(score), 0.0) / max_score
        for score in scores
    ]


def filter_by_vector_score(matches):
    return [
        match
        for match in matches
        if float(match.score) >= MIN_VECTOR_SCORE
    ]


def merge_matches(all_matches):
    merged = {}

    for match in all_matches:
        match_id = match.id
        vector_score = float(match.score)

        if match_id not in merged:
            merged[match_id] = {
                "match": match,
                "best_vector_score": vector_score
            }
        else:
            if vector_score > merged[match_id]["best_vector_score"]:
                merged[match_id]["best_vector_score"] = vector_score
                merged[match_id]["match"] = match

    return [
        item["match"]
        for item in merged.values()
    ]


def is_contact_chunk(match):
    return str(match.id).endswith("-contact")


def calculate_scores(query, matches, bm25_scores):
    results = []

    if not matches:
        return []

    normalized_bm25 = normalize_bm25_scores(
        bm25_scores
    )

    intents = detect_query_intents(query)

    for index, match in enumerate(matches):
        metadata = match.metadata or {}

        content = metadata.get(
            "content",
            ""
        )

        vector_score = float(match.score)

        bm25_score = (
            float(bm25_scores[index])
            if index < len(bm25_scores)
            else 0.0
        )

        bm25_normalized = (
            normalized_bm25[index]
            if index < len(normalized_bm25)
            else 0.0
        )

        keyword_score = keyword_match_score(
            query,
            content
        )

        hybrid_score = (
            VECTOR_WEIGHT * vector_score
            + BM25_WEIGHT * bm25_normalized
            + KEYWORD_WEIGHT * keyword_score
        )

        contact_boost = 0.0

        if is_contact_chunk(match):
            if any(
                intent in intents
                for intent in ["address", "phone", "timing"]
            ):
                contact_boost = CONTACT_BOOST

        ranking_score = hybrid_score + contact_boost

        results.append({
            "id": match.id,
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "bm25_normalized": bm25_normalized,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score,
            "contact_boost": contact_boost,
            "ranking_score": ranking_score,
            "metadata": metadata
        })

    return results


def remove_duplicates(results):
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
        unique_results.append(result)

    return unique_results


def apply_source_diversity(results):
    selected = []
    source_counts = {}

    for result in results:
        source = result["metadata"].get(
            "source",
            ""
        )

        count = source_counts.get(
            source,
            0
        )

        if count >= MAX_CHUNKS_PER_SOURCE:
            continue

        selected.append(result)

        source_counts[source] = count + 1

        if len(selected) >= FINAL_TOP_K:
            break

    return selected


def sort_results(results):
    results.sort(
        key=lambda item: (
            item["ranking_score"],
            item["keyword_score"],
            item["vector_score"]
        ),
        reverse=True
    )

    return results


def retrieve(query):
    intents = detect_query_intents(query)

    print(
        f"Detected intents: {intents}"
    )

    search_queries = build_search_queries(
        query
    )

    print()
    print("Search queries:")

    for search_query in search_queries:
        print(
            f"- {search_query}"
        )

    index = create_pinecone_index()

    all_matches = []

    for search_query in search_queries:
        matches = vector_search(
            search_query,
            index
        )

        print()
        print(
            f"Query: {search_query}"
        )

        print(
            f"Candidates: {len(matches)}"
        )

        filtered = filter_by_vector_score(
            matches
        )

        print(
            f"After vector filter: "
            f"{len(filtered)}"
        )

        all_matches.extend(filtered)

    print()
    print(
        f"Total collected candidates: "
        f"{len(all_matches)}"
    )

    matches = merge_matches(
        all_matches
    )

    print(
        f"Unique Pinecone candidates: "
        f"{len(matches)}"
    )

    if not matches:
        return []

    bm25_scores = build_bm25_scores(
        query,
        matches
    )

    results = calculate_scores(
        query,
        matches,
        bm25_scores
    )

    results = sort_results(
        results
    )

    results = remove_duplicates(
        results
    )

    print(
        f"Candidates after duplicate removal: "
        f"{len(results)}"
    )

    results = apply_source_diversity(
        results
    )

    results = sort_results(
        results
    )

    return results[:FINAL_TOP_K]


def display_results(query, results):
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
            f"BM25 normalized: "
            f"{result['bm25_normalized']:.4f}"
        )

        print(
            f"Keyword score: "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"Contact boost: "
            f"{result['contact_boost']:.4f}"
        )

        print(
            f"Hybrid score: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Ranking score: "
            f"{result['ranking_score']:.4f}"
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
            content[:900]
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
    print("Embedding model:")
    print(OPENROUTER_MODEL)

    print()
    print("Embedding dimension:")
    print(EMBEDDING_DIMENSION)

    print()
    print("Searching Pinecone...")

    results = retrieve(query)

    display_results(
        query,
        results
    )


if __name__ == "__main__":
    main()