import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

load_dotenv(PROJECT_ROOT / ".env")


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
        (score - minimum)
        / (maximum - minimum)
        for score in scores
    ]


def calculate_bm25_scores(query, matches):

    documents = []

    for match in matches:

        metadata = match.metadata or {}

        documents.append(
            metadata.get(
                "content",
                ""
            )
        )

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25.get_scores(
        tokenize(query)
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


def extract_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get(
                    "text"
                )

                if text:
                    parts.append(text)

        return "\n".join(parts).strip()

    return str(content).strip()


def create_llm_providers():

    return [
        (
            "groq",
            ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0
            )
        ),
        (
            "openrouter",
            ChatOpenRouter(
                model="openai/gpt-oss-20b",
                temperature=0
            )
        ),
        (
            "gemini",
            ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=0
            )
        ),
        (
            "openai",
            ChatOpenAI(
                model="gpt-5-nano",
                temperature=0
            )
        )
    ]


def create_prompt(question, context):

    return f"""
You are the official AI assistant for St. Vincent's Academy.

Answer the user's question using ONLY the information provided
in the CONTEXT below.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information, say:
   "I could not find this information in the available school documents."
4. Keep the answer clear and concise.
5. When dates, fees, times, requirements, or other specific details
   are available, preserve them accurately.
6. Do not mention internal retrieval, embeddings, Pinecone,
   BM25, or the LLM.
7. Do not claim something is current unless the provided source
   supports it.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


def generate_answer(question, context):

    prompt = create_prompt(
        question,
        context
    )

    providers = create_llm_providers()

    errors = []

    for name, llm in providers:

        print(
            f"Trying provider: {name}"
        )

        try:

            response = llm.invoke(
                prompt
            )

            answer = extract_response_text(
                response
            )

            if answer:

                return {
                    "answer": answer,
                    "provider": name,
                    "error": None
                }

        except Exception as error:

            error_message = str(error)

            errors.append(
                f"{name}: {error_message}"
            )

            print(
                f"{name} error: "
                f"{error_message}"
            )

    return {
        "answer": (
            "The AI assistant is temporarily "
            "unavailable."
        ),
        "provider": None,
        "error": errors
    }


def chat(question):

    matches = vector_search(
        question
    )

    if not matches:

        return {
            "answer": (
                "I could not find this information "
                "in the available school documents."
            ),
            "sources": [],
            "provider": None
        }

    results = hybrid_rerank(
        question,
        matches
    )

    context = build_context(
        results
    )

    llm_result = generate_answer(
        question,
        context
    )

    sources = []

    for result in results:

        sources.append(
            {
                "title": result["title"],
                "source": result["source"],
                "source_type": result["source_type"]
            }
        )

    return {
        "answer": llm_result["answer"],
        "sources": sources,
        "provider": llm_result["provider"]
    }


def main():

    print("=" * 60)
    print("ST. VINCENT - RAG CHAT")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        return

    print()
    print(
        "Retrieving school information..."
    )

    result = chat(
        question
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print()

    print(
        result["answer"]
    )

    print()
    print(
        f"Provider: "
        f"{result['provider']}"
    )

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, source in enumerate(
        result["sources"],
        start=1
    ):

        print()
        print(
            f"{index}. "
            f"{source['title']}"
        )

        print(
            f"   {source['source']}"
        )

        print(
            f"   Type: "
            f"{source['source_type']}"
        )


if __name__ == "__main__":
    main()