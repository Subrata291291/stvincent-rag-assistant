import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.json"

OPENROUTER_MODEL = "google/gemini-embedding-001"
GEMINI_MODEL = "models/gemini-embedding-001"
OPENAI_MODEL = "text-embedding-3-large"

EMBEDDING_DIMENSION = 3072
SAVE_EVERY = 10


def load_chunks():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_existing_embeddings():
    if not OUTPUT_FILE.exists():
        return {}

    try:
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)

        return {
            record["chunk_id"]: record
            for record in records
            if record.get("chunk_id")
        }

    except (json.JSONDecodeError, KeyError):
        return {}


def save_embeddings(records):
    ordered_records = list(records.values())

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            ordered_records,
            file,
            indent=4,
            ensure_ascii=False
        )


def remove_stale_embeddings(existing, chunks):
    current_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    stale_ids = [
        chunk_id
        for chunk_id in existing
        if chunk_id not in current_ids
    ]

    for chunk_id in stale_ids:
        del existing[chunk_id]

    return stale_ids


def create_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def create_gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_MODEL,
        google_api_key=api_key
    )


def create_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key
    )


def embed_with_openrouter(client, text):
    response = client.embeddings.create(
        model=OPENROUTER_MODEL,
        input=text
    )

    vector = response.data[0].embedding

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"OpenRouter returned {len(vector)} dimensions"
        )

    return vector


def embed_with_gemini(model, text):
    vector = model.embed_query(text)

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Gemini returned {len(vector)} dimensions"
        )

    return vector


def embed_with_openai(client, text):
    response = client.embeddings.create(
        model=OPENAI_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSION
    )

    vector = response.data[0].embedding

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"OpenAI returned {len(vector)} dimensions"
        )

    return vector


def select_provider():
    openrouter = create_openrouter_client()

    if openrouter:
        print("Trying provider: OpenRouter")

        try:
            vector = embed_with_openrouter(
                openrouter,
                "embedding provider test"
            )

            print("OpenRouter: AVAILABLE")

            return {
                "name": "openrouter",
                "client": openrouter,
                "model": OPENROUTER_MODEL,
                "test_dimension": len(vector)
            }

        except Exception as error:
            print(f"OpenRouter failed: {error}")

    gemini = create_gemini_model()

    if gemini:
        print("Trying provider: Gemini")

        try:
            vector = embed_with_gemini(
                gemini,
                "embedding provider test"
            )

            print("Gemini: AVAILABLE")

            return {
                "name": "gemini",
                "client": gemini,
                "model": GEMINI_MODEL,
                "test_dimension": len(vector)
            }

        except Exception as error:
            print(f"Gemini failed: {error}")

    openai = create_openai_client()

    if openai:
        print("Trying provider: OpenAI")

        try:
            vector = embed_with_openai(
                openai,
                "embedding provider test"
            )

            print("OpenAI: AVAILABLE")

            return {
                "name": "openai",
                "client": openai,
                "model": OPENAI_MODEL,
                "test_dimension": len(vector)
            }

        except Exception as error:
            print(f"OpenAI failed: {error}")

    return None


def embed_text(provider, text):
    if provider["name"] == "openrouter":
        return embed_with_openrouter(
            provider["client"],
            text
        )

    if provider["name"] == "gemini":
        return embed_with_gemini(
            provider["client"],
            text
        )

    if provider["name"] == "openai":
        return embed_with_openai(
            provider["client"],
            text
        )

    raise ValueError(
        f"Unknown provider: {provider['name']}"
    )


def create_record(chunk, vector, provider):
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "chunk_index": chunk["chunk_index"],
        "title": chunk["title"],
        "content": chunk["content"],
        "source": chunk["source"],
        "source_type": chunk["source_type"],
        "embedding": vector,
        "embedding_provider": provider["name"],
        "embedding_model": provider["model"]
    }


def create_embeddings(chunks):
    existing = load_existing_embeddings()

    print(
        f"Existing embeddings before cleanup: "
        f"{len(existing)}"
    )

    stale_ids = remove_stale_embeddings(
        existing,
        chunks
    )

    print(
        f"Stale embeddings removed: "
        f"{len(stale_ids)}"
    )

    save_embeddings(existing)

    print(
        f"Current embeddings after cleanup: "
        f"{len(existing)}"
    )

    if len(existing) == len(chunks):
        return existing

    provider = select_provider()

    if provider is None:
        raise RuntimeError(
            "All embedding providers are unavailable."
        )

    print()
    print(
        f"Selected provider: "
        f"{provider['name']}"
    )
    print(
        f"Embedding model: "
        f"{provider['model']}"
    )
    print(
        f"Embedding dimension: "
        f"{provider['test_dimension']}"
    )
    print()

    for index, chunk in enumerate(
        chunks,
        start=1
    ):
        chunk_id = chunk["chunk_id"]

        if chunk_id in existing:
            print(
                f"[{index}/{len(chunks)}] "
                f"{chunk_id} - SKIPPED"
            )
            continue

        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk_id}"
        )

        try:
            vector = embed_text(
                provider,
                chunk["content"]
            )

        except Exception as error:
            save_embeddings(existing)

            print()
            print(
                f"Provider failed: "
                f"{provider['name']}"
            )
            print(f"Error: {error}")
            print(
                f"Progress saved: "
                f"{len(existing)} embeddings"
            )

            raise

        existing[chunk_id] = create_record(
            chunk,
            vector,
            provider
        )

        if len(existing) % SAVE_EVERY == 0:
            save_embeddings(existing)

            print(
                f"Progress saved: "
                f"{len(existing)} embeddings"
            )

    save_embeddings(existing)

    return existing


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    print("=" * 60)
    print("ST. VINCENT - EMBEDDING CREATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    chunks = load_chunks()

    print(
        f"\nChunks loaded: "
        f"{len(chunks)}"
    )

    print("\nFallback order:")
    print("1. OpenRouter")
    print("2. Gemini")
    print("3. OpenAI")

    print(
        f"\nTarget dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    records = create_embeddings(chunks)

    print()
    print("=" * 60)
    print("EMBEDDING CREATION COMPLETED")
    print("=" * 60)

    print(
        f"\nTotal embeddings: "
        f"{len(records)}"
    )

    providers = {}

    for record in records.values():
        provider = record.get(
            "embedding_provider",
            "unknown"
        )

        providers[provider] = (
            providers.get(provider, 0) + 1
        )

    for provider, count in providers.items():
        print(
            f"{provider}: {count}"
        )

    print(
        f"\nSaved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()