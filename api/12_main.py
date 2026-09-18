import sys
from pathlib import Path
import importlib.util

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# Load chatbot
CHAT_FILE = (
    PROJECT_ROOT
    / "api"
    / "11_chat.py"
)

spec = importlib.util.spec_from_file_location(
    "chat_module",
    CHAT_FILE
)

chat_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    chat_module
)

retrieve = chat_module.retrieve
generate_answer = chat_module.generate_answer


app = FastAPI(
    title="St. Vincent's Academy AI Assistant",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stvincentcbseburdwan.org",
        "https://www.stvincentcbseburdwan.org",
        "http://localhost",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):

    question: str


class ChatResponse(BaseModel):

    answer: str
    sources: list


def build_sources(results):

    sources = []
    seen = set()

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            ""
        )

        source = metadata.get(
            "source",
            ""
        )

        key = (
            title,
            source
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "title": title,
                "source": source
            }
        )

    return sources


@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "St. Vincent's Academy AI Assistant",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest
):

    question = request.question.strip()

    if not question:

        return ChatResponse(
            answer="Please enter a question.",
            sources=[]
        )

    try:

        results = retrieve(
            question
        )

        answer, provider = generate_answer(
            question,
            results
        )

        sources = build_sources(
            results
        )

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception as error:

        print(
            f"Chat API error: {error}"
        )

        return ChatResponse(
            answer=(
                "The AI assistant is temporarily "
                "unavailable."
            ),
            sources=[]
        )