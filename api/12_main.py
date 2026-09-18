import importlib
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


chat_module = importlib.import_module(
    "api.11_chat"
)

chat = chat_module.chat


app = FastAPI(
    title="St. Vincent's Academy AI Assistant",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):

    question: str


class ChatResponse(BaseModel):

    answer: str
    sources: list
    provider: str | None


@app.get("/")
def root():

    return {
        "name": "St. Vincent's Academy AI Assistant",
        "status": "running"
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
def chat_endpoint(request: ChatRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = chat(
            question
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            provider=result["provider"]
        )

    except Exception as error:

        print(
            f"Chat error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.12_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )