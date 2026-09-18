import os

from dotenv import load_dotenv


load_dotenv()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = "stvincent-school-rag"

PINECONE_NAMESPACE = "school"

EMBEDDING_DIMENSION = 3072

PINECONE_CLOUD = "aws"

PINECONE_REGION = "us-east-1"