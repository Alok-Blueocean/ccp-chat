import os
from dotenv import load_dotenv
from openai import OpenAI
# from rank_bm25 import BM25Okapi

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

CONFIG = {
    "langchain": {
        "api_key": os.getenv("LANGCHAIN_API_KEY"),
        "tracing": os.getenv("LANGCHAIN_TRACING_V2"),
        "project": os.getenv("LANGCHAIN_PROJECT")
    },
    "models": {
        "openai": os.getenv("OPENAI_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "huggingface": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        "openrouter": os.getenv("OPENROUTER_API_KEY")
    },
    "qdrant": {
        "api_key": os.getenv("QDRANT_API_KEY"),
        "url": os.getenv("QDRANT_URL"),
        "collection": os.getenv("QDRANT_COLLECTION_NAME")
    },
    "azure": {
        "speech_key": os.getenv("AZURE_SPEECH_KEY"),
        "region": os.getenv("AZURE_SPEECH_REGION")
    },
    "algolia": {
        "index_name": os.getenv("INDEX_NAME"),
        "app_id": os.getenv("APPLICATION_ID"),
        "search_key": os.getenv("SEARCH_API_KEY"),
        "write_key": os.getenv("WRITE_API_KEY")
    },
}

openai_client = OpenAI()
