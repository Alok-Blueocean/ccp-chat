import os
import re
import asyncio
import pandas as pd

from langdetect import detect
from typing import Any, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.documents import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings



from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

from indexing_pipeline.index_qdrant import QdrantClintManager
from app.core.configs import get_settings


# =========================================================
# SETTINGS
# =========================================================

settings = get_settings()

os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key


# =========================================================
# QDRANT
# =========================================================

qdrant_manager = QdrantClintManager()

COLLECTION_NAME = qdrant_manager.PARENT_COLLECTION


# =========================================================
# CLEANING
# =========================================================

def fix_encoding(text: str) -> str:
    """
    Fix corrupted UTF-8 text like:
    à¤­à¤•à¥à¤¤
    """

    if not isinstance(text, str):
        return ""

    try:
        text = text.encode("latin1").decode("utf-8")
    except Exception:
        pass

    return text


def clean_text(text: str) -> str:

    text = fix_encoding(text)

    text = text.strip()

    text = text.encode(
        "utf-8",
        "ignore",
    ).decode("utf-8")

    text = re.sub(r"\s+", " ", text)

    return text


def is_english(text: str) -> bool:

    try:

        text = clean_text(text)

        if len(text.split()) < 20:
            return False

        return detect(text) == "en"

    except Exception:
        return False


# =========================================================
# MODELS
# =========================================================

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    temperature=0,
)


# =========================================================
# FALLBACK MODEL
# =========================================================

class FallbackChatModel(BaseChatModel):

    primary_llm: BaseChatModel
    fallback_llm: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return "fallback-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:

        try:

            print("🚀 Using Gemini")

            return self.primary_llm._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        except Exception as e:

            print(f"❌ Gemini failed: {e}")
            print("🔁 Falling back to GPT-4o-mini")

            return self.fallback_llm._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )


# =========================================================
# FINAL LLM
# =========================================================

llm = FallbackChatModel(
    primary_llm=gemini_llm,
    fallback_llm=openai_llm,
)

# Or directly:
llm = openai_llm


# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key,
)


# =========================================================
# RUN CONFIG
# =========================================================

run_config = RunConfig(
    max_workers=1,
    max_retries=1,
    max_wait=30,
    timeout=60,
)


# =========================================================
# FETCH DOCUMENTS
# =========================================================

def fetch_chunks_from_qdrant(
    collection_name: str,
    limit: int = 100,
):

    print(f"📥 Fetching chunks from: {collection_name}")

    records, _ = qdrant_manager.client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    documents = []

    for record in records:

        payload = record.payload or {}

        chunk_text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("page_content")
            or ""
        )

        chunk_text = clean_text(chunk_text)

        # Skip tiny chunks
        if len(chunk_text.split()) < 50:
            continue

        # Skip non-English
        if not is_english(chunk_text):
            print("❌ Skipping non-English chunk")
            continue

        document = Document(
            page_content=chunk_text,
            metadata={
                "document_id": payload.get(
                    "document_id",
                    "unknown",
                ),
                "filename": payload.get(
                    "filename",
                    "unknown",
                ),
                "source": payload.get(
                    "source",
                    "unknown",
                ),
            },
        )

        documents.append(document)

    print(f"✅ Loaded {len(documents)} chunks")

    return documents


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    split_docs = splitter.split_documents(documents)

    print(f"✂️ Split into {len(split_docs)} chunks")

    return split_docs


# =========================================================
# GENERATE TESTSET
# =========================================================

async def generate_testset():

    # -----------------------------------------------------
    # Fetch
    # -----------------------------------------------------

    documents = fetch_chunks_from_qdrant(
        collection_name=COLLECTION_NAME,
        limit=100,
    )

    if not documents:
        print("❌ No documents found")
        return

    # -----------------------------------------------------
    # Split docs
    # -----------------------------------------------------

    split_docs = split_documents(documents)

    # -----------------------------------------------------
    # Generator
    # -----------------------------------------------------

    print("🧠 Initializing Ragas Generator")

    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    print("🚀 Generating Testset")

    testset = generator.generate_with_langchain_docs(
        documents=split_docs,
        testset_size=100,
        run_config=run_config,
        with_debugging_logs=True,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    df = testset.to_pandas()

    output_file = "qdrant_ragas_testset.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(f"🎉 Saved: {output_file}")

    print(df.head())


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    asyncio.run(generate_testset())
    