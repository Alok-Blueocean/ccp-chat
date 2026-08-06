import numpy as np
import pandas as pd
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from datasets import Dataset

# ── 1. SAMPLE DOCUMENTS ──────────────────────────────────────────────────────

docs = [
    Document(page_content="Python is a high-level programming language known for simplicity."),
    Document(page_content="FastAPI is a modern web framework for building APIs with Python."),
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
]

# ── 2. GENERATE EVAL DATASET (synthetic Q&A via RAGAS) ───────────────────────

llm        = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
testset   = generator.generate_with_langchain_docs(docs, testset_size=5)
df        = testset.to_pandas()
# df has: user_input | reference | reference_contexts

# ── 3. SIMULATE RETRIEVAL + ANSWER ───────────────────────────────────────────

def retrieve(query):
    # replace with your real vector search
    return [docs[0].page_content, docs[1].page_content]

def generate_answer(query, contexts):
    context_text = "\n".join(contexts)
    return llm.invoke(f"Answer: {query}\n\nContext: {context_text}").content

rows = []
for _, row in df.iterrows():
    contexts = retrieve(row["user_input"])
    answer   = generate_answer(row["user_input"], contexts)
    rows.append({
        "user_input":          row["user_input"],
        "response":            answer,
        "retrieved_contexts":  contexts,
        "reference":           row["reference"],
    })

# ── 4. EVALUATE WITH RAGAS ───────────────────────────────────────────────────

dataset      = Dataset.from_list(rows)
ragas_result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
results_df   = ragas_result.to_pandas()

# ── 5. FIND OUTLIERS ─────────────────────────────────────────────────────────

for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
    scores = results_df[metric].dropna()
    cutoff = scores.mean() - 1.5 * scores.std()
    outliers = results_df[(results_df[metric] < cutoff) | (results_df[metric] < 0.35)]
    print(f"\n--- Low {metric} ---")
    print(outliers[["user_input", metric]])



import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI

client = OpenAI()

# ── 1. GOLDEN DATASET ────────────────────────────────────────────────────────

GOLDEN_DATASET = [
    {
        "input": "What is FastAPI?",
        "expected_output": "FastAPI is a modern Python web framework for building APIs.",
        "context": [
            "FastAPI is a modern, fast web framework for building APIs with Python.",
            "It is based on standard Python type hints and supports async out of the box.",
        ],
    },
    {
        "input": "What language is FastAPI written in?",
        "expected_output": "FastAPI is written in Python.",
        "context": [
            "FastAPI is a Python framework built on top of Starlette and Pydantic.",
        ],
    },
]

# ── 2. REAL PIPELINE (retrieval + generation) ─────────────────────────────────

def retrieve(query: str) -> list[str]:
    # replace with your actual vector search
    return GOLDEN_DATASET[0]["context"]

def generate(query: str, contexts: list[str]) -> str:
    context_text = "\n".join(contexts)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using only this context:\n{context_text}"},
            {"role": "user",   "content": query},
        ],
    )
    return response.choices[0].message.content

# ── 3. TESTS ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry", GOLDEN_DATASET)
def test_answer_relevancy(entry):
    contexts = retrieve(entry["input"])
    answer   = generate(entry["input"], contexts)

    test_case = LLMTestCase(
        input=entry["input"],
        actual_output=answer,                    # real pipeline answer
        expected_output=entry["expected_output"],
        retrieval_context=contexts,
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")])


@pytest.mark.parametrize("entry", GOLDEN_DATASET)
def test_faithfulness(entry):
    contexts = retrieve(entry["input"])
    answer   = generate(entry["input"], contexts)

    test_case = LLMTestCase(
        input=entry["input"],
        actual_output=answer,
        retrieval_context=contexts,
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")])


def test_hallucination_caught():
    test_case = LLMTestCase(
        input="What year was FastAPI created?",
        actual_output="FastAPI was created in 1995.",     # fabricated — not in context
        context=["FastAPI is a modern Python web framework."],
    )
    # expect metric to FAIL — hallucination should be detected
    try:
        assert_test(test_case, [HallucinationMetric(threshold=0.5, model="gpt-4o-mini")])
        pytest.fail("Hallucination should have been caught")
    except AssertionError:
        pass  # correct — fabricated answer was flagged



import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock

app = FastAPI()

# ── SAMPLE APP ────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(payload: dict):
    if not payload.get("message"):
        return {"error": "message required"}, 400
    return {"reply": f"You said: {payload['message']}"}

@app.get("/health")
def health():
    return {"status": "ok"}

client = TestClient(app)


# ── 1. FASTAPI ENDPOINT TESTS ─────────────────────────────────────────────────

class TestChatEndpoint:

    def test_valid_message_returns_reply(self):
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        assert "reply" in response.json()

    def test_empty_message_returns_error(self):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 400

    def test_missing_body_returns_error(self):
        response = client.post("/chat", json={})
        assert response.status_code == 400

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ── 2. UNIT TESTS (pure logic) ────────────────────────────────────────────────

def chunk_text(text: str, size: int = 100) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]

class TestChunkText:

    def test_splits_correctly(self):
        result = chunk_text("a" * 250, size=100)
        assert len(result) == 3

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_shorter_than_chunk_size(self):
        result = chunk_text("hello", size=100)
        assert result == ["hello"]


# ── 3. INTEGRATION TEST (with mocked external service) ───────────────────────

def call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

class TestLLMIntegration:

    @patch("openai.OpenAI")
    def test_llm_called_with_prompt(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "mocked reply"

        result = call_llm("What is Python?")

        mock_client.chat.completions.create.assert_called_once()
        assert result == "mocked reply"


# ── 4. EDGE CASE TESTS ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_long_message(self):
        long_msg = "a" * 10000
        response = client.post("/chat", json={"message": long_msg})
        assert response.status_code == 200

    def test_special_characters(self):
        response = client.post("/chat", json={"message": "<script>alert('xss')</script>"})
        assert response.status_code == 200

    def test_sql_injection_attempt(self):
        response = client.post("/chat", json={"message": "'; DROP TABLE users;--"})
        assert response.status_code == 200

    @pytest.mark.parametrize("msg", ["hello", "hi there", "what is karma?"])
    def test_multiple_valid_inputs(self, msg):
        response = client.post("/chat", json={"message": msg})
        assert response.status_code == 200
