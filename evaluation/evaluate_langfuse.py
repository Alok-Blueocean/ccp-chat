"""
Evaluate retrieval and LLM answers against the Ragas-generated test set
WITH Langfuse tracing for A/B testing.

Requirements:
pip install -U \
    langfuse \
    ragas \
    langchain-openai \
    openai \
    pandas \
    datasets

Run:
python evaluation/evaluate_testset.py

python evaluation/evaluate_testset.py \
    --experiment baseline \
    --top-k 2 \
    --limit 10

python evaluation/evaluate_testset.py \
    --experiment reranker_v2 \
    --top-k 5 \
    --limit 20
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import Dataset

# =========================================================
# ROOT
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =========================================================
# SETTINGS
# =========================================================

from app.core.configs import get_settings

settings = get_settings()

# =========================================================
# ENV VARIABLES
# =========================================================

os.environ["OPENAI_API_KEY"] = settings.openai_api_key
os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
os.environ["LANGFUSE_HOST"] = settings.langfuse_host

# =========================================================
# APP IMPORTS
# =========================================================

from app.factories.retrieval_factory import (
    get_retrieval_pipeline,
)

from app.services.chat_context import (
    build_numbered_chat_context,
    ordered_parent_slots,
)

from app.services.llm_service import (
    llm_service,
)

# =========================================================
# LANGFUSE
# =========================================================

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

# =========================================================
# RAGAS
# =========================================================

from ragas import evaluate
from ragas.run_config import RunConfig

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

# =========================================================
# PATHS
# =========================================================

DEFAULT_TESTSET = (
    ROOT / "qdrant_ragas_testset.csv"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

# =========================================================
# METRICS
# =========================================================

LEXICAL_METRIC_COLS = [
    "lexical_context_recall",
    "reference_context_hit",
]

RAGAS_METRIC_COLS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

SCORE_COLS = (
    LEXICAL_METRIC_COLS
    + RAGAS_METRIC_COLS
)

# =========================================================
# HELPERS
# =========================================================

def parse_reference_contexts(
    value,
) -> list[str]:

    if isinstance(value, list):
        return [
            str(x)
            for x in value
            if str(x).strip()
        ]

    if (
        value is None
        or (
            isinstance(value, float)
            and pd.isna(value)
        )
    ):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):

            return [
                str(x)
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    return [text]


def _normalize_words(
    text: str,
) -> set[str]:

    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def lexical_context_recall(
    retrieved: list[str],
    reference_contexts: list[str],
) -> float:

    if not reference_contexts:
        return 0.0

    ref_words = _normalize_words(
        " ".join(reference_contexts)
    )

    ret_words = _normalize_words(
        " ".join(retrieved)
    )

    if not ref_words:
        return 0.0

    return (
        len(ref_words & ret_words)
        / len(ref_words)
    )


def reference_context_hit(
    retrieved: list[str],
    reference_contexts: list[str],
) -> bool:

    if not retrieved:
        return False

    combined = " ".join(retrieved).lower()

    for ref_ctx in reference_contexts:

        ref_words = _normalize_words(ref_ctx)

        if not ref_words:
            continue

        ret_words = _normalize_words(combined)

        overlap = (
            len(ref_words & ret_words)
            / len(ref_words)
        )

        if overlap >= 0.45:
            return True

    return False

# =========================================================
# RETRIEVAL HELPERS
# =========================================================

def get_context_texts_from_nodes(
    nodes,
    pipeline,
) -> list[str]:

    slots = ordered_parent_slots(nodes)

    parent_ids = [
        pid
        for pid, _ in slots
        if pid
    ]

    parent_payloads = (
        pipeline.retriever
        .qdrant
        .retrieve_parents_by_ids(parent_ids)
    )

    texts = []

    for parent_id, nws in slots:

        payload = (
            parent_payloads.get(parent_id)
            if parent_id
            else None
        )

        if payload:
            text = payload.get("text") or ""
        else:
            text = nws.node.text or ""

        text = text.strip()

        # IMPORTANT:
        # Reduce token explosion for Ragas
        text = text[:1500]

        if text:
            texts.append(text)

    return texts

# =========================================================
# SINGLE CASE
# =========================================================

def run_case(
    query: str,
    top_k: int,
    pipeline,
    trace,
):

    retrieval_span = trace.span(
        name="retrieval",
        input={
            "query": query,
            "top_k": top_k,
        },
    )

    nodes = pipeline.retrieve(
        query=query,
        top_k=top_k,
    )

    if not nodes:

        retrieval_span.end(
            output={
                "retrieved": 0,
            }
        )

        return {
            "retrieved_contexts": [],
            "generated_answer": "",
            "retrieval_ok": False,
        }

    slots = ordered_parent_slots(nodes)

    parent_ids = [
        pid
        for pid, _ in slots
        if pid
    ]

    parent_payloads = (
        pipeline.retriever
        .qdrant
        .retrieve_parents_by_ids(parent_ids)
    )

    llm_context, _refs = (
        build_numbered_chat_context(
            nodes,
            parent_payloads,
        )
    )

    retrieved_contexts = (
        get_context_texts_from_nodes(
            nodes,
            pipeline,
        )
    )

    retrieval_span.end(
        output={
            "retrieved_chunks": len(retrieved_contexts),
        }
    )

    generation = trace.generation(
        name="llm_generation",
        model="gpt-4o-mini",
        input=query,
    )

    answer = llm_service.generate_chat_answer(
        query,
        llm_context,
    )

    generation.end(
        output=answer,
    )

    return {
        "retrieved_contexts": retrieved_contexts,
        "generated_answer": answer,
        "retrieval_ok": True,
    }

# =========================================================
# RAGAS DATASET
# =========================================================

def build_ragas_dataset(
    rows: list[dict],
):

    return Dataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["generated_answer"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
                "reference_contexts": r["reference_contexts"],
            }
            for r in rows
        ]
    )

# =========================================================
# RAGAS EVALUATION
# =========================================================

def run_ragas_evaluation(
    dataset: Dataset,
):

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
        max_tokens=2048,
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )

    ragas_llm = (
        LangchainLLMWrapper(
            evaluator_llm
        )
    )

    ragas_embeddings = (
        LangchainEmbeddingsWrapper(
            evaluator_embeddings
        )
    )

    run_config = RunConfig(
        max_workers=1,
        timeout=60,
        max_retries=1,
    )

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testset",
        type=Path,
        default=DEFAULT_TESTSET,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.testset
    )

    if args.limit:
        df = df.head(args.limit)

    pipeline = (
        get_retrieval_pipeline()
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    experiment_trace = langfuse.trace(
        name="rag_evaluation",
        metadata={
            "experiment": args.experiment,
            "top_k": args.top_k,
            "cases": len(df),
        }
    )

    rows = []
    case_traces = []

    print(
        f"\n🚀 Experiment: {args.experiment}"
    )

    print(
        f"📊 Total cases: {len(df)}"
    )

    for idx, record in df.iterrows():

        query = str(
            record["user_input"]
        )

        reference = str(
            record["reference"]
        )

        reference_contexts = (
            parse_reference_contexts(
                record[
                    "reference_contexts"
                ]
            )
        )

        print(
            f"\n[{idx+1}/{len(df)}]"
        )

        print(query[:120])

        case_trace = langfuse.trace(
            name="single_test_case",
            metadata={
                "experiment": args.experiment,
                "index": idx,
            },
            input={
                "query": query,
            },
        )

        case = run_case(
            query=query,
            top_k=args.top_k,
            pipeline=pipeline,
            trace=case_trace,
        )

        row = {
            "user_input": query,
            "reference": reference,
            "reference_contexts": reference_contexts,
            "generated_answer": case["generated_answer"],
            "retrieved_contexts": case["retrieved_contexts"],
            "retrieval_ok": case["retrieval_ok"],
        }

        row["lexical_context_recall"] = (
            lexical_context_recall(
                case["retrieved_contexts"],
                reference_contexts,
            )
        )

        row["reference_context_hit"] = (
            reference_context_hit(
                case["retrieved_contexts"],
                reference_contexts,
            )
        )

        case_trace.score(
            name="lexical_context_recall",
            value=float(
                row["lexical_context_recall"]
            ),
        )

        case_trace.score(
            name="reference_context_hit",
            value=float(
                row["reference_context_hit"]
            ),
        )

        case_traces.append(case_trace)
        rows.append(row)

    predictions_df = pd.DataFrame(rows)

    # =====================================================
    # RAGAS
    # =====================================================

    ragas_rows = [
        r
        for r in rows
        if r["retrieval_ok"]
    ]

    ragas_dataset = (
        build_ragas_dataset(
            ragas_rows
        )
    )

    print(
        "\n🧠 Running Ragas evaluation..."
    )

    ragas_result = (
        run_ragas_evaluation(
            ragas_dataset
        )
    )

    ragas_df = (
        ragas_result.to_pandas()
    )

    # =====================================================
    # MERGE METRICS
    # =====================================================

    ragas_metric_cols = [
        c for c in ragas_df.columns
        if c not in {
            "user_input", "response", "retrieved_contexts",
            "reference", "reference_contexts",
        }
    ]

    for col in ragas_metric_cols:
        predictions_df[col] = None

    ragas_idx = 0
    for i, row_data in enumerate(rows):
        if not row_data["retrieval_ok"]:
            continue
        if ragas_idx >= len(ragas_df):
            break
        for col in ragas_metric_cols:
            val = ragas_df.iloc[ragas_idx][col]
            predictions_df.at[i, col] = val
            if pd.notna(val):
                case_traces[i].score(name=col, value=float(val))
        ragas_idx += 1

    langfuse.flush()

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_file = (
        RESULTS_DIR
        / f"{args.experiment}_{stamp}.csv"
    )

    predictions_df.to_csv(
        output_file,
        index=False,
    )

    # =====================================================
    # FINAL METRICS
    # =====================================================

    print("\n==============================")
    print("📊 FINAL METRICS")
    print("==============================")

    for col in SCORE_COLS:

        if col not in predictions_df.columns:
            continue

        value = pd.to_numeric(
            predictions_df[col],
            errors="coerce",
        ).mean()

        if pd.isna(value):
            continue

        print(
            f"{col}: {value:.4f}"
        )

        experiment_trace.score(
            name=col,
            value=float(value),
        )


    langfuse.flush()

    print(
        f"\n✅ Saved: {output_file}"
    )

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()