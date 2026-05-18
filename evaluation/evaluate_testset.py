"""
Evaluate retrieval and LLM answers against the Ragas-generated test set.

For each row in qdrant_ragas_testset.csv the script:
  1. Retrieves chunks via the production retrieval pipeline
  2. Builds numbered context and generates a chat answer
  3. Scores retrieval + generation with Ragas metrics

Usage:
  python evaluation/evaluate_testset.py
  python evaluation/evaluate_testset.py --limit 5
  python evaluation/evaluate_testset.py --skip-ragas

Outputs (under evaluation/results/):
  *_predictions.csv       — retrieval + answers (no Ragas)
  *_results.csv           — full row + per-case Ragas columns
  *_metrics_per_case.csv  — one row per test case, scores only (easy to scan)
  *_summary.json          — means/min/max + per_case[] + low_outliers[]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.configs import get_settings
from app.factories.retrieval_factory import get_retrieval_pipeline
from app.services.chat_context import build_numbered_chat_context, ordered_parent_slots
from app.services.llm_service import llm_service

DEFAULT_TESTSET = ROOT / "qdrant_ragas_testset.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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
SCORE_COLS = LEXICAL_METRIC_COLS + RAGAS_METRIC_COLS
META_COLS = ["persona_name", "query_style", "query_length", "synthesizer_name"]


def parse_reference_contexts(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [text]


def _normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_context_recall(retrieved: list[str], reference_contexts: list[str]) -> float:
    if not reference_contexts:
        return float("nan")
    ref_words = _normalize_words(" ".join(reference_contexts))
    if not ref_words:
        return float("nan")
    ret_words = _normalize_words(" ".join(retrieved))
    return len(ref_words & ret_words) / len(ref_words)


def reference_context_hit(retrieved: list[str], reference_contexts: list[str]) -> bool:
    if not retrieved or not reference_contexts:
        return False
    combined = " ".join(retrieved).lower()
    for ref_ctx in reference_contexts:
        ref_words = _normalize_words(ref_ctx)
        if len(ref_words) >= 8:
            ret_words = _normalize_words(combined)
            if len(ref_words & ret_words) / len(ref_words) >= 0.45:
                return True
        snippet = ref_ctx[:250].strip().lower()
        if len(snippet) >= 40 and snippet in combined:
            return True
    return False


def get_context_texts_from_nodes(nodes, pipeline) -> list[str]:
    slots = ordered_parent_slots(nodes)
    parent_ids = [pid for pid, _ in slots if pid]
    parent_payloads = pipeline.retriever.qdrant.retrieve_parents_by_ids(parent_ids)

    texts: list[str] = []
    for parent_id, nws in slots:
        meta = nws.node.metadata or {}
        payload = parent_payloads.get(parent_id) if parent_id else None
        if parent_id and payload:
            text = payload.get("text") or ""
        else:
            text = nws.node.text or ""
        text = text.strip()
        if text:
            texts.append(text)
    return texts


def run_case(
    query: str,
    top_k: int,
    pipeline,
) -> dict:
    nodes = pipeline.retrieve(query=query, top_k=top_k)
    if not nodes:
        return {
            "retrieved_contexts": [],
            "retrieved_scores": [],
            "retrieved_node_ids": [],
            "llm_context": "",
            "generated_answer": "",
            "retrieval_ok": False,
        }

    slots = ordered_parent_slots(nodes)
    parent_ids = [pid for pid, _ in slots if pid]
    parent_payloads = pipeline.retriever.qdrant.retrieve_parents_by_ids(parent_ids)
    llm_context, _refs = build_numbered_chat_context(nodes, parent_payloads)
    retrieved_contexts = get_context_texts_from_nodes(nodes, pipeline)

    answer = ""
    if llm_context:
        answer = llm_service.generate_chat_answer(query, llm_context)

    return {
        "retrieved_contexts": retrieved_contexts,
        "retrieved_scores": [float(n.score) for n in nodes],
        "retrieved_node_ids": [n.node.node_id for n in nodes],
        "llm_context": llm_context,
        "generated_answer": answer,
        "retrieval_ok": bool(retrieved_contexts),
    }


def build_ragas_dataset(rows: list[dict]) -> Dataset:
    records = []
    for row in rows:
        records.append(
            {
                "user_input": row["user_input"],
                "response": row["generated_answer"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["reference"],
                "reference_contexts": row["reference_contexts"],
            }
        )
    return Dataset.from_list(records)


def run_ragas_evaluation(dataset: Dataset):
    from openai import AsyncOpenAI
    from ragas import evaluate
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.openai_model, client=client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client,
        interface="modern",
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
        llm=llm,
        embeddings=embeddings,
    )


def _metric_stats(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "min": None, "max": None, "std": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "count": int(len(values)),
    }


def _find_low_outliers(
    df: pd.DataFrame,
    col: str,
    *,
    z_threshold: float = 1.5,
    absolute_floor: float = 0.35,
) -> list[dict]:
    """Cases unusually low vs batch mean or below an absolute floor."""
    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return []

    mean = float(valid.mean())
    std = float(valid.std()) if len(valid) > 1 else 0.0
    cutoff = mean - z_threshold * std if std > 0 else mean

    outliers: list[dict] = []
    for idx, row in df.iterrows():
        val = values.loc[idx]
        if pd.isna(val):
            continue
        reasons: list[str] = []
        if std > 0 and float(val) < cutoff:
            reasons.append(f"below mean−{z_threshold}σ ({cutoff:.3f})")
        if float(val) < absolute_floor:
            reasons.append(f"below {absolute_floor}")
        if reasons:
            outliers.append(
                {
                    "row": int(idx),
                    "user_input": str(row["user_input"])[:120],
                    "metric": col,
                    "value": float(val),
                    "reason": "; ".join(reasons),
                }
            )
    return outliers


def build_scores_summary(df: pd.DataFrame, score_cols: list[str]) -> dict:
    """Aggregate stats plus per-row scores for inspection."""
    per_case: list[dict] = []
    for idx, row in df.iterrows():
        entry: dict = {"row": int(idx), "user_input": str(row["user_input"])}
        for col in META_COLS:
            if col in row and pd.notna(row[col]):
                entry[col] = row[col]
        for col in score_cols:
            if col not in df.columns:
                continue
            val = row[col]
            if col == "reference_context_hit":
                entry[col] = bool(val) if pd.notna(val) else None
            else:
                entry[col] = (
                    None if pd.isna(val) else float(val)
                )
        per_case.append(entry)

    by_metric: dict[str, dict] = {}
    outliers: list[dict] = []
    for col in score_cols:
        if col not in df.columns:
            continue
        by_metric[col] = _metric_stats(df[col])
        if col != "reference_context_hit":
            outliers.extend(_find_low_outliers(df, col))

    return {
        "per_case": per_case,
        "by_metric": by_metric,
        "low_outliers": outliers,
    }


def print_per_case_scores(df: pd.DataFrame, score_cols: list[str]) -> None:
    """Print a compact per-row score table to the console."""
    display_cols = ["user_input", *[c for c in score_cols if c in df.columns]]
    view = df[display_cols].copy()
    view["user_input"] = view["user_input"].astype(str).str.slice(0, 72)

    for col in score_cols:
        if col not in view.columns:
            continue
        if col == "reference_context_hit":
            view[col] = view[col].map(
                lambda x: "Y" if x is True or x == 1 else ("N" if pd.notna(x) else "-")
            )
        else:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )

    print("\nPer-case scores (full detail in *_metrics_per_case.csv):")
    print(view.to_string(index=True))


def merge_ragas_scores(
    predictions_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    ragas_row_count: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach Ragas columns row-by-row in evaluation order."""
    metric_cols = [
        c
        for c in ragas_df.columns
        if c
        not in {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
            "reference_contexts",
        }
    ]

    merged = predictions_df.copy()
    for col in metric_cols:
        merged[col] = None

    ragas_idx = 0
    for i, row in merged.iterrows():
        if not row["retrieval_ok"] or not str(row["generated_answer"]).strip():
            continue
        if ragas_idx >= ragas_row_count:
            break
        for col in metric_cols:
            merged.at[i, col] = ragas_df.iloc[ragas_idx][col]
        ragas_idx += 1

    return merged, metric_cols


def load_testset(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_input", "reference_contexts", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test set missing columns: {sorted(missing)}")
    if limit is not None:
        df = df.head(limit)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and LLM answers on the Ragas test set.",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=DEFAULT_TESTSET,
        help=f"Path to CSV test set (default: {DEFAULT_TESTSET})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for smoke tests)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run retrieval + generation only; skip Ragas scoring",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for result files (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    if not args.testset.exists():
        raise FileNotFoundError(f"Test set not found: {args.testset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_prefix = args.output_dir / f"eval_{stamp}"

    df = load_testset(args.testset, args.limit)
    pipeline = get_retrieval_pipeline()

    print(f"Loaded {len(df)} test cases from {args.testset}")
    print(f"top_k={args.top_k}")

    rows: list[dict] = []
    for idx, record in df.iterrows():
        query = str(record["user_input"])
        reference = str(record["reference"])
        reference_contexts = parse_reference_contexts(record["reference_contexts"])

        print(f"[{idx + 1}/{len(df)}] {query[:80]}...")
        case = run_case(query, args.top_k, pipeline)

        row = {
            "user_input": query,
            "reference": reference,
            "reference_contexts": reference_contexts,
            "generated_answer": case["generated_answer"],
            "retrieved_contexts": case["retrieved_contexts"],
            "retrieved_scores": case["retrieved_scores"],
            "retrieved_node_ids": case["retrieved_node_ids"],
            "retrieval_ok": case["retrieval_ok"],
            "lexical_context_recall": lexical_context_recall(
                case["retrieved_contexts"],
                reference_contexts,
            ),
            "reference_context_hit": reference_context_hit(
                case["retrieved_contexts"],
                reference_contexts,
            ),
        }
        for col in ("persona_name", "query_style", "query_length", "synthesizer_name"):
            if col in record:
                row[col] = record[col]
        rows.append(row)

    predictions_df = pd.DataFrame(rows)
    predictions_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")

    lexical_summary = {
        "cases": len(rows),
        "retrieval_ok_rate": float(predictions_df["retrieval_ok"].mean()),
        "reference_context_hit_rate": float(predictions_df["reference_context_hit"].mean()),
        "mean_lexical_context_recall": float(predictions_df["lexical_context_recall"].mean(skipna=True)),
    }
    print("\nLexical retrieval summary:")
    for key, value in lexical_summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    summary: dict = {"lexical": lexical_summary}

    score_cols_present = [c for c in LEXICAL_METRIC_COLS if c in predictions_df.columns]
    scores_summary = build_scores_summary(predictions_df, score_cols_present)
    summary["scores"] = scores_summary

    metrics_cols = ["user_input", *[c for c in META_COLS if c in predictions_df.columns], *score_cols_present]
    metrics_path = output_prefix.with_name(output_prefix.name + "_metrics_per_case.csv")
    predictions_df[metrics_cols].to_csv(metrics_path, index=False)
    print_per_case_scores(predictions_df, score_cols_present)

    if args.skip_ragas:
        summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nSkipped Ragas (--skip-ragas). Per-case metrics: {metrics_path}")
        print(f"Summary: {summary_path}")
        return

    ragas_rows = [r for r in rows if r["retrieval_ok"] and r["generated_answer"].strip()]
    if not ragas_rows:
        print("\nNo successful retrieval+answer rows; skipping Ragas evaluation.")
        summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    print(f"\nRunning Ragas on {len(ragas_rows)} cases...")
    ragas_dataset = build_ragas_dataset(ragas_rows)
    ragas_result = run_ragas_evaluation(ragas_dataset)
    ragas_df = ragas_result.to_pandas()

    merged, ragas_metric_cols = merge_ragas_scores(
        predictions_df,
        ragas_df,
        len(ragas_rows),
    )

    score_cols_present = [c for c in SCORE_COLS if c in merged.columns]
    scores_summary = build_scores_summary(merged, score_cols_present)
    summary["scores"] = scores_summary

    metrics_cols = ["user_input", *[c for c in META_COLS if c in merged.columns], *score_cols_present]
    metrics_path = output_prefix.with_name(output_prefix.name + "_metrics_per_case.csv")
    merged[metrics_cols].to_csv(metrics_path, index=False)

    # Back-compat: flat means in summary.ragas
    summary["ragas"] = {
        col: scores_summary["by_metric"].get(col, {}).get("mean")
        for col in ragas_metric_cols
    }

    results_path = output_prefix.with_name(output_prefix.name + "_results.csv")
    merged.to_csv(results_path, index=False)

    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print_per_case_scores(merged, score_cols_present)

    print("\nMetric aggregates (mean / min / max):")
    for col, stats in scores_summary["by_metric"].items():
        if stats["count"] == 0:
            print(f"  {col}: (no values)")
            continue
        print(
            f"  {col}: mean={stats['mean']:.4f}  "
            f"min={stats['min']:.4f}  max={stats['max']:.4f}  n={stats['count']}"
        )

    if scores_summary["low_outliers"]:
        print("\nPossible low outliers:")
        for item in scores_summary["low_outliers"]:
            print(
                f"  row {item['row']} | {item['metric']}={item['value']:.3f} | "
                f"{item['reason']} | {item['user_input']}"
            )

    print(f"\nSaved full results: {results_path}")
    print(f"Saved per-case metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
