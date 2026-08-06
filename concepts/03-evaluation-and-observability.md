# Evaluation & Observability Concepts

How do you know a RAG system is actually good, and how do you catch it when it silently gets worse? This file covers the repo's two-track answer: fast deterministic tests that gate CI, and slower LLM-judged batch/production evaluation that tracks quality and drift over time, all tied together by Langfuse tracing.

## RAGAS Metrics

**What it is**
RAGAS (Retrieval-Augmented Generation Assessment) scores a RAG answer using *other* LLM calls as judges, instead of exact-match string comparison. The repo uses `gpt-4o-mini` as both the judge LLM and (via `text-embedding-3-small`) the judge's embedding model, and evaluates five metrics that split cleanly into "was retrieval good" (`context_precision`, `context_recall`) and "was the generated answer good" (`faithfulness`, `answer_relevancy`, `answer_correctness`). Each metric returns a 0–1 score per row of a `datasets.Dataset`. Because the judge is itself an LLM call, RAGAS scoring is slow, nondeterministic, and costs real tokens — which is exactly why the repo does not run it in CI (see **Two-tier eval strategy** below) and instead runs it as a separate batch job or as a cheap background task.

**How it works — one line per metric**
- **`faithfulness`** — of the individual claims made in the generated answer, what fraction are actually supported by the retrieved context. Penalizes *unsupported/fabricated claims* (hallucination).
- **`answer_relevancy`** — how on-topic the answer is to the question, measured by having the judge LLM generate several questions that the answer *would* be answering, then embedding-comparing those back to the original question. Penalizes incomplete, redundant, or off-topic answers.
- **`context_precision`** — of the chunks that were retrieved, what fraction were actually relevant/needed, with relevant chunks ranked higher weighted more. Penalizes *retrieved-but-not-needed* noise, especially noise ranked near the top.
- **`context_recall`** — of the facts present in the reference (ground-truth) answer, what fraction can be traced back to the retrieved context. Penalizes *needed-but-missed* information — i.e. retrieval gaps.
- **`answer_correctness`** — a composite of factual overlap (F1 between claims in the generated vs. reference answer) and semantic similarity (embedding distance) between the two answers. The closest thing to an overall "is this answer right" score.

**Worked numeric examples**
- `context_recall`: reference answer contains 5 factual statements; only 3 of them can be attributed to the retrieved chunks → `context_recall = 3/5 = 0.6`.
- `faithfulness`: generated answer makes 4 discrete claims; 3 are grounded in context, 1 is invented → `faithfulness = 3/4 = 0.75`.
- `context_precision`: 5 chunks retrieved, relevance flags by rank `[1,1,0,1,0]` → precision is the mean of precision@k evaluated only at the relevant ranks: `(1/1 + 2/2 + 3/4) / 3 ≈ 0.92`.
- `answer_correctness`: factual F1 = 0.80, semantic similarity = 0.95, default weights `[0.75, 0.25]` → `0.75×0.80 + 0.25×0.95 = 0.8375`.

**Example**
```python
# File: evaluation/evaluate_testset.py
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import (
    answer_correctness, answer_relevancy,
    context_precision, context_recall, faithfulness,
)

llm = llm_factory(settings.openai_model, client=client)          # gpt-4o-mini judge
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]
return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
```
Both offline scripts additionally wrap the judge LLM in a `RunConfig(max_workers=1, timeout=60, max_retries=1)` — deliberately serializing and capping judge calls so a batch run doesn't blow through OpenAI rate limits or hang forever on one bad case.

**Where in this repo**
`evaluation/evaluate_testset.py::run_ragas_evaluation`, `evaluation/evaluate_langfuse.py::run_ragas_evaluation`, and a reference-free subset (`faithfulness`, `answer_relevancy` only) in `app/routers/chat.py::_score_ragas_background`.

**Interview angle**
Q: Why can't you run `context_precision`/`context_recall`/`answer_correctness` on live production traffic the way you can `faithfulness`?
A: Those three metrics need a `reference` (ground-truth answer) and/or `reference_contexts` to compare against — they measure *did we retrieve/answer what the correct answer needed*. Production traffic has no ground truth, only the query, the retrieved contexts, and the generated answer, so only the two reference-free metrics (`faithfulness`, `answer_relevancy`) are usable there; that's precisely the metric list `_score_ragas_background` uses.

## Synthetic Test Set Generation

**What it is**
Hand-writing hundreds of realistic Q&A pairs against a corpus is slow and biased toward what the author thinks to ask. RAGAS's `TestsetGenerator` instead builds a knowledge graph over real indexed chunks and has an LLM synthesize plausible questions plus ground-truth answers and source contexts directly from that content — scaling test-set creation to the actual size of the corpus. The repo's real test set, `qdrant_ragas_testset.csv`, has 102 rows generated this way against the production Qdrant index.

**How it works**
- Feed the generator real document chunks (in the simplified demo in `sample.py`, plain `Document` objects; in production, chunks pulled from the actual indexed corpus).
- The generator assigns each question a **persona** (e.g. `Ethical Educator`, `Aspiring Archer Eklavya`) and a **query style/synthesizer** so the test set isn't uniform — it mimics how different real users phrase things.
- Output columns match exactly what downstream evaluation expects: `user_input`, `reference_contexts`, `reference`, plus generation metadata `persona_name`, `query_style`, `query_length`, `synthesizer_name`.
- `evaluate_testset.py` reads these metadata columns back out (`META_COLS`) so per-case results can be sliced by persona or query style later.

**Example**
```python
# File: sample.py
from ragas.testset import TestsetGenerator
generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
testset = generator.generate_with_langchain_docs(docs, testset_size=5)
df = testset.to_pandas()   # user_input | reference | reference_contexts
```
A real row from `qdrant_ragas_testset.csv` shows the diversity the generator produces on its own — note the deliberately misspelled, short-form query:
```
user_input: "Wht r the imp concerns Arjuna raises in BG?"
persona_name: Ethical Educator
query_style: MISSPELLED
query_length: SHORT
synthesizer_name: single_hop_specific_query_synthesizer
```

**Where in this repo**
`qdrant_ragas_testset.csv` (the generated artifact, 102 rows), generation pattern demonstrated in `sample.py`; consumed by `evaluation/evaluate_testset.py::load_testset` and `evaluation/evaluate_langfuse.py::main`.

**Interview angle**
Q: What's the blind spot of an LLM-generated synthetic test set?
A: It can only generate questions from content that's *already indexed* — it has no notion of what's missing from the corpus, so it can't catch out-of-domain or "the answer isn't in our data at all" failure modes. It also silently goes stale: if you re-chunk or re-embed the corpus, `reference_contexts` may no longer match the new chunk boundaries, so the test set needs periodic regeneration, not just periodic re-running.

## Custom Lexical Metrics

**What it is**
Before paying for an LLM judge, the repo runs two cheap, fully deterministic, word-overlap-based checks that isolate *retrieval* quality in isolation from generation quality. `lexical_context_recall` gives a graded 0–1 score; `reference_context_hit` gives a strict pass/fail. Neither metric calls an LLM, so they're free, instant, and 100% reproducible — useful as a sanity gate before spending judge-model budget, or when you don't trust the judge model at all.

**How it works**
- Normalize text to a `set` of lowercase alphanumeric words via regex (`_normalize_words`).
- `lexical_context_recall = |ref_words ∩ retrieved_words| / |ref_words|` — what fraction of the reference's vocabulary shows up anywhere in what was retrieved.
- `reference_context_hit` is `True` if *either* the word-overlap ratio between a reference context and the combined retrieved text is `≥ 0.45` (only checked when the reference has ≥ 8 words), *or* a 250-character snippet of the reference context appears verbatim (lowercased) inside the retrieved text.
- Interestingly, `evaluate_testset.py` and `evaluate_langfuse.py` reimplement these two functions independently and have drifted slightly: the `evaluate_langfuse.py` version drops the verbatim-snippet check and the `≥8`-word gate, and additionally truncates each retrieved chunk to 1500 characters before comparing ("reduce token explosion for Ragas") — a real bit of duplication/drift worth flagging if asked to review this code.

**Worked numeric example**
Reference context has 40 unique content words; the retrieved chunks together contain 25 of them → `lexical_context_recall = 25/40 = 0.625`. If additionally none of the individual reference contexts clears the 0.45 overlap ratio or the verbatim-snippet check, `reference_context_hit = False` even though the recall score is decent — the two metrics can disagree because one is graded and one is a stricter binary gate.

**Example**
```python
# File: evaluation/evaluate_testset.py
def lexical_context_recall(retrieved: list[str], reference_contexts: list[str]) -> float:
    ref_words = _normalize_words(" ".join(reference_contexts))
    ret_words = _normalize_words(" ".join(retrieved))
    return len(ref_words & ret_words) / len(ref_words)

def reference_context_hit(retrieved: list[str], reference_contexts: list[str]) -> bool:
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
```

**Where in this repo**
`evaluation/evaluate_testset.py::lexical_context_recall`, `::reference_context_hit`; duplicated with minor differences in `evaluation/evaluate_langfuse.py`.

**Interview angle**
Q: Why keep a non-LLM metric around when you already have RAGAS?
A: It decouples two failure modes. If `lexical_context_recall` is high but RAGAS `faithfulness` is low, the problem is generation (the LLM ignored good context). If `lexical_context_recall` itself is low, the problem is retrieval, and no amount of prompt tuning will fix it — you need to look at chunking, embeddings, or the reranker instead.

## DeepEval CI Regression Tests

**What it is**
`test_llm_quality.py` is a `pytest` suite that runs on a small, hand-curated **golden dataset** of 5 domain-specific Q&A pairs (real Chaitanya Charan Das / Bhagavad Gita content) plus 2 **hallucination probes**, using DeepEval's LLM-judged metrics with hard pass/fail thresholds. Unlike the RAGAS batch scripts, this is meant to run in CI on every change — it's the "did we just break something we already know works" check, not a broad drift scan.

**How it works**
- `AnswerRelevancyMetric(threshold=0.7)` and `FaithfulnessMetric(threshold=0.7)` run against every golden-dataset entry; `deepeval.assert_test` raises if the score is below threshold, failing the pytest run.
- `ContextualPrecisionMetric(threshold=0.6)` / `ContextualRecallMetric(threshold=0.6)` run on a subset (`GOLDEN_DATASET[:3]`) — DeepEval's own analogues of RAGAS's context metrics.
- The hallucination section is the interesting one: it doesn't just assert "hallucination score is low." `test_hallucinated_answer_high_hallucination` feeds a **deliberately fabricated** answer (specific dates/numbers not present in the context) and asserts the metric *fails* — wrapping the normally-failing `assert_test` call in `try/except AssertionError` and calling `pytest.fail` only if it unexpectedly *passed*. This tests the detector in both directions.
- Sections 5–6 of the same file (prompt injection resistance, PII anonymization) reuse the same pytest structure to test the guardrail layer, not RAG quality — see `concepts/02-guardrails-and-safety.md`.

**Example**
```python
# File: test_llm_quality.py
def test_hallucinated_answer_high_hallucination(self):
    """A hallucinated answer should fail the metric (score above threshold)."""
    probe = HALLUCINATION_PROBES[0]
    metric = HallucinationMetric(threshold=0.5, model="gpt-4o-mini")
    test_case = LLMTestCase(
        input=probe["input"],
        actual_output=probe["hallucination_answer"],   # fabricated: "1486 AD... 48 years..."
        context=probe["context"],                       # context never states a date
    )
    try:
        assert_test(test_case, [metric])
        pytest.fail("Expected hallucination to be detected but metric passed.")
    except AssertionError:
        pass  # correct — hallucinated answer was caught
```
A simplified, domain-generic mirror of the same pattern (FastAPI trivia instead of spiritual-philosophy content) exists in `sample.py` — useful for seeing the pattern stripped of the real golden dataset.

**Where in this repo**
`test_llm_quality.py` (classes `TestAnswerRelevancy`, `TestFaithfulness`, `TestHallucinationDetection`, `TestContextualMetrics`); generic pattern reference in `sample.py`.

**Interview angle**
Q: Why write a test that expects a metric to *fail*?
A: A metric that always reports "no hallucination" would make every CI run green regardless of whether the detector actually works — a silent false negative. Asserting on both a faithful case (should pass) and a fabricated case (should fail) is how you test the *test* itself, catching a broken or miscalibrated threshold before it ever gets a chance to rubber-stamp a real regression.

## Langfuse Tracing

**What it is**
Langfuse gives full request-level observability: every `/chat` call becomes a tree of nested spans showing exactly what happened at each stage, with structured input/output captured automatically. This is the tool for answering "was this bad answer a retrieval problem or a generation problem?" without re-running anything — you just open the trace.

**How it works**
- `app/core/langfuse_client.py::_init` builds a singleton `Langfuse` client from settings and — importantly — sets `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` as environment variables, because the `@observe` decorator reads credentials from the environment, not from an explicit client reference.
- The trace tree, built purely from decorator placement: `chat` (router) → `retrieval_pipeline` → (`hyde_transform`, `multiquery_transform`, `reranker`) → `llm_generate` (marked `as_type="generation"` so Langfuse renders it as an LLM call with token/cost fields).
- `langfuse_context.update_current_trace(...)` is called explicitly inside the `chat` endpoint to attach the session id and the sanitized query as trace-level `input`, then again at the end to set `output` to the final (PII-restored) answer — so the top-level trace shows the full user-facing exchange even though the decorator itself doesn't know about session/PII logic.

**Example**
```python
# File: app/routers/chat.py
@router.post("", response_model=ChatResponse)
@observe(name="chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks, ...):
    ...
    langfuse_context.update_current_trace(name="chat", session_id=str(session_uuid), input=safe_query)
    ...
    langfuse_context.update_current_trace(output=final_answer)
    trace_id = langfuse_context.get_current_trace_id()
```
```python
# File: app/services/llm_service.py
@observe(name="llm_generate", as_type="generation")
def generate_chat_answer(self, query: str, context: str, history=None) -> str:
    ...
```

**Where in this repo**
`app/core/langfuse_client.py`; `@observe` call sites: `app/routers/chat.py` (`chat`), `app/services/retrieval_pipeline.py` (`retrieve`), `app/services/transforms/hyde.py` / `multiquery.py` (`transform`), `app/services/rerankers/cross_encoder.py` (`rerank`), `app/services/llm_service.py` (`generate_chat_answer`).

**Interview angle**
Q: If a user reports a wrong answer, how do you debug it with this setup?
A: Pull up the trace by session id in Langfuse and read down the span tree: check whether `retrieval_pipeline` returned the right chunks (a retrieval miss), whether `reranker` demoted the correct chunk, or whether `llm_generate`'s captured input context had the right facts but the model still answered wrong (a generation/prompt problem) — the trace tells you which stage to fix without needing to reproduce the bug live.

## Offline Eval Tied to Langfuse Experiments (A/B Testing)

**What it is**
`evaluate_langfuse.py` is a distinct evaluation entrypoint from `evaluate_testset.py`: instead of just writing scores to a CSV, it pushes every retrieval span, generation, and metric score into Langfuse as a structured **experiment trace**, tagged with a `--experiment` name. Running the same script twice with different flags — e.g. `--experiment baseline --top-k 2` vs `--experiment reranker_v2 --top-k 5` — produces two separately-labeled trace groups you can compare side-by-side in the Langfuse UI, turning an offline batch eval into a lightweight A/B testing harness for retrieval-config changes (top-k, reranker version, prompt changes) before shipping them.

**How it works**
- One parent `rag_evaluation` trace per run, tagged with `metadata={"experiment": ..., "top_k": ..., "cases": len(df)}`.
- One child `single_test_case` trace per test-set row, each containing a `retrieval` span and an `llm_generation` generation (mirroring the production trace shape from `run_case`).
- Both the deterministic lexical scores and the RAGAS judge scores get attached per-case via `case_trace.score(name=col, value=...)`, *and* the batch-level means get attached to the parent trace via `experiment_trace.score(...)` — so Langfuse shows both per-case detail and an aggregate you can compare across experiments.
- `langfuse.flush()` is called explicitly (twice) because this is a short-lived script process, not a long-running server — without a manual flush, buffered events could be lost when the process exits.

**Example**
```python
# File: evaluation/evaluate_langfuse.py
experiment_trace = langfuse.trace(
    name="rag_evaluation",
    metadata={"experiment": args.experiment, "top_k": args.top_k, "cases": len(df)},
)
...
for col in SCORE_COLS:
    value = pd.to_numeric(predictions_df[col], errors="coerce").mean()
    if pd.isna(value):
        continue
    experiment_trace.score(name=col, value=float(value))
langfuse.flush()
```

**Where in this repo**
`evaluation/evaluate_langfuse.py::main` (experiment/case trace construction), `::run_case` (per-case span/generation).

**Interview angle**
Q: How would you prove a new reranker actually improved retrieval before merging it?
A: Run `evaluate_langfuse.py` once against the baseline retrieval config and once against the new reranker, tagging each with a distinct `--experiment` name, then compare the aggregate `context_precision`/`context_recall`/`faithfulness` scores attached to each experiment's parent trace in Langfuse — a quantified, side-by-side comparison instead of eyeballing a handful of manual examples.

## Async Production Scoring

**What it is**
Running an LLM-judge evaluation *before* returning a chat response would add real latency to every request for a benefit the user never sees. Instead, `_score_ragas_background` runs faithfulness/relevancy scoring on the real, just-served query+answer+context *after* the HTTP response has already gone out, via FastAPI's `BackgroundTasks`, and writes the resulting scores back onto the very same Langfuse trace the request created.

**How it works**
- `chat()` grabs `trace_id = langfuse_context.get_current_trace_id()` right before returning, and calls `background_tasks.add_task(_score_ragas_background, trace_id=..., query=safe_query, answer=final_answer, contexts=retrieved_contexts)`.
- Because there's no ground-truth reference available in production, the RAGAS dataset is built with only `user_input`/`response`/`retrieved_contexts` — no `reference` column — which restricts scoring to the two reference-free metrics: `faithfulness` and `answer_relevancy`.
- The judge call is bounded with the same `RunConfig(max_workers=1, timeout=60, max_retries=1)` used by the offline scripts, so one slow/failing judge call can't hang the background worker indefinitely.
- Scores are attached to the original trace with `lf.score(trace_id=trace_id, name=col, value=float(val))` — so anyone looking at that request's trace in Langfuse later sees quality scores sitting right next to it, computed after the fact.
- The whole function body is wrapped in `try/except Exception` with only a log call — a judge-scoring failure must never surface to, or affect, the user, since their response was already sent.

**Example**
```python
# File: app/routers/chat.py
def _score_ragas_background(trace_id: str, query: str, answer: str, contexts: list[str]) -> None:
    try:
        lf = get_langfuse()
        if lf is None:
            return
        dataset = Dataset.from_list([{
            "user_input": query, "response": answer, "retrieved_contexts": contexts,
        }])
        result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy],
                           llm=llm, embeddings=embeddings,
                           run_config=RunConfig(max_workers=1, timeout=60, max_retries=1))
        for col in ["faithfulness", "answer_relevancy"]:
            val = result.to_pandas().iloc[0][col]
            if pd.notna(val):
                lf.score(trace_id=trace_id, name=col, value=float(val))
        lf.flush()
    except Exception as e:
        logging.getLogger(__name__).error("[RAGAS] %s: %s", type(e).__name__, e)
```

**Where in this repo**
`app/routers/chat.py::_score_ragas_background`, wired up in `chat()` via `background_tasks.add_task(...)`.

**Interview angle**
Q: Doesn't scoring every single production request with an LLM judge get expensive fast?
A: Yes — this is the tradeoff being made deliberately: it buys per-request quality visibility (every trace gets scored, not a sample) at the cost of one extra `gpt-4o-mini` call per request, run off the critical path. At high volume you'd want to sample (e.g. score 1-in-N requests) rather than score every single one; the current code scores 100% of requests, which is fine at low traffic but would need a sampling gate to stay cheap at scale.

## Outlier-Aware Analysis

**What it is**
A batch mean can hide real problems: a `faithfulness` mean of 0.85 across 100 cases sounds healthy, but if 5 of those cases scored 0.1, the mean is masking real hallucinations. Rather than trust the mean alone, `evaluate_testset.py::_find_low_outliers` flags individual cases as outliers using a combination of a **relative** cutoff (below the batch's own mean minus 1.5 standard deviations) and an **absolute floor** (below 0.35 regardless of how the rest of the batch scored).

**How it works**
- Compute `mean` and `std` of a metric column across all cases in the batch.
- `cutoff = mean - 1.5 * std` (only if `std > 0`; if every case scored identically, there's no distribution to be an outlier from, so the cutoff collapses to the mean and nothing but the absolute floor can flag).
- A case is flagged if its score is below `cutoff` **or** below the absolute floor of `0.35` — the two reasons are tracked and reported separately (`"below mean−1.5σ (0.612)"` vs `"below 0.35"`), and a case can be flagged for both.
- The relative cutoff catches "this batch is generally fine but a few cases are disasters"; the absolute floor catches "this whole batch, including its mean, is mediocre" (a scenario where every score might be *above* its own `mean - 1.5σ` yet still objectively bad).
- Results feed into `_summary.json`'s `low_outliers` list and are printed directly to the console (`row`, `metric`, `value`, `reason`, truncated `user_input`) so a human can jump straight to the failing query.

**Worked numeric example**
Suppose `faithfulness` scores across 20 cases average `mean = 0.80` with `std = 0.12`. `cutoff = 0.80 - 1.5×0.12 = 0.62`. A case scoring `0.55` is flagged as "below mean−1.5σ (0.620)" (relative problem) even though it's nowhere near the 0.35 absolute floor; a separate case scoring `0.20` is flagged for *both* reasons simultaneously.

**Example**
```python
# File: evaluation/evaluate_testset.py
def _find_low_outliers(df, col, *, z_threshold=1.5, absolute_floor=0.35) -> list[dict]:
    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.dropna()
    mean = float(valid.mean())
    std = float(valid.std()) if len(valid) > 1 else 0.0
    cutoff = mean - z_threshold * std if std > 0 else mean
    outliers = []
    for idx, row in df.iterrows():
        val = values.loc[idx]
        reasons = []
        if std > 0 and float(val) < cutoff:
            reasons.append(f"below mean−{z_threshold}σ ({cutoff:.3f})")
        if float(val) < absolute_floor:
            reasons.append(f"below {absolute_floor}")
        if reasons:
            outliers.append({"row": int(idx), "metric": col, "value": float(val), "reason": "; ".join(reasons)})
    return outliers
```
A minimal version of the same idea, one metric at a time with pandas boolean filtering instead of a reasons list, appears in `sample.py`:
```python
# File: sample.py
for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
    scores = results_df[metric].dropna()
    cutoff = scores.mean() - 1.5 * scores.std()
    outliers = results_df[(results_df[metric] < cutoff) | (results_df[metric] < 0.35)]
```

**Where in this repo**
`evaluation/evaluate_testset.py::_find_low_outliers` (thorough version, feeding `summary.json`); simplified pattern in `sample.py`.

**Interview angle**
Q: Why use both a relative (std-based) and an absolute cutoff instead of just one?
A: A purely relative cutoff is meaningless if the whole batch is quietly bad — every score can sit comfortably within 1.5σ of a low mean and nothing gets flagged. A purely absolute cutoff misses genuine regressions in an otherwise-good batch, since a handful of 0.5 scores might never dip under a fixed floor of 0.35 while still being far worse than everything else. Combining both catches "a few catastrophic cases" and "the whole batch drifted down" as two distinct failure signatures.

## Two-Tier Eval Strategy

**What it is**
The repo deliberately runs two evaluation regimes that answer different questions and are not substitutes for each other: a fast, deterministic, LLM-judged-but-narrow **golden-set test in CI** (`test_llm_quality.py`), and a slower, broader, entirely-LLM-judged **batch evaluation** run on demand or on a schedule (`evaluation/evaluate_testset.py` / `evaluate_langfuse.py`).

**How it works**
- **Tier 1 — CI (`test_llm_quality.py`)**: `pytest`-native, runs on every change, uses a small (5 + 2 case) hand-picked golden dataset with fixed pass/fail thresholds (`AnswerRelevancyMetric(threshold=0.7)`, etc.). It's cheap enough and small enough to gate a merge/deploy. It answers: *"did this change break a case we already know the answer to?"*
- **Tier 2 — batch eval (`evaluate_testset.py`)**: runs the full live retrieval pipeline against all 102 rows of the synthetically-generated `qdrant_ragas_testset.csv`, scores with the full 5-metric RAGAS suite plus the 2 lexical metrics, and writes `*_predictions.csv`, `*_results.csv`, `*_metrics_per_case.csv`, and `*_summary.json` (means/min/max + per-case scores + low outliers). It's too slow, too costly, and too nondeterministic (LLM-judge variance) to gate CI, but it's the only thing that exercises the *real* production retrieval pipeline (`get_retrieval_pipeline()`, live Qdrant) end-to-end and can catch drift across a much wider and more realistic set of query styles.
- Neither tier replaces the other: Tier 1 would never catch "retrieval quality slowly degraded after a re-embedding," and Tier 2 is far too slow and expensive to run on every commit.

**Example**
```bash
# Tier 1 — fast, blocks CI
pytest test_llm_quality.py -v

# Tier 2 — slow, run on demand / scheduled (e.g. via Airflow)
python evaluation/evaluate_testset.py --limit 20     # smoke test
python evaluation/evaluate_testset.py                # full 102-row batch run
python evaluation/evaluate_langfuse.py --experiment reranker_v2 --top-k 5
```

**Where in this repo**
Tier 1: `test_llm_quality.py`. Tier 2: `evaluation/evaluate_testset.py`, `evaluation/evaluate_langfuse.py`, consuming `qdrant_ragas_testset.csv`.

**Interview angle**
Q: Why not just run the full RAGAS batch suite in CI and skip the separate golden-dataset tests?
A: Cost, latency, and determinism. RAGAS's LLM-judge metrics take real API calls per case per metric (5 metrics × 102 rows here), so a full run is slow and non-free, and judge-model outputs vary run to run — exactly the properties you don't want gating a merge. The golden-dataset tests trade breadth for speed and stability: a small, fixed set of cases with fixed thresholds that can fail fast and reliably on every PR, while the broader RAGAS batch run is reserved for periodic or on-demand drift detection where variance and cost are acceptable.
