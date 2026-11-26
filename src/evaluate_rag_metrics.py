#!/usr/bin/env python3
"""
Unified RAG Evaluation CLI

Computes *all* metrics from the previous LLM and no-LLM evaluators:

Per-row (LLM / Ragas collections, when an LLM is configured):
  - Context Precision, Context Recall, Context Entities Recall
  - Noise Sensitivity, Response Relevancy (AnswerRelevancy)
  - Faithfulness, Answer Accuracy, Context Relevance
  - Response Groundedness, Factual Correctness
  - NonLLMStringSimilarity, BleuScore, CHRFScore, RougeScore (Ragas)

Per-row (always, pure lexical / local-embedding fallbacks):
  - metric_bleu, metric_chrf, metric_rouge_l
  - semantic_similarity  (sentence-transformers, when installed)

Corpus-level (when ranx is available):
  - MRR, MAP, NDCG@k, MAP@k

For every metric the script also records:
  - <metric>_latency_ms
  - <metric>_input_tokens
  - <metric>_output_tokens

Token counts are taken from the underlying LLM/embedding response when
available; otherwise they are estimated from text length (≈ 4 chars/token)
for LLM calls and set to 0 for pure lexical / ranking metrics.

Usage examples:
  python evaluate_rag_metrics_all.py -i questions.csv
  python evaluate_rag_metrics_all.py -i input.csv -o out.csv --limit 10
  python evaluate_rag_metrics_all.py -i input.csv --llm-config llm_config.yaml
  python evaluate_rag_metrics_all.py -i input.csv --no-llm --no-ranking
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------
HAS_RAGAS = False
HAS_RANX = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_NLTK = False
HAS_OPENAI = False
HAS_TIKTOKEN = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    pass

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    pass

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        ContextEntityRecall,
        NoiseSensitivity,
        AnswerRelevancy,
        Faithfulness,
        AnswerAccuracy,
        ContextRelevance,
        ResponseGroundedness,
        FactualCorrectness,
        NonLLMStringSimilarity,
        DistanceMeasure,
        BleuScore as RagasBleuScore,
        CHRFScore as RagasCHRFScore,
        RougeScore as RagasRougeScore,
    )
    HAS_RAGAS = True
except ImportError as e:
    print(f"[warn] RAGAS / datasets not fully available: {e}", file=sys.stderr)

try:
    from ranx import Qrels, Run, evaluate as ranx_evaluate
    HAS_RANX = True
except ImportError as e:
    print(f"[warn] ranx not available: {e}", file=sys.stderr)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.tokenize import word_tokenize

    for res in ("punkt", "punkt_tab", "wordnet"):
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    HAS_NLTK = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Token / latency helpers
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Score + instrumentation for a single metric evaluation."""
    value: Optional[float] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈ 4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    if not HAS_TIKTOKEN or not text:
        return estimate_tokens(text)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return estimate_tokens(text)
    return len(enc.encode(text))


def estimate_kwargs_tokens(kwargs: Dict[str, Any], model: str = "gpt-4o") -> int:
    """Sum token estimates for all string / list-of-string arguments."""
    total = 0
    for v in kwargs.values():
        if isinstance(v, str):
            total += count_tokens_tiktoken(v, model)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    total += count_tokens_tiktoken(item, model)
    return total


# ---------------------------------------------------------------------------
# Config helpers (LLM path)
# ---------------------------------------------------------------------------

def load_llm_config(
    config_path: Optional[str],
    entry: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not config_path:
        return None, None
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if entry not in config:
        raise KeyError(
            f"Entry '{entry}' not found in {config_path}. "
            f"Available keys: {list(config.keys())}"
        )

    entry_cfg = config[entry]
    provider = entry_cfg.get("provider")
    if not provider:
        raise ValueError(f"No 'provider' key under '{entry}' in config.")

    if "api_key" in entry_cfg:
        key = entry_cfg["api_key"]
        if "anthropic" in provider.lower() or "claude" in provider.lower():
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            os.environ.setdefault("OPENAI_API_KEY", key)

    return provider, entry_cfg.get("model")


def create_evaluation_llm(model: str, temperature: float = 0.0, max_tokens: int = 2048):
    """Create a Ragas LLM judge via llm_factory (OpenAI-compatible client)."""
    if not HAS_OPENAI or not HAS_RAGAS:
        raise RuntimeError("OpenAI client and RAGAS are required for LLM metrics")
    client = AsyncOpenAI()
    return llm_factory(
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_embeddings(model: str = "text-embedding-3-small"):
    """Create embeddings needed by AnswerRelevancy."""
    if not HAS_OPENAI or not HAS_RAGAS:
        raise RuntimeError("OpenAI client and RAGAS are required for embeddings")
    client = AsyncOpenAI()
    return embedding_factory(
        "openai",
        model=model,
        client=client,
    )


# ---------------------------------------------------------------------------
# Data prep helpers (shared)
# ---------------------------------------------------------------------------

def parse_generation_chunks(raw: Any) -> List[Dict[str, Any]]:
    """Parse the generation_chunks column (stringified list of dicts)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []


def extract_contexts(chunks: List[Dict[str, Any]]) -> List[str]:
    """Return ordered list of context texts (highest score first)."""
    if not chunks:
        return []
    try:
        sorted_c = sorted(
            chunks,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
    except Exception:
        sorted_c = chunks
    return [str(c.get("text", "")).strip() for c in sorted_c if c.get("text")]


def extract_chunks_legacy(x) -> pd.Series:
    """Legacy helper used by the original LLM script (scores + texts)."""
    if pd.isna(x):
        return pd.Series([[], []])
    chunks = ast.literal_eval(x) if isinstance(x, str) else x
    if not isinstance(chunks, list):
        return pd.Series([[], []])
    scores = [chunk.get("score") for chunk in chunks if isinstance(chunk, dict)]
    texts = [chunk.get("text") for chunk in chunks if isinstance(chunk, dict)]
    return pd.Series([scores, texts])


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalise the columns required by the metrics."""
    required = ["question", "generation_chunks", "generation_answer", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    sample = df[required].copy()
    sample[["generation_chunk_scores", "generation_chunks"]] = (
        sample["generation_chunks"].apply(extract_chunks_legacy)
    )
    sample["generation_chunks"] = sample["generation_chunks"].apply(
        lambda xs: [t for t in (xs or []) if isinstance(t, str) and t.strip()]
    )
    return sample


def token_overlap_ratio(a: str, b: str) -> float:
    """Simple Jaccard-like overlap on lower-cased word tokens."""
    if not a or not b:
        return 0.0
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_relevant_chunk(
    chunk_text: str,
    gold_context: str,
    gold_answer: str,
    threshold: float = 0.25,
) -> bool:
    """Heuristic relevance: high token overlap with gold context or answer."""
    if not chunk_text:
        return False
    o1 = token_overlap_ratio(chunk_text, gold_context or "")
    o2 = token_overlap_ratio(chunk_text, gold_answer or "")
    return max(o1, o2) >= threshold


# ---------------------------------------------------------------------------
# Lexical metrics (always available fallbacks)
# ---------------------------------------------------------------------------

def compute_bleu(pred: str, ref: str) -> float:
    if not HAS_NLTK or not pred or not ref:
        return np.nan
    try:
        ref_toks = [word_tokenize(ref.lower())]
        pred_toks = word_tokenize(pred.lower())
        smoothie = SmoothingFunction().method1
        return float(sentence_bleu(ref_toks, pred_toks, smoothing_function=smoothie))
    except Exception:
        return np.nan


def compute_rouge_l(pred: str, ref: str) -> float:
    """Lightweight ROUGE-L F1 (no external rouge package required)."""
    if not pred or not ref:
        return np.nan
    try:
        pred_toks = re.findall(r"\w+", pred.lower())
        ref_toks = re.findall(r"\w+", ref.lower())
        if not pred_toks or not ref_toks:
            return 0.0
        m, n = len(pred_toks), len(ref_toks)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_toks[i - 1] == ref_toks[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        prec = lcs / m if m else 0.0
        rec = lcs / n if n else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    except Exception:
        return np.nan


def compute_chrf(pred: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (approx CHRF)."""
    if not pred or not ref:
        return np.nan
    try:
        from collections import Counter

        def ngrams(s: str, order: int):
            s = s.lower().replace(" ", "")
            return [s[i : i + order] for i in range(max(0, len(s) - order + 1))]

        precisions, recalls = [], []
        for order in range(1, n + 1):
            pn = ngrams(pred, order)
            rn = ngrams(ref, order)
            if not pn or not rn:
                continue
            pc, rc = Counter(pn), Counter(rn)
            overlap = sum((pc & rc).values())
            precisions.append(overlap / len(pn))
            recalls.append(overlap / len(rn))
        if not precisions:
            return 0.0
        p = float(np.mean(precisions))
        r = float(np.mean(recalls))
        if p + r == 0:
            return 0.0
        beta2 = beta ** 2
        return (1 + beta2) * p * r / (beta2 * p + r)
    except Exception:
        return np.nan


def timed_lexical(fn, pred: str, ref: str) -> MetricResult:
    """Run a pure-lexical metric and record latency (tokens = 0)."""
    t0 = time.perf_counter()
    try:
        val = fn(pred, ref)
        latency = (time.perf_counter() - t0) * 1000.0
        return MetricResult(value=float(val) if pd.notna(val) else None, latency_ms=latency)
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        return MetricResult(latency_ms=latency, error=str(exc))


# ---------------------------------------------------------------------------
# Ranking metrics via Ranx
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    rows: List[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5, 10),
    relevance_threshold: float = 0.25,
) -> Dict[str, MetricResult]:
    """
    Build Qrels / Run from the dataset using the relevance heuristic and
    compute MRR, NDCG@k, MAP@k with Ranx. Returns MetricResult objects.
    """
    metric_names = (
        ["mrr", "map"]
        + [f"ndcg@{k}" for k in k_values]
        + [f"map@{k}" for k in k_values]
    )
    empty = {m: MetricResult(value=None) for m in metric_names}
    if not HAS_RANX:
        return empty

    t0 = time.perf_counter()
    qrels_dict: Dict[str, Dict[str, int]] = {}
    run_dict: Dict[str, Dict[str, float]] = {}

    for i, row in enumerate(rows):
        qid = f"q{i}"
        gold_ctx = str(row.get("chunks") or "")
        gold_ans = str(row.get("answer") or "")
        gen_chunks = parse_generation_chunks(row.get("generation_chunks"))

        rels = {}
        for c in gen_chunks:
            cid = str(c.get("chunk_id", id(c)))
            if is_relevant_chunk(
                str(c.get("text", "")),
                gold_ctx,
                gold_ans,
                threshold=relevance_threshold,
            ):
                rels[cid] = 1
        if rels:
            qrels_dict[qid] = rels

        ranking = {}
        for c in sorted(
            gen_chunks, key=lambda x: float(x.get("score", 0.0)), reverse=True
        ):
            cid = str(c.get("chunk_id", id(c)))
            ranking[cid] = float(c.get("score", 0.0))
        if ranking:
            run_dict[qid] = ranking

    latency = (time.perf_counter() - t0) * 1000.0

    if not qrels_dict or not run_dict:
        return {m: MetricResult(latency_ms=latency) for m in metric_names}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    try:
        results = ranx_evaluate(qrels, run, metric_names, make_comparable=True)
        return {
            m: MetricResult(
                value=float(results.get(m, np.nan)) if not np.isnan(results.get(m, np.nan)) else None,
                latency_ms=latency / len(metric_names),  # share total time
            )
            for m in metric_names
        }
    except Exception as e:
        print(f"[warn] Ranx evaluation failed: {e}", file=sys.stderr)
        return {m: MetricResult(latency_ms=latency / len(metric_names), error=str(e)) for m in metric_names}


# ---------------------------------------------------------------------------
# Ragas LLM metric runners (instrumented)
# ---------------------------------------------------------------------------

async def safe_ascore_instrumented(
    scorer,
    model_name: str = "gpt-4o",
    **kwargs,
) -> MetricResult:
    """
    Call scorer.ascore, measure wall-clock latency and estimate tokens from
    the kwargs that were sent to the metric (best-effort; true usage would
    require a full client callback).
    """
    input_toks = estimate_kwargs_tokens(kwargs, model=model_name)
    t0 = time.perf_counter()
    try:
        result = await scorer.ascore(**kwargs)
        latency = (time.perf_counter() - t0) * 1000.0
        value = float(result.value) if result is not None and result.value is not None else None

        # Best-effort output token estimate from the score object if it carries text
        output_toks = 0
        if hasattr(result, "reason") and isinstance(result.reason, str):
            output_toks = count_tokens_tiktoken(result.reason, model_name)
        elif hasattr(result, "explanation") and isinstance(getattr(result, "explanation"), str):
            output_toks = count_tokens_tiktoken(result.explanation, model_name)

        return MetricResult(
            value=value,
            latency_ms=latency,
            input_tokens=input_toks,
            output_tokens=output_toks,
        )
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"  [warn] {scorer.__class__.__name__} failed: {exc}", file=sys.stderr)
        return MetricResult(
            latency_ms=latency,
            input_tokens=input_toks,
            error=str(exc),
        )


async def score_row_llm(
    row: pd.Series,
    scorers: Dict[str, Any],
    model_name: str = "gpt-4o",
) -> Dict[str, MetricResult]:
    """Run every LLM/Ragas metric on a single row."""
    user_input = row["question"]
    response = row["generation_answer"] if pd.notna(row["generation_answer"]) else ""
    reference = row["answer"] if pd.notna(row["answer"]) else ""
    contexts: List[str] = row["generation_chunks"] or []

    results: Dict[str, MetricResult] = {}

    # Context Precision (needs reference)
    results["context_precision"] = await safe_ascore_instrumented(
        scorers["context_precision"],
        model_name=model_name,
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Context Recall
    results["context_recall"] = await safe_ascore_instrumented(
        scorers["context_recall"],
        model_name=model_name,
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Context Entities Recall
    results["context_entities_recall"] = await safe_ascore_instrumented(
        scorers["context_entities_recall"],
        model_name=model_name,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Noise Sensitivity
    results["noise_sensitivity"] = await safe_ascore_instrumented(
        scorers["noise_sensitivity"],
        model_name=model_name,
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Response Relevancy (AnswerRelevancy)
    results["response_relevancy"] = await safe_ascore_instrumented(
        scorers["response_relevancy"],
        model_name=model_name,
        user_input=user_input,
        response=response,
    )

    # Faithfulness
    results["faithfulness"] = await safe_ascore_instrumented(
        scorers["faithfulness"],
        model_name=model_name,
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts,
    )

    # Answer Accuracy
    results["answer_accuracy"] = await safe_ascore_instrumented(
        scorers["answer_accuracy"],
        model_name=model_name,
        user_input=user_input,
        response=response,
        reference=reference,
    )

    # Context Relevance
    results["context_relevance"] = await safe_ascore_instrumented(
        scorers["context_relevance"],
        model_name=model_name,
        user_input=user_input,
        retrieved_contexts=contexts,
    )

    # Response Groundedness
    results["response_groundedness"] = await safe_ascore_instrumented(
        scorers["response_groundedness"],
        model_name=model_name,
        response=response,
        retrieved_contexts=contexts,
    )

    # Factual Correctness
    results["factual_correctness"] = await safe_ascore_instrumented(
        scorers["factual_correctness"],
        model_name=model_name,
        response=response,
        reference=reference,
    )

    # Non-LLM string similarity (still goes through the collections API)
    results["nonllm_string_similarity"] = await safe_ascore_instrumented(
        scorers["nonllm_string_similarity"],
        model_name=model_name,
        response=response,
        reference=reference,
    )

    # Ragas Bleu / CHRF / Rouge (collections)
    results["ragas_bleu"] = await safe_ascore_instrumented(
        scorers["ragas_bleu"],
        model_name=model_name,
        response=response,
        reference=reference,
    )
    results["ragas_chrf"] = await safe_ascore_instrumented(
        scorers["ragas_chrf"],
        model_name=model_name,
        response=response,
        reference=reference,
    )
    results["ragas_rouge"] = await safe_ascore_instrumented(
        scorers["ragas_rouge"],
        model_name=model_name,
        response=response,
        reference=reference,
    )

    return results


# ---------------------------------------------------------------------------
# Per-row pure lexical
# ---------------------------------------------------------------------------

def compute_semantic_similarity(
    pred: str,
    ref: str,
    model: Optional["SentenceTransformer"] = None,
) -> float:
    """Cosine similarity of sentence-transformer embeddings (0–1)."""
    if model is None or not pred or not ref:
        return np.nan
    try:
        emb = model.encode([pred, ref], show_progress_bar=False, normalize_embeddings=True)
        # already L2-normalised → dot product == cosine
        return float(np.dot(emb[0], emb[1]))
    except Exception:
        return np.nan


def add_per_row_lexical(df: pd.DataFrame) -> pd.DataFrame:
    """Add BLEU / CHRF / ROUGE-L / semantic-similarity columns + latency / token columns."""
    bleus, chrfs, rouges, semsims = [], [], [], []
    bleu_lat, chrf_lat, rouge_lat, sem_lat = [], [], [], []

    # Load a local embedding model once if available
    st_model = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            print("[info] Loading sentence-transformers model 'all-MiniLM-L6-v2' for semantic similarity ...")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[warn] Could not load SentenceTransformer: {e}", file=sys.stderr)

    for _, row in df.iterrows():
        pred = str(row.get("generation_answer") or "")
        ref = str(row.get("answer") or "")

        r = timed_lexical(compute_bleu, pred, ref)
        bleus.append(r.value)
        bleu_lat.append(r.latency_ms)

        r = timed_lexical(compute_chrf, pred, ref)
        chrfs.append(r.value)
        chrf_lat.append(r.latency_ms)

        r = timed_lexical(compute_rouge_l, pred, ref)
        rouges.append(r.value)
        rouge_lat.append(r.latency_ms)

        # Semantic similarity (local embeddings, no LLM tokens)
        t0 = time.perf_counter()
        sim = compute_semantic_similarity(pred, ref, model=st_model)
        sem_lat.append((time.perf_counter() - t0) * 1000.0)
        semsims.append(sim if pd.notna(sim) else None)

    out = df.copy()
    out["metric_bleu"] = bleus
    out["metric_bleu_latency_ms"] = bleu_lat
    out["metric_bleu_input_tokens"] = 0
    out["metric_bleu_output_tokens"] = 0

    out["metric_chrf"] = chrfs
    out["metric_chrf_latency_ms"] = chrf_lat
    out["metric_chrf_input_tokens"] = 0
    out["metric_chrf_output_tokens"] = 0

    out["metric_rouge_l"] = rouges
    out["metric_rouge_l_latency_ms"] = rouge_lat
    out["metric_rouge_l_input_tokens"] = 0
    out["metric_rouge_l_output_tokens"] = 0

    out["semantic_similarity"] = semsims
    out["semantic_similarity_latency_ms"] = sem_lat
    out["semantic_similarity_input_tokens"] = 0
    out["semantic_similarity_output_tokens"] = 0
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified RAG metrics evaluation (LLM + no-LLM) with latency & token counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input CSV (must contain question, answer, generation_answer, generation_chunks)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_all_metrics.csv",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for testing)",
    )
    # LLM options
    p.add_argument(
        "--llm-config",
        default=None,
        help="Path to llm_config.yaml (required for LLM-based Ragas metrics)",
    )
    p.add_argument(
        "--llm-entry",
        default="claude_5_5",
        help="Key inside the YAML config that selects the model / API key",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model used by Response Relevancy",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip all LLM-based Ragas metrics even if a config is provided",
    )
    # Ranking / RAGAS batch options
    p.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.25,
        help="Token-overlap threshold used to build qrels for ranking metrics",
    )
    p.add_argument(
        "--no-ranking",
        action="store_true",
        help="Skip ranking metrics (MRR / NDCG / MAP)",
    )
    return p.parse_args(argv)


def _attach_metric_columns(df: pd.DataFrame, col: str, results: List[MetricResult]) -> None:
    """Write value / latency / tokens columns for a list of MetricResult."""
    df[col] = [r.value for r in results]
    df[f"{col}_latency_ms"] = [r.latency_ms for r in results]
    df[f"{col}_input_tokens"] = [r.input_tokens for r in results]
    df[f"{col}_output_tokens"] = [r.output_tokens for r in results]


async def main_async(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"[info] Loading {args.input} ...")
    raw_df = pd.read_csv(args.input)
    print(f"[info] Loaded {len(raw_df)} rows, columns: {list(raw_df.columns)}")

    required = {"question", "answer", "generation_answer", "generation_chunks"}
    missing = required - set(raw_df.columns)
    if missing:
        print(f"[error] Missing required columns: {missing}", file=sys.stderr)
        return 1

    if args.limit is not None:
        raw_df = raw_df.head(args.limit).copy()
        print(f"[info] Limited to first {len(raw_df)} rows")

    # ------------------------------------------------------------------
    # 1. Per-row pure lexical metrics (always)
    # ------------------------------------------------------------------
    print("[info] Computing per-row lexical / embedding metrics (BLEU, CHRF, ROUGE-L, semantic similarity) ...")
    raw_df = add_per_row_lexical(raw_df)

    # ------------------------------------------------------------------
    # 2. Ranking metrics (Ranx) – corpus level
    # ------------------------------------------------------------------
    ranking_results: Dict[str, MetricResult] = {}
    if not args.no_ranking and HAS_RANX:
        print("[info] Computing ranking metrics with Ranx (MRR, NDCG, MAP) ...")
        rows_as_dicts = raw_df.to_dict(orient="records")
        ranking_results = compute_ranking_metrics(
            rows_as_dicts,
            relevance_threshold=args.relevance_threshold,
        )
        for k, r in ranking_results.items():
            if r.value is None or (isinstance(r.value, float) and np.isnan(r.value)):
                print(f"       {k}: nan  (latency {r.latency_ms:.1f} ms)")
            else:
                print(f"       {k}: {r.value:.4f}  (latency {r.latency_ms:.1f} ms)")
            # broadcast corpus metrics to every row
            raw_df[f"corpus_{k}"] = r.value
            raw_df[f"corpus_{k}_latency_ms"] = r.latency_ms
            raw_df[f"corpus_{k}_input_tokens"] = 0
            raw_df[f"corpus_{k}_output_tokens"] = 0
    else:
        reason = "disabled by --no-ranking" if args.no_ranking else "ranx not installed"
        print(f"[info] Skipping ranking metrics ({reason})")

    # ------------------------------------------------------------------
    # 3. LLM-based Ragas metrics (per-row)
    # ------------------------------------------------------------------
    llm_metric_names: List[str] = []
    use_llm = (
        not args.no_llm
        and args.llm_config is not None
        and HAS_RAGAS
        and HAS_OPENAI
    )

    if use_llm:
        provider, llm_model = load_llm_config(args.llm_config, args.llm_entry)
        if not llm_model:
            print("[error] Could not resolve model from LLM config", file=sys.stderr)
            return 1
        print(f"[info] Using provider={provider}, model={llm_model}")

        evaluation_llm = create_evaluation_llm(llm_model, temperature=0.0, max_tokens=2048)
        embeddings = create_embeddings(args.embedding_model)

        scorers = {
            "context_precision": ContextPrecision(llm=evaluation_llm),
            "context_recall": ContextRecall(llm=evaluation_llm),
            "context_entities_recall": ContextEntityRecall(llm=evaluation_llm),
            "noise_sensitivity": NoiseSensitivity(llm=evaluation_llm),
            "response_relevancy": AnswerRelevancy(llm=evaluation_llm, embeddings=embeddings),
            "faithfulness": Faithfulness(llm=evaluation_llm),
            "answer_accuracy": AnswerAccuracy(llm=evaluation_llm),
            "context_relevance": ContextRelevance(llm=evaluation_llm),
            "response_groundedness": ResponseGroundedness(llm=evaluation_llm),
            "factual_correctness": FactualCorrectness(llm=evaluation_llm),
            "nonllm_string_similarity": NonLLMStringSimilarity(
                distance_measure=DistanceMeasure.LEVENSHTEIN
            ),
            "ragas_bleu": RagasBleuScore(),
            "ragas_chrf": RagasCHRFScore(),
            "ragas_rouge": RagasRougeScore(),
        }
        llm_metric_names = list(scorers.keys())

        # Prepare the normalised sample used by the original LLM script
        sample = prepare_dataframe(raw_df)
        n = len(sample)
        print(f"[info] Running LLM/Ragas evaluation on {n:,} records ...")

        # Collect results per metric
        collected: Dict[str, List[MetricResult]] = {name: [] for name in llm_metric_names}

        for idx in tqdm(range(n), desc="Evaluating (LLM)"):
            row = sample.iloc[idx]
            scores = await score_row_llm(row, scorers, model_name=llm_model or "gpt-4o")
            for name, mr in scores.items():
                collected[name].append(mr)

        for name, results in collected.items():
            _attach_metric_columns(raw_df, name, results)

        # Quick summary
        print("\n[summary] LLM metric means (non-null):")
        for name in llm_metric_names:
            mean_val = raw_df[name].mean(skipna=True)
            mean_lat = raw_df[f"{name}_latency_ms"].mean(skipna=True)
            mean_in = raw_df[f"{name}_input_tokens"].mean(skipna=True)
            mean_out = raw_df[f"{name}_output_tokens"].mean(skipna=True)
            if pd.notna(mean_val):
                print(
                    f"  {name:28s}: {mean_val:.4f}  "
                    f"(lat {mean_lat:.0f} ms, in {mean_in:.0f} tok, out {mean_out:.0f} tok)"
                )
            else:
                print(f"  {name:28s}: n/a")
    else:
        reasons = []
        if args.no_llm:
            reasons.append("--no-llm")
        if args.llm_config is None:
            reasons.append("no --llm-config")
        if not HAS_RAGAS:
            reasons.append("RAGAS not installed")
        if not HAS_OPENAI:
            reasons.append("openai package not installed")
        print(f"[info] Skipping LLM metrics ({', '.join(reasons)})")

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        out_path = args.input.with_name(args.input.stem + "_all_metrics.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer Helper.Log if present (as in the original scripts), else plain write
    try:
        from Helper import Log
        with Log.save_file(out_path) as filename:
            raw_df.to_csv(filename, index=False, quoting=1)
    except ImportError:
        raw_df.to_csv(out_path, index=False, quoting=1)
        print(f"[info] Wrote enriched CSV → {out_path}")

    # Summary JSON
    summary = {
        "n_rows": len(raw_df),
        "lexical_means": {
            "bleu": float(np.nanmean(raw_df["metric_bleu"])) if "metric_bleu" in raw_df else None,
            "chrf": float(np.nanmean(raw_df["metric_chrf"])) if "metric_chrf" in raw_df else None,
            "rouge_l": float(np.nanmean(raw_df["metric_rouge_l"])) if "metric_rouge_l" in raw_df else None,
            "semantic_similarity": (
                float(np.nanmean(raw_df["semantic_similarity"]))
                if "semantic_similarity" in raw_df
                else None
            ),
        },
        "ranking": {
            k: {"value": r.value, "latency_ms": r.latency_ms}
            for k, r in ranking_results.items()
        },
        "llm_metrics": llm_metric_names,
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[info] Wrote summary → {summary_path}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
