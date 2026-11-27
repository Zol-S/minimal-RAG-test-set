"""Shared metric computation helpers and result containers."""

from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------
HAS_RAGAS = False
HAS_RANX = False
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
    from datasets import Dataset  # noqa: F401
    from ragas import evaluate as ragas_evaluate  # noqa: F401
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


@dataclass
class MetricResult:
    """Score + instrumentation for a single metric evaluation."""

    value: Optional[float] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    confidence: Optional[float] = None  # 0–1 heuristic confidence
    margin_of_error: Optional[float] = None  # absolute margin (±)


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
# Lexical metrics
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
        # Heuristic confidence: higher when both strings are reasonably long
        conf = None
        moe = None
        if pd.notna(val):
            conf = min(1.0, (len(pred) + len(ref)) / 400.0)
            moe = max(0.01, 0.15 * (1.0 - conf))
        return MetricResult(
            value=float(val) if pd.notna(val) else None,
            latency_ms=latency,
            confidence=conf,
            margin_of_error=moe,
        )
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        return MetricResult(latency_ms=latency, error=str(exc))


# ---------------------------------------------------------------------------
# Ranking metrics via Ranx
# ---------------------------------------------------------------------------

class RankingMetrics:
    """Corpus-level ranking metrics (MRR, MAP, nDCG@k)."""

    def __init__(
        self,
        k_values: Sequence[int] = (1, 3, 5, 10),
        relevance_threshold: float = 0.25,
    ):
        self.k_values = list(k_values)
        self.relevance_threshold = relevance_threshold

    def compute(
        self,
        rows: List[Dict[str, Any]],
        is_relevant_fn,
    ) -> Dict[str, MetricResult]:
        metric_names = (
            ["mrr", "map"]
            + [f"ndcg@{k}" for k in self.k_values]
            + [f"map@{k}" for k in self.k_values]
        )
        empty = {m: MetricResult(value=None) for m in metric_names}
        if not HAS_RANX:
            return empty

        from .data_prep import parse_generation_chunks  # local import to avoid cycles

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
                if is_relevant_fn(
                    str(c.get("text", "")),
                    gold_ctx,
                    gold_ans,
                    threshold=self.relevance_threshold,
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
                    value=(
                        float(results.get(m, np.nan))
                        if not np.isnan(results.get(m, np.nan))
                        else None
                    ),
                    latency_ms=latency / len(metric_names),
                    confidence=0.85,  # ranking metrics are deterministic given qrels
                    margin_of_error=0.02,
                )
                for m in metric_names
            }
        except Exception as e:
            print(f"[warn] Ranx evaluation failed: {e}", file=sys.stderr)
            return {
                m: MetricResult(
                    latency_ms=latency / len(metric_names), error=str(e)
                )
                for m in metric_names
            }


# ---------------------------------------------------------------------------
# Ragas LLM metric runners
# ---------------------------------------------------------------------------

class RagasMetricRunner:
    """Instrumented async runner for Ragas collection metrics."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enabled_metrics: Optional[Dict[str, bool]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not HAS_OPENAI or not HAS_RAGAS:
            raise RuntimeError("OpenAI client and RAGAS are required for LLM metrics")

        self.model_name = model_name
        self.enabled = enabled_metrics or {}

        # Prefer explicit credentials from llm.yaml over bare env lookup.
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        if not api_key and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "No LLM API key available. Set llm.config_path / llm.entry in "
                "evaluation.yaml (pointing at configs/llm.yaml), or export "
                "OPENAI_API_KEY."
            )

        client = AsyncOpenAI(**client_kwargs)
        evaluation_llm = llm_factory(
            model=model_name,
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        embeddings = embedding_factory(
            "openai",
            model=embedding_model,
            client=client,
        )

        all_scorers = {
            "context_precision": ContextPrecision(llm=evaluation_llm),
            "context_recall": ContextRecall(llm=evaluation_llm),
            "context_entities_recall": ContextEntityRecall(llm=evaluation_llm),
            "noise_sensitivity": NoiseSensitivity(llm=evaluation_llm),
            "response_relevancy": AnswerRelevancy(
                llm=evaluation_llm, embeddings=embeddings
            ),
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
        # Keep only enabled metrics (default True if key absent)
        self.scorers = {
            k: v
            for k, v in all_scorers.items()
            if self.enabled.get(k, True)
        }

    async def safe_ascore(self, scorer, **kwargs) -> MetricResult:
        input_toks = estimate_kwargs_tokens(kwargs, model=self.model_name)
        t0 = time.perf_counter()
        try:
            result = await scorer.ascore(**kwargs)
            latency = (time.perf_counter() - t0) * 1000.0
            value = (
                float(result.value)
                if result is not None and result.value is not None
                else None
            )
            output_toks = 0
            if hasattr(result, "reason") and isinstance(result.reason, str):
                output_toks = count_tokens_tiktoken(result.reason, self.model_name)
            elif hasattr(result, "explanation") and isinstance(
                getattr(result, "explanation"), str
            ):
                output_toks = count_tokens_tiktoken(
                    result.explanation, self.model_name
                )
            # Heuristic confidence / margin for LLM judges
            conf = 0.75 if value is not None else None
            moe = 0.08 if value is not None else None
            return MetricResult(
                value=value,
                latency_ms=latency,
                input_tokens=input_toks,
                output_tokens=output_toks,
                confidence=conf,
                margin_of_error=moe,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            print(
                f"  [warn] {scorer.__class__.__name__} failed: {exc}",
                file=sys.stderr,
            )
            return MetricResult(
                latency_ms=latency,
                input_tokens=input_toks,
                error=str(exc),
            )

    async def score_row(
        self,
        question: str,
        response: str,
        reference: str,
        contexts: List[str],
    ) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        scorers = self.scorers

        async def run(name: str, **kw):
            if name not in scorers:
                return
            results[name] = await self.safe_ascore(scorers[name], **kw)

        await run(
            "context_precision",
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        await run(
            "context_recall",
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        await run(
            "context_entities_recall",
            reference=reference,
            retrieved_contexts=contexts,
        )
        await run(
            "noise_sensitivity",
            user_input=question,
            response=response,
            reference=reference,
            retrieved_contexts=contexts,
        )
        await run(
            "response_relevancy",
            user_input=question,
            response=response,
        )
        await run(
            "faithfulness",
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
        )
        await run(
            "answer_accuracy",
            user_input=question,
            response=response,
            reference=reference,
        )
        await run(
            "context_relevance",
            user_input=question,
            retrieved_contexts=contexts,
        )
        await run(
            "response_groundedness",
            response=response,
            retrieved_contexts=contexts,
        )
        await run(
            "factual_correctness",
            response=response,
            reference=reference,
        )
        await run(
            "nonllm_string_similarity",
            response=response,
            reference=reference,
        )
        await run("ragas_bleu", response=response, reference=reference)
        await run("ragas_chrf", response=response, reference=reference)
        await run("ragas_rouge", response=response, reference=reference)

        return results
