#!/usr/bin/env python3
"""
RAG Evaluation CLI
Computes Context Precision, Context Recall, Context Entities Recall,
Context Relevance, MRR, NDCG, MAP, Response Relevancy, Groundedness,
Faithfulness, Answer Accuracy, Factual Correctness, BLEU, CHRF, ROUGE,
Noise Sensitivity, Semantic Similarity and related metrics on a
questions_with_answers.csv produced by a RAG pipeline.

Usage:
  python evaluate_rag_metrics.py --input questions_with_answers.csv --output metrics_results.csv
  python evaluate_rag_metrics.py -i input.csv -o out.csv --api-key sk-... --limit 5
  OPENAI_API_KEY=sk-... python evaluate_rag_metrics.py -i input.csv

  # Same YAML config style as generate_qa_set.py
  python evaluate_rag_metrics.py -i input.csv --llm-config llm_config.yaml --llm-entry claude_5_5
  python evaluate_rag_metrics.py -i input.csv --model anthropic/claude-3-5-sonnet --api-key $ANTHROPIC_API_KEY
  python evaluate_rag_metrics.py -i input.csv --model ollama/llama3 --api-base http://localhost:11434

Requires:
  pip install ragas ranx datasets pandas nltk sentence-transformers openai langchain-openai pyyaml
  # optional, for LiteLLM multi-provider support:
  pip install litellm langchain-community
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------
HAS_RAGAS = False
HAS_RANX = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_NLTK = False

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import (
        AnswerAccuracy,
        AnswerRelevancy,
        BleuScore,
        CHRFScore,
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        FactualCorrectness,
        Faithfulness,
        NoiseSensitivity,
        ResponseGroundedness,
        RougeScore,
        SemanticSimilarity,
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
# Helpers
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
        # Fallback: try json
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []


def extract_contexts(chunks: List[Dict[str, Any]]) -> List[str]:
    """Return ordered list of context texts (highest score first)."""
    if not chunks:
        return []
    # Prefer original order (already ranked); otherwise sort by score desc
    try:
        sorted_c = sorted(
            chunks,
            key=lambda c: float(c.get("score", 0.0)),
            reverse=True,
        )
    except Exception:
        sorted_c = chunks
    return [str(c.get("text", "")).strip() for c in sorted_c if c.get("text")]


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


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


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
        # LCS length
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
            from collections import Counter
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


# ---------------------------------------------------------------------------
# Ranking metrics via Ranx
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    rows: List[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Build Qrels / Run from the dataset using the relevance heuristic and
    compute MRR, NDCG@k, MAP@k with Ranx.
    """
    if not HAS_RANX:
        return {m: np.nan for m in ["mrr", "map"] + [f"ndcg@{k}" for k in k_values] + [f"map@{k}" for k in k_values]}

    qrels_dict: Dict[str, Dict[str, int]] = {}
    run_dict: Dict[str, Dict[str, float]] = {}

    for i, row in enumerate(rows):
        qid = f"q{i}"
        gold_ctx = str(row.get("chunks") or "")
        gold_ans = str(row.get("answer") or "")
        gen_chunks = parse_generation_chunks(row.get("generation_chunks"))

        # Relevance judgments
        rels = {}
        for c in gen_chunks:
            cid = str(c.get("chunk_id", id(c)))
            if is_relevant_chunk(str(c.get("text", "")), gold_ctx, gold_ans):
                rels[cid] = 1
        if rels:
            qrels_dict[qid] = rels

        # Retrieved ranking (score descending)
        ranking = {}
        for c in sorted(gen_chunks, key=lambda x: float(x.get("score", 0.0)), reverse=True):
            cid = str(c.get("chunk_id", id(c)))
            ranking[cid] = float(c.get("score", 0.0))
        if ranking:
            run_dict[qid] = ranking

    if not qrels_dict or not run_dict:
        return {m: np.nan for m in ["mrr", "map"] + [f"ndcg@{k}" for k in k_values]}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    metrics = ["mrr", "map"] + [f"ndcg@{k}" for k in k_values] + [f"map@{k}" for k in k_values]
    try:
        results = ranx_evaluate(qrels, run, metrics, make_comparable=True)
        return {m: float(results.get(m, np.nan)) for m in metrics}
    except Exception as e:
        print(f"[warn] Ranx evaluation failed: {e}", file=sys.stderr)
        return {m: np.nan for m in metrics}


# ---------------------------------------------------------------------------
# RAGAS evaluation (batch)
# ---------------------------------------------------------------------------

def build_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Prepare a HuggingFace Dataset for RAGAS."""
    records = []
    for _, row in df.iterrows():
        contexts = extract_contexts(parse_generation_chunks(row.get("generation_chunks")))
        records.append(
            {
                "question": str(row.get("question") or ""),
                "answer": str(row.get("generation_answer") or ""),
                "contexts": contexts,
                "ground_truth": str(row.get("answer") or ""),
                "reference": str(row.get("answer") or ""),  # alias used by some metrics
            }
        )
    return Dataset.from_list(records)


def _parse_provider_model(model: str) -> Tuple[str, str]:
    """
    Split a litellm-style 'provider/model' string.
    Returns (provider, model_name). Defaults provider to 'openai'.
    """
    if "/" in model and not model.startswith("gpt-") and not model.startswith("o1"):
        # e.g. anthropic/claude-3-5-sonnet, azure/gpt-4o, ollama/llama3
        provider, _, rest = model.partition("/")
        # litellm sometimes uses 'openai/gpt-4o' – keep as openai
        return provider.lower(), rest or model
    return "openai", model


def _build_openai_client(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
):
    """Build a sync openai.OpenAI client (works with any OpenAI-compatible endpoint)."""
    from openai import OpenAI

    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or "no-key"
    )
    base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")
    kwargs: Dict[str, Any] = {"api_key": key}
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def get_embeddings(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: str = "text-embedding-3-small",
):
    """
    Return a modern RAGAS embeddings instance via embedding_factory.

    Prefers OpenAI-compatible embeddings (required by collections metrics).
    Falls back to None if the factory / client cannot be created.
    """
    try:
        from ragas.embeddings.base import embedding_factory
    except ImportError:
        try:
            from ragas.embeddings import embedding_factory
        except ImportError:
            print("[warn] ragas embedding_factory not available", file=sys.stderr)
            return None

    try:
        client = _build_openai_client(api_key=api_key, api_base=api_base)
        emb = embedding_factory(
            provider="openai",
            model=model,
            client=client,
            interface="modern",
            base_url=api_base or os.environ.get("OPENAI_API_BASE"),
        )
        print(f"[info] Using modern RAGAS embeddings model={model!r}", file=sys.stderr)
        return emb
    except Exception as e:
        print(f"[warn] Could not create modern embeddings: {e}", file=sys.stderr)
        return None


def load_llm_config(
    config_path: Optional[str],
    entry: str,
) -> str:
    """
    Load model name (and optionally set API key) from a YAML config file.

    Same contract as generate_qa_set.py. Expected structure::

        claude_5_5:
          provider: claude-haiku-5-5   # or any litellm model string
          api_key: sk-...

    Returns the resolved model string. Sets ANTHROPIC_API_KEY / OPENAI_API_KEY
    in the environment when the config provides an api_key.
    """
    if config_path is None:
        model = os.environ.get("LLM_MODEL")
        if not model:
            raise ValueError(
                "No --llm-config provided and LLM_MODEL environment variable is not set."
            )
        return model

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load --llm-config. Install with: pip install pyyaml"
        )

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
    model = entry_cfg.get("provider") or entry_cfg.get("model")
    if not model:
        raise ValueError(f"No 'provider' or 'model' key under '{entry}' in config.")

    # Set common API keys if present (same heuristic as generate_qa_set.py)
    if "api_key" in entry_cfg:
        key = entry_cfg["api_key"]
        if "anthropic" in str(model).lower() or "claude" in str(model).lower():
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            os.environ.setdefault("OPENAI_API_KEY", key)

    # Optional api_base from config
    if entry_cfg.get("api_base") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = str(entry_cfg["api_base"])
    if entry_cfg.get("api_base") and not os.environ.get("LITELLM_API_BASE"):
        os.environ["LITELLM_API_BASE"] = str(entry_cfg["api_base"])

    return model


def get_llm(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Return a modern RAGAS InstructorLLM via ``llm_factory`` (required by
    collections metrics in RAGAS >= 0.4).

    Uses an OpenAI-compatible client under the hood. For non-OpenAI providers
    (Anthropic, Ollama, Azure, …) either:
      - pass an OpenAI-compatible ``api_base``, or
      - use a litellm-style model string and ``adapter='litellm'``.
    """
    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
    )
    base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")

    if not key and not base:
        return None

    try:
        from ragas.llms import llm_factory
    except ImportError as e:
        print(f"[warn] ragas.llms.llm_factory not available: {e}", file=sys.stderr)
        return None

    provider, model_name = _parse_provider_model(model)

    # Decide adapter: use litellm for non-openai providers or when model looks like provider/model
    adapter = "auto"
    if provider not in ("openai",) or "/" in model:
        # litellm adapter supports 100+ providers; instructor needs a native client
        try:
            import litellm  # noqa: F401
            adapter = "litellm"
        except ImportError:
            adapter = "auto"

    try:
        client = _build_openai_client(api_key=key, api_base=base)
        kwargs: Dict[str, Any] = {
            "model": model_name if adapter != "litellm" else model,
            "provider": provider if adapter != "litellm" else "openai",
            "client": client,
            "adapter": adapter,
            "temperature": temperature,
        }
        # When using litellm adapter, pass the full model string (provider/model)
        if adapter == "litellm":
            kwargs["model"] = model
            kwargs["provider"] = "openai"  # client is still OpenAI-shaped; litellm routes

        llm = llm_factory(**kwargs)
        print(
            f"[info] Using RAGAS llm_factory model={model!r} provider={provider!r} adapter={adapter!r}",
            file=sys.stderr,
        )
        return llm
    except Exception as e:
        print(f"[warn] llm_factory failed ({e}); no LLM available", file=sys.stderr)
        return None


def run_ragas_metrics(
    dataset: Dataset,
    llm=None,
    embeddings=None,
    metrics_to_run: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run selected RAGAS metrics. Returns a flat dict of aggregate scores.
    Metrics that need an LLM are skipped when no LLM is supplied.
    """
    if not HAS_RAGAS:
        return {}

    available = {
        "context_precision": lambda: ContextPrecision(llm=llm) if llm else None,
        "context_recall": lambda: ContextRecall(llm=llm) if llm else None,
        "context_entity_recall": lambda: ContextEntityRecall(llm=llm) if llm else None,
        "context_relevance": lambda: ContextRelevance(llm=llm) if llm else None,
        "faithfulness": lambda: Faithfulness(llm=llm) if llm else None,
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings) if llm else None,
        "response_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings) if llm else None,
        "answer_accuracy": lambda: AnswerAccuracy(llm=llm) if llm else None,
        "factual_correctness": lambda: FactualCorrectness(llm=llm) if llm else None,
        "response_groundedness": lambda: ResponseGroundedness(llm=llm) if llm else None,
        "groundedness": lambda: ResponseGroundedness(llm=llm) if llm else None,
        "noise_sensitivity": lambda: NoiseSensitivity(llm=llm) if llm else None,
        "bleu_score": lambda: BleuScore(),
        "chrf_score": lambda: CHRFScore(),
        "rouge_score": lambda: RougeScore(),
        "semantic_similarity": lambda: SemanticSimilarity(embeddings=embeddings) if embeddings else None,
    }

    if metrics_to_run is None:
        metrics_to_run = list(available.keys())

    selected = []
    for name in metrics_to_run:
        key = name.lower().replace(" ", "_").replace("-", "_")
        factory = available.get(key)
        if factory is None:
            # try fuzzy
            for k, f in available.items():
                if key in k or k in key:
                    factory = f
                    break
        if factory is None:
            continue
        try:
            m = factory()
            if m is not None:
                selected.append(m)
        except Exception as e:
            print(f"[warn] Could not init metric {name}: {e}", file=sys.stderr)

    if not selected:
        print("[warn] No RAGAS metrics could be initialised.", file=sys.stderr)
        return {}

    print(f"[info] Running RAGAS metrics: {[getattr(m, 'name', type(m).__name__) for m in selected]}")
    # RAGAS 0.4 collections metrics use a different evaluate path and often need
    # extra packages (sacrebleu, etc.). We fall back to per-metric .score when
    # the classic evaluate() rejects the objects.
    try:
        result = evaluate(dataset, metrics=selected)
        if hasattr(result, "scores"):
            return {k: float(v) for k, v in result.scores.items() if v is not None}
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        try:
            return {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            print(f"[warn] Unexpected RAGAS result type: {type(result)}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"[warn] Classic RAGAS evaluate failed ({e}); trying per-row scoring ...", file=sys.stderr)

    # Per-row fallback for metrics that support .score(response=..., reference=...)
    scores_acc: Dict[str, List[float]] = {}
    for i in range(len(dataset)):
        row = dataset[i]
        for m in selected:
            name = getattr(m, "name", type(m).__name__)
            try:
                # Most lexical metrics accept response + reference
                res = m.score(response=row.get("answer", ""), reference=row.get("ground_truth", ""))
                val = float(res.value) if hasattr(res, "value") else float(res)
                scores_acc.setdefault(name, []).append(val)
            except Exception:
                # Context metrics need different kwargs; skip silently in fallback
                pass
    return {k: float(np.nanmean(v)) for k, v in scores_acc.items() if v}


# ---------------------------------------------------------------------------
# Per-row lexical metrics (always computed)
# ---------------------------------------------------------------------------

def add_per_row_lexical(df: pd.DataFrame) -> pd.DataFrame:
    """Add BLEU / CHRF / ROUGE-L columns computed row-wise."""
    bleus, chrfs, rouges = [], [], []
    for _, row in df.iterrows():
        pred = str(row.get("generation_answer") or "")
        ref = str(row.get("answer") or "")
        bleus.append(compute_bleu(pred, ref))
        chrfs.append(compute_chrf(pred, ref))
        rouges.append(compute_rouge_l(pred, ref))
    out = df.copy()
    out["metric_bleu"] = bleus
    out["metric_chrf"] = chrfs
    out["metric_rouge_l"] = rouges
    return out


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RAG metrics (RAGAS + Ranx) on a questions_with_answers CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input CSV (must contain question, answer, generation_answer, generation_chunks, chunks)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_metrics.csv next to the input.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider (or set OPENAI_API_KEY / LITELLM_API_KEY / ANTHROPIC_API_KEY). "
             "Overrides any key found in --llm-config.",
    )
    p.add_argument(
        "--llm-config",
        default=None,
        help="Path to llm_config.yaml (same format as generate_qa_set.py). "
             "If omitted, uses --model / LLM_MODEL env var.",
    )
    p.add_argument(
        "--llm-entry",
        default="claude_5_5",
        help="Key inside the YAML config that selects the model / API key.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the model name (litellm format). Takes precedence over --llm-config. "
             "Examples: gpt-4o-mini, anthropic/claude-3-5-sonnet, ollama/llama3, azure/gpt-4o.",
    )
    p.add_argument(
        "--api-base",
        default=None,
        help="Optional API base URL (OpenAI-compatible proxy, Ollama, vLLM, Azure, …). "
             "Also read from OPENAI_API_BASE / LITELLM_API_BASE or the YAML config.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for testing)",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Subset of metrics to run (default: all available). "
             "Examples: context_precision faithfulness bleu_score mrr",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip all LLM-dependent metrics even if a key is present",
    )
    p.add_argument(
        "--no-ranking",
        action="store_true",
        help="Skip Ranx ranking metrics (MRR / NDCG / MAP)",
    )
    p.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.25,
        help="Token-overlap threshold used to build qrels for ranking metrics",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"[info] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"[info] Loaded {len(df)} rows, columns: {list(df.columns)}")

    required = {"question", "answer", "generation_answer", "generation_chunks"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] Missing required columns: {missing}", file=sys.stderr)
        return 1

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"[info] Limited to first {len(df)} rows")

    # ------------------------------------------------------------------
    # 1. Per-row lexical metrics (BLEU / CHRF / ROUGE-L)
    # ------------------------------------------------------------------
    print("[info] Computing per-row lexical metrics (BLEU, CHRF, ROUGE-L) ...")
    df = add_per_row_lexical(df)

    # ------------------------------------------------------------------
    # 2. Ranking metrics (Ranx)
    # ------------------------------------------------------------------
    ranking_scores: Dict[str, float] = {}
    if not args.no_ranking and HAS_RANX:
        print("[info] Computing ranking metrics with Ranx (MRR, NDCG, MAP) ...")
        rows_as_dicts = df.to_dict(orient="records")
        ranking_scores = compute_ranking_metrics(rows_as_dicts)
        for k, v in ranking_scores.items():
            print(f"       {k}: {v:.4f}" if not np.isnan(v) else f"       {k}: nan")
    else:
        print("[info] Skipping ranking metrics")

    # ------------------------------------------------------------------
    # 3. RAGAS metrics (batch)
    # ------------------------------------------------------------------
    ragas_scores: Dict[str, float] = {}
    if HAS_RAGAS:
        llm = None
        if not args.no_llm:
            # Resolve model the same way as generate_qa_set.py:
            #   1. explicit --model
            #   2. --llm-config / --llm-entry (or LLM_MODEL env)
            #   3. fallback default
            model = args.model
            if model is None:
                try:
                    model = load_llm_config(args.llm_config, args.llm_entry)
                except Exception as exc:
                    if args.llm_config is not None:
                        print(f"[error] Failed to load LLM config: {exc}", file=sys.stderr)
                        return 1
                    # No config and no --model → use a sensible default
                    model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
                    print(f"[info] No --model / --llm-config; defaulting to model={model!r}")

            llm = get_llm(
                api_key=args.api_key,
                model=model,
                api_base=args.api_base,
            )

        # Modern embeddings share the same credentials / base URL
        embeddings = get_embeddings(
            api_key=args.api_key,
            api_base=args.api_base,
        )
        if llm is None:
            print("[info] No LLM available – only non-LLM RAGAS metrics will run")
        if embeddings is None:
            print("[info] No embeddings available – SemanticSimilarity will be skipped")

        print("[info] Building RAGAS dataset ...")
        ragas_ds = build_ragas_dataset(df)

        metrics_filter = args.metrics
        ragas_scores = run_ragas_metrics(
            ragas_ds,
            llm=llm,
            embeddings=embeddings,
            metrics_to_run=metrics_filter,
        )
        for k, v in sorted(ragas_scores.items()):
            print(f"       {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"       {k}: {v}")
    else:
        print("[info] RAGAS not available – skipping its metrics")

    # ------------------------------------------------------------------
    # 4. Aggregate summary row + write output
    # ------------------------------------------------------------------
    # Attach corpus-level scores as constant columns so the CSV stays row-aligned
    for name, val in {**ranking_scores, **ragas_scores}.items():
        col = f"corpus_{name}"
        df[col] = val

    # Summary statistics for the per-row lexical metrics
    summary = {
        "metric_bleu_mean": float(np.nanmean(df["metric_bleu"])),
        "metric_chrf_mean": float(np.nanmean(df["metric_chrf"])),
        "metric_rouge_l_mean": float(np.nanmean(df["metric_rouge_l"])),
        "n_rows": len(df),
    }
    for k, v in summary.items():
        print(f"[summary] {k}: {v}")

    out_path = args.output
    if out_path is None:
        out_path = args.input.with_name(args.input.stem + "_metrics.csv")

    # Keep every original input column plus the metric columns we added.
    # QUOTE_ALL protects multiline fields (chunks, generation_chunks, etc.).
    metric_cols = [c for c in df.columns if c.startswith("metric_") or c.startswith("corpus_")]
    original_cols = [c for c in df.columns if c not in metric_cols]
    out_df = df[original_cols + metric_cols].copy()

    out_df.to_csv(out_path, index=False, quoting=1)  # QUOTE_ALL
    print(f"[info] Wrote detailed results ({len(out_df)} rows, {len(out_df.columns)} cols) → {out_path}")

    # Also write a compact summary JSON next to it
    summary_path = out_path.with_suffix(".summary.json")
    full_summary = {
        "n_rows": len(df),
        "lexical_means": {
            "bleu": summary["metric_bleu_mean"],
            "chrf": summary["metric_chrf_mean"],
            "rouge_l": summary["metric_rouge_l_mean"],
        },
        "ranking": ranking_scores,
        "ragas": ragas_scores,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"[info] Wrote summary → {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
