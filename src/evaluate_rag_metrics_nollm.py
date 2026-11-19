#!/usr/bin/env python3
"""
RAG Evaluation CLI (no-LLM)

Computes:
  - Per-row lexical metrics: BLEU, CHRF, ROUGE-L
  - Ranking metrics (via ranx): MRR, MAP, NDCG@k, MAP@k
  - Optional RAGAS lexical metrics: BleuScore, CHRFScore, RougeScore
    (and SemanticSimilarity when sentence-transformers is available)

Input: a questions_with_answers.csv produced by a RAG pipeline.
Required columns: question, answer, generation_answer, generation_chunks
Optional column: chunks (gold context text used for relevance judgments)

Usage:
  python evaluate_rag_metrics_nollm.py -i questions_with_answers.csv
  python evaluate_rag_metrics_nollm.py -i input.csv -o metrics.csv --limit 20
  python evaluate_rag_metrics_nollm.py -i input.csv --no-ranking
  python evaluate_rag_metrics_nollm.py -i input.csv --relevance-threshold 0.3

Requires:
  pip install pandas numpy
  # optional:
  pip install ragas datasets ranx nltk sentence-transformers
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

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
    from ragas.metrics.collections import BleuScore, CHRFScore, RougeScore

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


# ---------------------------------------------------------------------------
# Ranking metrics via Ranx
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    rows: List[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5, 10),
    relevance_threshold: float = 0.25,
) -> Dict[str, float]:
    """
    Build Qrels / Run from the dataset using the relevance heuristic and
    compute MRR, NDCG@k, MAP@k with Ranx.
    """
    metric_names = (
        ["mrr", "map"]
        + [f"ndcg@{k}" for k in k_values]
        + [f"map@{k}" for k in k_values]
    )
    if not HAS_RANX:
        return {m: np.nan for m in metric_names}

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

    if not qrels_dict or not run_dict:
        return {m: np.nan for m in metric_names}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    try:
        results = ranx_evaluate(qrels, run, metric_names, make_comparable=True)
        return {m: float(results.get(m, np.nan)) for m in metric_names}
    except Exception as e:
        print(f"[warn] Ranx evaluation failed: {e}", file=sys.stderr)
        return {m: np.nan for m in metric_names}


# ---------------------------------------------------------------------------
# RAGAS evaluation (batch, non-LLM metrics only)
# ---------------------------------------------------------------------------

def build_ragas_dataset(df: pd.DataFrame) -> "Dataset":
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
                "reference": str(row.get("answer") or ""),
            }
        )
    return Dataset.from_list(records)


def run_ragas_metrics(
    dataset: "Dataset",
    metrics_to_run: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run selected non-LLM RAGAS metrics. Returns a flat dict of aggregate scores.
    """
    if not HAS_RAGAS:
        return {}

    available: Dict[str, Any] = {
        "bleu_score": lambda: BleuScore(),
        "chrf_score": lambda: CHRFScore(),
        "rouge_score": lambda: RougeScore(),
    }

    # Optional semantic similarity via local sentence-transformers
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            from ragas.metrics import SemanticSimilarity  # type: ignore

            emb_model = SentenceTransformer("all-MiniLM-L6-v2")

            class _LocalEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return emb_model.encode(texts, show_progress_bar=False).tolist()

                def embed_query(self, text: str) -> List[float]:
                    return emb_model.encode([text], show_progress_bar=False)[0].tolist()

            available["semantic_similarity"] = lambda: SemanticSimilarity(
                embeddings=_LocalEmbeddings()
            )
        except Exception as e:
            print(f"[warn] Could not initialise SemanticSimilarity: {e}", file=sys.stderr)

    if metrics_to_run is None:
        metrics_to_run = list(available.keys())

    selected = []
    for name in metrics_to_run:
        key = name.lower().replace(" ", "_").replace("-", "_")
        factory = available.get(key)
        if factory is None:
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

    print(
        f"[info] Running RAGAS metrics: "
        f"{[getattr(m, 'name', type(m).__name__) for m in selected]}"
    )

    try:
        result = evaluate(dataset, metrics=selected)
        if hasattr(result, "scores"):
            return {k: float(v) for k, v in result.scores.items() if v is not None}
        if isinstance(result, dict):
            return {
                k: float(v)
                for k, v in result.items()
                if isinstance(v, (int, float, np.floating))
            }
        try:
            return {
                k: float(v)
                for k, v in dict(result).items()
                if isinstance(v, (int, float, np.floating))
            }
        except Exception:
            print(f"[warn] Unexpected RAGAS result type: {type(result)}", file=sys.stderr)
            return {}
    except Exception as e:
        print(
            f"[warn] Classic RAGAS evaluate failed ({e}); trying per-row scoring ...",
            file=sys.stderr,
        )

    # Per-row fallback for metrics that support .score(response=..., reference=...)
    scores_acc: Dict[str, List[float]] = {}
    for i in range(len(dataset)):
        row = dataset[i]
        for m in selected:
            name = getattr(m, "name", type(m).__name__)
            try:
                res = m.score(
                    response=row.get("answer", ""),
                    reference=row.get("ground_truth", ""),
                )
                val = float(res.value) if hasattr(res, "value") else float(res)
                scores_acc.setdefault(name, []).append(val)
            except Exception:
                pass
    return {k: float(np.nanmean(v)) for k, v in scores_acc.items() if v}


# ---------------------------------------------------------------------------
# Per-row lexical metrics
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
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RAG metrics (no-LLM) on a questions_with_answers CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input CSV (must contain question, answer, generation_answer, generation_chunks)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_metrics.csv next to the input.",
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
        help="Subset of RAGAS metrics to run (default: all available non-LLM). "
        "Examples: bleu_score chrf_score rouge_score semantic_similarity",
    )
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
    p.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS batch metrics",
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
        ranking_scores = compute_ranking_metrics(
            rows_as_dicts,
            relevance_threshold=args.relevance_threshold,
        )
        for k, v in ranking_scores.items():
            if np.isnan(v):
                print(f"       {k}: nan")
            else:
                print(f"       {k}: {v:.4f}")
    else:
        reason = "disabled by --no-ranking" if args.no_ranking else "ranx not installed"
        print(f"[info] Skipping ranking metrics ({reason})")

    # ------------------------------------------------------------------
    # 3. RAGAS metrics (batch, non-LLM only)
    # ------------------------------------------------------------------
    ragas_scores: Dict[str, float] = {}
    if not args.no_ragas and HAS_RAGAS:
        print("[info] Building RAGAS dataset ...")
        ragas_ds = build_ragas_dataset(df)
        ragas_scores = run_ragas_metrics(
            ragas_ds,
            metrics_to_run=args.metrics,
        )
        for k, v in sorted(ragas_scores.items()):
            if isinstance(v, float) and not np.isnan(v):
                print(f"       {k}: {v:.4f}")
            else:
                print(f"       {k}: {v}")
    else:
        reason = "disabled by --no-ragas" if args.no_ragas else "RAGAS not installed"
        print(f"[info] Skipping RAGAS metrics ({reason})")

    # ------------------------------------------------------------------
    # 4. Aggregate summary + write output
    # ------------------------------------------------------------------
    for name, val in {**ranking_scores, **ragas_scores}.items():
        df[f"corpus_{name}"] = val

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
        out_path = args.input.with_name(args.input.stem + "_nollm.csv")

    metric_cols = [c for c in df.columns if c.startswith("metric_") or c.startswith("corpus_")]
    original_cols = [c for c in df.columns if c not in metric_cols]
    out_df = df[original_cols + metric_cols].copy()

    out_df.to_csv(out_path, index=False, quoting=1)  # QUOTE_ALL
    print(
        f"[info] Wrote detailed results "
        f"({len(out_df)} rows, {len(out_df.columns)} cols) → {out_path}"
    )

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
