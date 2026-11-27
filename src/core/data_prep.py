"""Data preparation helpers shared by evaluation pipelines."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    sample[["generation_chunk_scores", "generation_chunks"]] = sample[
        "generation_chunks"
    ].apply(extract_chunks_legacy)
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
