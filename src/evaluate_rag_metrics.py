#!/usr/bin/env python3
"""
Evaluate RAG outputs with Ragas metrics (collections API).

Metrics computed per row:
  - Context Precision
  - Context Recall
  - Context Entities Recall
  - Noise Sensitivity
  - Response Relevancy (AnswerRelevancy)
  - Faithfulness
  - Answer Accuracy
  - Context Relevance
  - Response Groundedness

Scores are attached as new columns and the enriched CSV is saved.
"""

import os
import ast
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm
from openai import AsyncOpenAI

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
    BleuScore,
    CHRFScore,
    RougeScore
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_llm_config(
    config_path: Optional[str],
    entry: str,
) -> Tuple[str, str]:
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
    client = AsyncOpenAI()
    return llm_factory(
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_embeddings(model: str = "text-embedding-3-small"):
    """Create embeddings needed by AnswerRelevancy."""
    client = AsyncOpenAI()
    return embedding_factory(
        "openai",
        model=model,
        client=client,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate agent outputs with Ragas metrics and save enriched CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV to evaluate.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path for the output CSV (default: <input>_scored.csv).",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Path to llm_config.yaml",
    )
    parser.add_argument(
        "--llm-entry",
        default="claude_5_5",
        help="Key inside the YAML config that selects the model / API key.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model used by Response Relevancy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only evaluate the first N rows (useful for dry-runs).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def extract_chunks(x) -> pd.Series:
    """Parse generation_chunks into (scores list, texts list)."""
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
        sample["generation_chunks"].apply(extract_chunks)
    )
    # Ensure list-of-str for contexts; drop None entries
    sample["generation_chunks"] = sample["generation_chunks"].apply(
        lambda xs: [t for t in (xs or []) if isinstance(t, str) and t.strip()]
    )
    return sample


# ---------------------------------------------------------------------------
# Metric runners
# ---------------------------------------------------------------------------

async def safe_ascore(scorer, **kwargs) -> Optional[float]:
    """Call scorer.ascore and return .value, or None on failure."""
    try:
        result = await scorer.ascore(**kwargs)
        return float(result.value) if result is not None else None
    except Exception as exc:
        print(f"  [warn] {scorer.__class__.__name__} failed: {exc}", file=sys.stderr)
        return None


async def score_row(
    row: pd.Series,
    scorers: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """Run every metric on a single row and return a dict of scores."""
    user_input = row["question"]
    response = row["generation_answer"] if pd.notna(row["generation_answer"]) else ""
    reference = row["answer"] if pd.notna(row["answer"]) else ""
    contexts: List[str] = row["generation_chunks"] or []

    scores: Dict[str, Optional[float]] = {}

    # Context Precision (needs reference)
    scores["context_precision"] = await safe_ascore(
        scorers["context_precision"],
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Context Recall (needs reference)
    scores["context_recall"] = await safe_ascore(
        scorers["context_recall"],
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Context Entities Recall
    scores["context_entities_recall"] = await safe_ascore(
        scorers["context_entities_recall"],
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Noise Sensitivity (default mode = relevant)
    scores["noise_sensitivity"] = await safe_ascore(
        scorers["noise_sensitivity"],
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=contexts,
    )

    # Response Relevancy (AnswerRelevancy) – needs embeddings
    scores["response_relevancy"] = await safe_ascore(
        scorers["response_relevancy"],
        user_input=user_input,
        response=response,
    )

    # Faithfulness
    scores["faithfulness"] = await safe_ascore(
        scorers["faithfulness"],
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts,
    )

    # Answer Accuracy (Nvidia)
    scores["answer_accuracy"] = await safe_ascore(
        scorers["answer_accuracy"],
        user_input=user_input,
        response=response,
        reference=reference,
    )

    # Context Relevance (Nvidia)
    scores["context_relevance"] = await safe_ascore(
        scorers["context_relevance"],
        user_input=user_input,
        retrieved_contexts=contexts,
    )

    # Response Groundedness (Nvidia)
    scores["response_groundedness"] = await safe_ascore(
        scorers["response_groundedness"],
        response=response,
        retrieved_contexts=contexts,
    )

    # Factual Correctness
    scores["factual_correctness"] = await safe_ascore(
        scorers["factual_correctness"],
        response=response,
        reference=reference,
    )

    # Non-LLM string similarity
    scores["nonllm_string_similarity"] = await safe_ascore(
        scorers["nonllm_string_similarity"],
        response=response,
        reference=reference,
    )

    # BLEU score
    scores["bleu"] = await safe_ascore(
        scorers["bleu"],
        response=response,
        reference=reference,
    )

    # CHRF
    scores["chrf"] = await safe_ascore(
        scorers["chrf"],
        response=response,
        reference=reference,
    )

    # CHRF
    scores["rouge"] = await safe_ascore(
        scorers["rouge"],
        response=response,
        reference=reference,
    )

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()

    # ---- LLM / embeddings setup ----
    if args.llm_config:
        provider, llm_model = load_llm_config(args.llm_config, args.llm_entry)
        print(f"Using provider={provider}, model={llm_model}")
    else:
        print(f"No llm-config provided; using model={llm_model}")

    evaluation_llm = create_evaluation_llm(llm_model, temperature=0.0, max_tokens=2048)
    embeddings = create_embeddings(args.embedding_model)

    # ---- Build scorers once ----
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
        "nonllm_string_similarity": NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN),
        "bleu": BleuScore(),
        "chrf": CHRFScore(),
        "rouge": RougeScore(),
    }

    # ---- Load & prepare data ----
    print(f"Loading CSV: {args.input}")
    raw_df = pd.read_csv(args.input)
    if args.limit is not None:
        raw_df = raw_df.head(args.limit)
        print(f"Limiting evaluation to first {args.limit} rows")

    sample = prepare_dataframe(raw_df)
    n = len(sample)
    print(f"Running evaluation on {n:,} records")

    # Columns that will hold the scores
    metric_columns = list(scorers.keys())

    # Pre-allocate score columns on the original dataframe
    for col in metric_columns:
        raw_df[col] = None

    # ---- Score every row ----
    for idx in tqdm(range(n), desc="Evaluating"):
        row = sample.iloc[idx]
        scores = await score_row(row, scorers)
        for col, val in scores.items():
            raw_df.at[raw_df.index[idx], col] = val

    # ---- Save ----
    if args.output:
        out_path = Path(args.output)
    else:
        in_path = Path(args.input)
        out_path = in_path.with_name(f"{in_path.stem}_scored{in_path.suffix}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_path, index=False)
    print(f"Saved scored CSV to: {out_path}")

    # Quick summary
    print("\nScore summary (mean over non-null):")
    for col in metric_columns:
        mean_val = raw_df[col].mean(skipna=True)
        print(f"  {col:25s}: {mean_val:.4f}" if pd.notna(mean_val) else f"  {col:25s}: n/a")


if __name__ == "__main__":
    asyncio.run(main())