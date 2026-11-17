#!/usr/bin/env python3
"""
CLI Question Generator for RAG Failure Modes

Converts the Jupyter notebook "Question generator for failure modes"
into a command-line application.

Usage:
  python generate_qa_set.py --input chunks.csv --output-dir ./output --source Laudon_MIS_17th.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional LiteLLM import (only required when actually generating questions)
# ---------------------------------------------------------------------------
try:
    import litellm
except ImportError:
    litellm = None  # type: ignore


# ===========================================================================
# Sampling helpers (from notebook)
# ===========================================================================

def sample_stratified(
    df: pd.DataFrame,
    num_entries: int,
    stratification_base: Union[str, List[str]],
) -> pd.DataFrame:
    """
    Perform stratified sampling on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    num_entries : int
        Desired number of samples (total size of the returned sample).
    stratification_base : str or list of str
        Column name (or list of column names) used to define the strata.

    Returns
    -------
    pd.DataFrame
        A stratified sample of approximately `num_entries` rows.
    """
    if not isinstance(stratification_base, (str, list, tuple)):
        raise TypeError(
            "stratification_base must be a column name (str) or a list of column names"
        )

    if isinstance(stratification_base, str):
        stratification_base = [stratification_base]

    missing = [col for col in stratification_base if col not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")

    if num_entries <= 0:
        raise ValueError("num_entries must be a positive integer")

    if num_entries >= len(df):
        return df.copy()

    strata_counts = df.groupby(stratification_base).size()
    strata_proportions = strata_counts / len(df)

    sample_sizes = (strata_proportions * num_entries).apply(np.floor).astype(int)
    remainder = num_entries - sample_sizes.sum()

    if remainder > 0:
        fractional_parts = (strata_proportions * num_entries) - sample_sizes
        extra_idx = fractional_parts.nlargest(remainder).index
        sample_sizes.loc[extra_idx] += 1

    samples = []
    for stratum, size in sample_sizes.items():
        if size == 0:
            continue
        if len(stratification_base) == 1:
            mask = df[stratification_base[0]] == stratum
        else:
            mask = True
            for col, val in zip(
                stratification_base, stratum if isinstance(stratum, tuple) else [stratum]
            ):
                mask &= df[col] == val

        stratum_df = df[mask]
        size = min(size, len(stratum_df))
        samples.append(stratum_df.sample(n=size, replace=False))

    if not samples:
        return df.iloc[0:0].copy()

    result = pd.concat(samples, axis=0)
    return result.sample(frac=1).reset_index(drop=True)


def sample_stratified_percentage(
    df: pd.DataFrame,
    coverage: float,
    stratification_base: str,
) -> pd.DataFrame:
    """
    Sample a fraction of the DataFrame while selecting almost equal numbers
    of rows from every group of the stratification column.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be a float in the interval (0, 1]")

    if stratification_base not in df.columns:
        raise ValueError(f"Column '{stratification_base}' not found in DataFrame")

    if df.empty:
        return df.copy()

    n_total = max(1, int(round(len(df) * coverage)))
    groups = df[stratification_base].unique()
    n_groups = len(groups)

    if n_groups == 0:
        return df.copy()

    base = n_total // n_groups
    remainder = n_total % n_groups

    sampled_parts = []
    for i, group_value in enumerate(groups):
        group_df = df[df[stratification_base] == group_value]
        target = base + (1 if i < remainder else 0)
        n_to_sample = min(target, len(group_df))
        if n_to_sample > 0:
            sampled_parts.append(group_df.sample(n=n_to_sample, random_state=None))

    if not sampled_parts:
        return df.iloc[0:0].copy()

    return pd.concat(sampled_parts, ignore_index=True)


def sample_multiple(
    df: pd.DataFrame,
    num_chunks: int,
    num_entries: int,
    column: str,
    *,
    replace: bool = False,
) -> pd.DataFrame:
    """
    Sample groups of rows from a DataFrame and return a summary DataFrame.

    For each group, the values of the specified column are collected into a list.
    Also collects the corresponding chapter values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if num_chunks < 1:
        raise ValueError("num_chunks must be a positive integer")
    if num_entries < 1:
        raise ValueError("num_entries must be a positive integer")
    if not replace and num_chunks > len(df):
        raise ValueError(
            f"num_chunks ({num_chunks}) cannot exceed the number of rows "
            f"in df ({len(df)}) when replace=False"
        )

    results = []
    for _ in range(num_entries):
        sampled = df.sample(n=num_chunks, replace=replace)

        results.append(
            {
                "num_chunks": num_chunks,
                "source": sampled["source"].astype(str).tolist(),
                "chunks": sampled[column].astype(str).tolist(),
                "chapter": sampled["chapter"].astype(str).tolist(),
            }
        )

    return pd.DataFrame(results)


# ===========================================================================
# Content sampling
# ===========================================================================

def perform_content_sampling(
    chunks_df: pd.DataFrame,
    num_entries: int = 10,
) -> pd.DataFrame:
    """
    Run the four sampling strategies described in the notebook and merge them.
    """
    # --- Basic ---
    pdf_coverage = sample_stratified(
        df=chunks_df,
        num_entries=num_entries,
        stratification_base="chapter",
    )
    pdf_coverage = pdf_coverage.copy()
    pdf_coverage["type"] = "basic"

    # --- Information Integration (multi-hop) ---
    pdf_info_integration = sample_multiple(
        chunks_df,
        num_chunks=2,
        num_entries=num_entries,
        column="chunks",
    )
    pdf_info_integration["type"] = "information_integration"

    # --- Negative / Unsupported questions ---
    pdf_unsupported = sample_stratified(
        chunks_df,
        num_entries=num_entries,
        stratification_base="chunks",
    )
    pdf_unsupported = pdf_unsupported.copy()
    pdf_unsupported["type"] = "negative_unsupported_questions"

    # --- Complexity (Flesch-oriented) ---
    pdf_complexity = sample_stratified(
        chunks_df,
        num_entries=num_entries,
        stratification_base="chunks",
    )
    pdf_complexity = pdf_complexity.copy()
    pdf_complexity["type"] = "complexity"

    # --- Merge ---
    # Align columns: basic / unsupported / complexity have scalar chapter & chunks
    # while info_integration has lists.
    frames = [
        pdf_coverage[["type", "source", "chapter", "chunks"]],
        pdf_info_integration[["type", "source", "chapter", "chunks"]],
        pdf_unsupported[["type", "source", "chapter", "chunks"]],
        pdf_complexity[["type", "source", "chapter", "chunks"]],
    ]
    pdf_all = pd.concat(frames, ignore_index=True)
    return pdf_all


# ===========================================================================
# Question generation
# ===========================================================================

GENERIC_PROMPT = """
You will be provided instruction on how to create a question. Generate a question and the answer to the question in JSON format, like this:

{{
    "question": <question generated from context>,
    "answer": <answer comes here>
}}

Instruction: {instruction}

Context: {context}
"""

QUESTION_PROMPTS: Dict[str, str] = {
    "basic": "Generate a question that can be answered with the provided context.",
    "information_integration": (
        "In the provided context there are 2 passages separated by | character, "
        "generate a question that can be answered only with both passages."
    ),
    "negative_unsupported_questions": """
You are an expert test-case generator for Retrieval-Augmented Generation (RAG) systems.

Your task is to create NEGATIVE TEST CASES of the type “Unsupported questions” (also called rejection-type or out-of-knowledge-base queries).

### Goal
Generate questions whose answers are **not** contained in the provided document(s).  
The correct expected behavior of a well-behaved RAG system is to refuse to answer and explicitly state that the information is insufficient (e.g., “Insufficient information to answer this question”, “I don’t know”, or an equivalent clear rejection).

These tests are intended to detect hallucinations: the system answering with plausible but unsupported content instead of acknowledging the knowledge gap.

### Input
You will receive chunks of context.  
Treat the entire provided content as the only knowledge base that the RAG system will have access to.

### Generation rules
1. The question must ask for information, facts, numbers, procedures, definitions, comparisons, or details that are **completely absent** from the given document(s).
2. The question should look realistic and natural — it can be thematically related to the document’s domain (so that retrieval might still pull some chunks) but the specific information requested must not be present.
3. Do **not** create questions that can be answered (even partially) from the document.
4. Prefer open-ended questions that would normally require a factual or explanatory answer (avoid pure yes/no questions).
5. Make the questions clear, unambiguous, and self-contained.
6. Vary the difficulty and style: some can be simple factual questions, others can request synthesis, specific numbers, causes, recommendations, etc., as long as the answer is unsupported.

### Two complementary strategies (use both)
- **Random / fabricated**: Invent plausible questions that simply have no corresponding information in the document.
- **Cross-document / topic-shift**: If multiple documents are provided, generate a question that would be answerable from one document but is asked while only another (unrelated) document is available. If only one document is given, create a question from a closely related but different topic.
""",
    "complexity": """You are an expert test-case generator for Retrieval-Augmented Generation (RAG) systems.

Your task is to create COMPLEXITY-ORIENTED TEST CASES that target the most difficult-to-read passages in a document, following the Flesch Reading Ease prioritization strategy used in RAG-TESTER.

### Goal
Generate questions based exclusively on the **most complex** passages of the provided document.  
Complex passages (low Flesch Reading Ease score) are harder for language models to parse, interpret, and synthesize. These tests are designed to expose failures related to comprehension deficits, incomplete extraction, or misinterpretation of dense, technical, or poorly readable text.

### Background on Flesch Reading Ease
- Score ranges from 0 to 100.
- Lower scores = higher complexity / more difficult to read.
- Typical interpretation:
  - 0–30  → Very difficult (academic, legal, scientific)
  - 30–50 → Difficult
  - 50–60 → Fairly difficult
  - 60–70 → Standard
  - 70+   → Easy

You should prefer (or be given) passages with the lowest Flesch scores.

### Input
You will receive one or more document chunks that have already been ranked by increasing Flesch Reading Ease score (most complex first).  
Each chunk should be used for at most one question.

### Generation rules
1. Create one question per provided complex chunk.
2. The question must be fully answerable from the content of that specific chunk (positive test).
3. Prefer questions that require careful reading, synthesis of dense information, interpretation of technical language, numbers, nested clauses, or subtle relationships present in the complex text.
4. Questions must be clear, unambiguous, and cannot be answered with a simple “yes” or “no”.
5. Do not invent information that is not present in the chunk.
6. The expected answer should be a faithful and complete extraction or synthesis of the relevant information from the chunk.

### Example
Complex chunk (low Flesch):  
“Notwithstanding the aforementioned provisions of subsection (b)(3)(A), the Secretary may, upon a finding that extraordinary circumstances exist which would render strict compliance with the requirements of this paragraph impracticable or contrary to the public interest, waive any such requirement for a period not to exceed 180 days…”

Good question:  
“Under what conditions and for how long may the Secretary waive the requirements of subsection (b)(3)(A)?”

Expected answer should accurately reflect the waiver conditions and the 180-day limit.

Now generate the requested number of high-quality complexity-oriented test cases based on the most difficult passages provided.""",
}


class GeneratedQuestionAnswer(BaseModel):
    question: str
    answer: str


def call_llm_json(
    message: str,
    model: str,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Call the LLM and parse a structured JSON response."""
    if litellm is None:
        raise RuntimeError(
            "litellm is not installed. Install it with: pip install litellm"
        )

    start = time.perf_counter()

    response = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": message}],
        response_format=GeneratedQuestionAnswer,
    )

    elapsed = time.perf_counter() - start

    parsed = GeneratedQuestionAnswer.model_validate_json(
        response.choices[0].message.content
    )

    return {
        "question": parsed.question,
        "answer": parsed.answer,
        "model": model,
        "input_token_count": response.usage.prompt_tokens,
        "output_token_count": response.usage.completion_tokens,
        "elapsed_time": elapsed,
    }


def generate_questions(
    df: pd.DataFrame,
    model: str,
    max_tokens: int = 1024,
) -> pd.DataFrame:
    """Generate a question/answer pair for every row in the dataframe."""
    df = df.copy()
    df["question"] = ""
    df["answer"] = ""
    df["model"] = ""
    df["input_token_count"] = 0
    df["output_token_count"] = 0
    df["elapsed_time"] = 0.0

    for i in tqdm(range(len(df)), desc="Generating questions"):
        row = df.iloc[i]
        if isinstance(row["chunks"], list):
            context = "|".join(row["chunks"])
        else:
            context = str(row["chunks"])

        prompt_type = row["type"]
        instruction = QUESTION_PROMPTS.get(prompt_type, QUESTION_PROMPTS["basic"])

        message = GENERIC_PROMPT.format(instruction=instruction, context=context)

        try:
            response = call_llm_json(message, model=model, max_tokens=max_tokens)
            df.loc[df.index[i], "question"] = response.get("question", "")
            df.loc[df.index[i], "answer"] = response.get("answer", "")
            df.loc[df.index[i], "model"] = response.get("model", "")
            df.loc[df.index[i], "input_token_count"] = response.get(
                "input_token_count", 0
            )
            df.loc[df.index[i], "output_token_count"] = response.get(
                "output_token_count", 0
            )
            df.loc[df.index[i], "elapsed_time"] = response.get("elapsed_time", 0.0)
        except Exception as exc:
            print(f"\n[ERROR] Row {i} ({prompt_type}): {exc}", file=sys.stderr)
            df.loc[df.index[i], "question"] = f"[GENERATION_FAILED] {exc}"
            df.loc[df.index[i], "answer"] = ""

    return df


# ===========================================================================
# Config helpers
# ===========================================================================

def load_llm_config(
    config_path: Optional[str],
    entry: str,
) -> str:
    """
    Load model name (and optionally set API key) from a YAML config file.

    Expected structure:
        claude_5_5:
          provider: claude-haiku-5-5   # or any litellm model string
          api_key: sk-...
    """
    if config_path is None:
        # Fall back to environment
        model = os.environ.get("LLM_MODEL")
        if not model:
            raise ValueError(
                "No --llm-config provided and LLM_MODEL environment variable is not set."
            )
        return model

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

    # Set common API keys if present
    if "api_key" in entry_cfg:
        # Heuristic: Anthropic keys start with sk-ant- or are long; OpenAI with sk-
        key = entry_cfg["api_key"]
        if "anthropic" in model.lower() or "claude" in model.lower():
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            os.environ.setdefault("OPENAI_API_KEY", key)

    return model


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a RAG failure-mode question set from pre-chunked text. "
            "Performs stratified sampling (basic, information-integration, "
            "negative-unsupported, complexity) and optionally calls an LLM "
            "to produce question/answer pairs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV (must contain columns 'chunks' and 'chapter').",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where the resulting QA CSV will be written.",
    )
    parser.add_argument(
        "--num-entries",
        "-n",
        type=int,
        default=10,
        help="Number of samples to draw for each failure-mode type.",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Path to llm_config.yaml (optional). If omitted, uses LLM_MODEL env var.",
    )
    parser.add_argument(
        "--llm-entry",
        default="claude_5_5",
        help="Key inside the YAML config that selects the model / API key.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name (litellm format). Takes precedence over config.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for each LLM completion.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only perform sampling + merge; do not call the LLM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility of sampling.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load & validate input CSV
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading CSV: {input_path}")
    chunks_df = pd.read_csv(input_path)

    required_cols = {"source","chunks", "chapter"}
    missing = required_cols - set(chunks_df.columns)
    if missing:
        print(
            f"ERROR: Input CSV is missing required column(s): {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    print(f"  → {len(chunks_df)} rows, columns: {list(chunks_df.columns)}")

    # ------------------------------------------------------------------
    # 2. Content sampling
    # ------------------------------------------------------------------
    print(f"\nPerforming content sampling (n={args.num_entries} per type)...")
    pdf_all = perform_content_sampling(chunks_df, num_entries=args.num_entries)

    print("  Type distribution:")
    print(pdf_all["type"].value_counts().to_string())
    coverage = pdf_all.shape[0] / max(len(chunks_df), 1) * 100
    print(f"  Overall coverage relative to input: {coverage:.2f}%")

    # ------------------------------------------------------------------
    # 3. Question generation (optional)
    # ------------------------------------------------------------------
    if args.skip_generation:
        print("\n--skip-generation set → skipping LLM calls.")
    else:
        # Resolve model
        if args.model:
            model = args.model
        else:
            try:
                model = load_llm_config(args.llm_config, args.llm_entry)
            except Exception as exc:
                print(f"ERROR resolving LLM model: {exc}", file=sys.stderr)
                return 1

        print(f"\nGenerating questions with model: {model}")
        pdf_all = generate_questions(
            pdf_all, model=model, max_tokens=args.max_tokens
        )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a clean filename
    source = pdf_all.loc[0, 'source']
    timestamp = time.strftime("%Y%m%d%H%M")
    out_file = out_dir / f"qa_set_{source}_{timestamp}.csv"

    cols = [
        "type",
        "source",
        "chapter",
        "question",
        "chunks",
        "answer",
        "model",
        "input_token_count",
        "output_token_count",
        "elapsed_time",
    ]
    # Keep only columns that exist (in case generation was skipped)
    cols = [c for c in cols if c in pdf_all.columns]
    pdf_all[cols].to_csv(out_file, index=False)

    size_kb = out_file.stat().st_size / 1024
    print(f"\nSaved → {out_file} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())