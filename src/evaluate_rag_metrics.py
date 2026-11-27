#!/usr/bin/env python3
"""
Unified RAG Evaluation CLI (agent-driven)

Loads a questions CSV, invokes the agent defined in configs/agent.yaml for
every row, scores only the metrics marked true in configs/evaluation.yaml,
and writes an enriched results CSV + summary JSON.

Usage (from project root):
  python src/evaluate_rag_metrics.py -i sample_questions.csv
  python src/evaluate_rag_metrics.py -i questions.csv -o out.csv --limit 5
  python src/evaluate_rag_metrics.py -i questions.csv --no-agent
  python src/evaluate_rag_metrics.py -i questions.csv \\
      --agent-config configs/agent.yaml \\
      --eval-config configs/evaluation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap: project root contains configs/ and src/
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = SRC_ROOT.parent                      # .../ (configs sibling)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.agent_runner import AgentRunner
from core.config import AgentConfig, EvaluationConfig
from core.evaluator import OnTheFlyEvaluator


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "On-the-fly RAG metrics evaluation: call agent, then score "
            "enabled metrics with latency, tokens, confidence & margin of error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input questions CSV (must contain a 'question' column)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_evaluated.csv",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows",
    )
    p.add_argument(
        "--agent-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "agent.yaml",
        help="Path to agent.yaml",
    )
    p.add_argument(
        "--eval-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evaluation.yaml",
        help="Path to evaluation.yaml",
    )
    p.add_argument(
        "--no-agent",
        action="store_true",
        help=(
            "Do not call the agent; expect generation_answer "
            "(and optionally generation_chunks) already present in the CSV"
        ),
    )
    p.add_argument(
        "--question-col",
        default="question",
        help="Column name holding the question text",
    )
    p.add_argument(
        "--answer-col",
        default="answer",
        help="Column name holding the gold / reference answer",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"[info] Loading questions from {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"[info] Loaded {len(df)} rows, columns: {list(df.columns)}")

    if args.question_col not in df.columns:
        print(
            f"[error] Missing required column '{args.question_col}'",
            file=sys.stderr,
        )
        return 1

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"[info] Limited to first {len(df)} rows")

    if not args.agent_config.exists():
        print(f"[error] Agent config not found: {args.agent_config}", file=sys.stderr)
        return 1
    if not args.eval_config.exists():
        print(f"[error] Eval config not found: {args.eval_config}", file=sys.stderr)
        return 1

    agent_cfg = AgentConfig.from_yaml(args.agent_config)
    eval_cfg = EvaluationConfig.from_yaml(args.eval_config, base_dir=PROJECT_ROOT)

    print(f"[info] Project root: {PROJECT_ROOT}")
    print(
        f"[info] Enabled metrics: "
        f"{', '.join(eval_cfg.enabled_metric_names()) or '(none)'}"
    )
    if eval_cfg.use_llm:
        key_preview = (
            (eval_cfg.llm_api_key[:8] + "…") if eval_cfg.llm_api_key else "(none)"
        )
        print(
            f"[info] LLM judge: entry={eval_cfg.llm_entry!r} "
            f"provider={eval_cfg.llm_provider!r} model={eval_cfg.llm_model!r} "
            f"api_key={key_preview} base_url={eval_cfg.llm_base_url!r} "
            f"config={eval_cfg.llm_config_path!r}"
        )
        if not eval_cfg.llm_api_key:
            print(
                "[warn] No API key resolved from llm config – LLM metrics will fail "
                "unless OPENAI_API_KEY is set in the environment.",
                flush=True,
            )

    pbar = tqdm(total=len(df), desc="Evaluating", unit="q")

    def on_progress(done: int, total: int, row_ev, elapsed: float) -> None:
        pbar.n = done
        short_q = row_ev.question[:40] + ("…" if len(row_ev.question) > 40 else "")
        pbar.set_postfix(last=short_q, elapsed=f"{elapsed:.1f}s", refresh=True)
        pbar.refresh()

    agent = AgentRunner(agent_cfg)
    evaluator = OnTheFlyEvaluator(
        agent=agent,
        eval_config=eval_cfg,
        progress_callback=on_progress,
    )

    call_agent = not args.no_agent
    if call_agent:
        print("[info] Calling agent on-the-fly for every question …")
    else:
        print("[info] Using pre-existing generation_answer columns (no agent call)")

    result_df = evaluator.evaluate_dataframe(
        df,
        question_col=args.question_col,
        answer_col=args.answer_col,
        call_agent=call_agent,
        limit=None,
    )
    pbar.close()

    out_path = args.output
    if out_path is None:
        out_path = args.input.with_name(args.input.stem + "_evaluated.csv")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer core.Helper.Log if present, else plain write
    try:
        from core.Helper import Log

        with Log.save_file(str(out_path)) as filename:
            result_df.to_csv(filename, index=False, quoting=1)
        print(f"[info] Wrote enriched CSV → {out_path} (via Helper.Log)")
    except Exception:
        result_df.to_csv(out_path, index=False, quoting=1)
        print(f"[info] Wrote enriched CSV → {out_path}")

    # Summary
    skip_suffixes = (
        "_latency_ms",
        "_input_tokens",
        "_output_tokens",
        "_confidence",
        "_margin_of_error",
        "_error",
    )
    meta_cols = {
        "question",
        "answer",
        "generation_answer",
        "agent_latency_ms",
        "agent_success",
        "agent_error",
        "n_contexts",
        "type",
        "source",
        "chapter",
    }
    metric_cols = [
        c
        for c in result_df.columns
        if not any(c.endswith(s) for s in skip_suffixes) and c not in meta_cols
    ]

    summary = {
        "n_rows": len(result_df),
        "agent_success_rate": (
            float(result_df["agent_success"].mean())
            if "agent_success" in result_df
            else None
        ),
        "mean_agent_latency_ms": (
            float(result_df["agent_latency_ms"].mean())
            if "agent_latency_ms" in result_df
            else None
        ),
        "metric_means": {},
        "metric_confidence_means": {},
        "metric_margin_means": {},
    }
    for col in metric_cols:
        if col in result_df and pd.api.types.is_numeric_dtype(result_df[col]):
            summary["metric_means"][col] = (
                float(np.nanmean(result_df[col]))
                if result_df[col].notna().any()
                else None
            )
            conf_col = f"{col}_confidence"
            moe_col = f"{col}_margin_of_error"
            if conf_col in result_df:
                summary["metric_confidence_means"][col] = float(
                    np.nanmean(result_df[conf_col])
                )
            if moe_col in result_df:
                summary["metric_margin_means"][col] = float(
                    np.nanmean(result_df[moe_col])
                )

    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[info] Wrote summary → {summary_path}")

    print("\n[summary] Metric means (± margin of error):")
    for name, val in summary["metric_means"].items():
        if val is None:
            continue
        moe = summary["metric_margin_means"].get(name)
        conf = summary["metric_confidence_means"].get(name)
        moe_s = f" ± {moe:.3f}" if moe is not None else ""
        conf_s = f"  (conf {conf:.2f})" if conf is not None else ""
        print(f"  {name:32s}: {val:.4f}{moe_s}{conf_s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
