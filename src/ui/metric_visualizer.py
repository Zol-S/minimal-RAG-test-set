"""
Streamlit on-the-fly RAG Evaluation Dashboard

- Loads a questions CSV
- Calls the agent (configs/agent.yaml) for every row
- Evaluates metrics selected in the sidebar checkboxes
- Live tqdm-style progress bar (%, elapsed, remaining)
- Charts update as each row completes
- Shows confidence and margin of error per metric
- Latency line charts (agent / retrieval / generation / end-to-end)
- Mean scores by question type
- Configurable confidence-level margin of error (worst-case σ = 0.5)

Usage (from project root):
  streamlit run src/ui/metric_visualizer.py
  streamlit run src/ui/metric_visualizer.py -- --questions path/to/questions.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap
#   this file lives at:  <PROJECT_ROOT>/src/ui/metric_visualizer.py
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]       # .../src
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.agent_runner import AgentRunner
from core.config import AgentConfig, EvaluationConfig
from core.evaluator import OnTheFlyEvaluator, RowEvaluation


# ---------------------------------------------------------------------------
# Metric catalog for the sidebar checkboxes
# default_on is only a fallback when evaluation.yaml has no metrics block;
# the UI initialises checkboxes from EvaluationConfig.metrics when available.
# ---------------------------------------------------------------------------
METRIC_CATALOG = [
    # (key, label, heavy_hint or None, default_on)
    ("context_precision", "Context Precision", "heavy · many LLM calls", False),
    ("context_recall", "Context Recall", "LLM", True),
    ("context_entities_recall", "Context Entities Recall", "LLM", True),
    ("context_relevance", "Context Relevance", "LLM", True),
    ("faithfulness", "Faithfulness", "heavy · LLM", True),
    ("response_groundedness", "Response Groundedness", "LLM", True),
    ("answer_accuracy", "Answer Accuracy", "LLM", True),
    ("factual_correctness", "Factual Correctness", "LLM", True),
    ("noise_sensitivity", "Noise Sensitivity", "very heavy · multi-step LLM", False),
    ("response_relevancy", "Response Relevancy", "heavy · embeddings + LLM", True),
    ("nonllm_string_similarity", "String Similarity", None, True),
    ("metric_bleu", "BLEU", None, True),
    ("metric_chrf", "CHRF", None, True),
    ("metric_rouge_l", "ROUGE-L", None, True),
    ("corpus_mrr", "MRR", None, True),
    ("corpus_map", "MAP", None, True),
    ("corpus_ndcg@1", "nDCG@1", None, True),
    ("corpus_ndcg@3", "nDCG@3", None, True),
    ("corpus_ndcg@5", "nDCG@5", None, True),
    ("corpus_ndcg@10", "nDCG@10", None, True),
    ("corpus_map@1", "MAP@1", None, True),
    ("corpus_map@3", "MAP@3", None, True),
    ("corpus_map@5", "MAP@5", None, True),
    ("corpus_map@10", "MAP@10", None, True),
]


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG On-the-Fly Evaluator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .health-card {
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .health-label { font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.25rem; }
    .health-value { font-size: 2rem; font-weight: 700; }
    .health-sub   { font-size: 0.8rem; opacity: 0.85; margin-top: 0.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metric groupings
# ---------------------------------------------------------------------------
RETRIEVAL_METRICS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "context_entities_recall": "Context Entities Recall",
    "context_relevance": "Context Relevance",
    "corpus_mrr": "MRR",
    "corpus_map": "MAP",
    "corpus_map@1": "MAP@1",
    "corpus_map@3": "MAP@3",
    "corpus_map@5": "MAP@5",
    "corpus_map@10": "MAP@10",
    "corpus_ndcg@1": "nDCG@1",
    "corpus_ndcg@3": "nDCG@3",
    "corpus_ndcg@5": "nDCG@5",
    "corpus_ndcg@10": "nDCG@10",
}

GENERATION_METRICS = {
    "faithfulness": "Faithfulness",
    "response_groundedness": "Response Groundedness",
    "answer_accuracy": "Answer Accuracy",
    "factual_correctness": "Factual Correctness",
    "metric_bleu": "BLEU",
    "metric_chrf": "CHRF",
    "metric_rouge_l": "ROUGE-L",
    "ragas_bleu": "RAGAS BLEU",
    "ragas_chrf": "RAGAS CHRF",
    "ragas_rouge": "RAGAS ROUGE",
}

END_TO_END_METRICS = {
    "noise_sensitivity": "Noise Sensitivity",
    "response_relevancy": "Response Relevancy",
    "nonllm_string_similarity": "String Similarity",
}

INVERT_FOR_HEALTH = {"noise_sensitivity"}

# z-scores for two-sided normal confidence intervals
CONFIDENCE_Z: Dict[float, float] = {
    0.80: 1.28155,
    0.90: 1.64485,
    0.95: 1.95996,
    0.98: 2.32635,
    0.99: 2.57583,
}

# Most pessimistic stdev for scores bounded in [0, 1]
PESSIMISTIC_STDEV = 0.5


def health_score(df: pd.DataFrame, metrics: List[str]) -> float:
    if not metrics:
        return float("nan")
    vals = []
    for m in metrics:
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if s.empty:
            continue
        mean_val = float(s.mean())
        if m in INVERT_FOR_HEALTH:
            mean_val = 1.0 - mean_val
        vals.append(mean_val)
    return float(np.mean(vals)) if vals else float("nan")


def health_color(score: float) -> str:
    if np.isnan(score):
        return "#95a5a6"
    if score >= 0.75:
        return "#27ae60"
    if score >= 0.50:
        return "#f39c12"
    return "#e74c3c"


def available_metrics(df: pd.DataFrame, mapping: dict) -> List[str]:
    return [m for m in mapping if m in df.columns and df[m].notna().any()]


def format_eta(seconds: float) -> str:
    if seconds < 0 or np.isnan(seconds) or np.isinf(seconds):
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def margin_of_error(n: int, confidence: float = 0.95) -> float:
    """Worst-case MoE for a mean of [0,1] scores: z * 0.5 / sqrt(n)."""
    if n <= 0:
        return float("nan")
    z = CONFIDENCE_Z.get(confidence, 1.95996)
    return float(z * PESSIMISTIC_STDEV / np.sqrt(n))


def _group_latency_cols(df: pd.DataFrame, metric_keys: Sequence[str]) -> List[str]:
    """Return latency column names that exist for the given metric keys."""
    cols = []
    for m in metric_keys:
        lat = f"{m}_latency_ms"
        if lat in df.columns and df[lat].notna().any():
            cols.append(lat)
    return cols


def _row_group_mean_latency(row: pd.Series, latency_cols: Sequence[str]) -> float:
    if not latency_cols:
        return float("nan")
    vals = [float(row[c]) for c in latency_cols if pd.notna(row.get(c))]
    return float(np.mean(vals)) if vals else float("nan")


def build_latency_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-row latency table:
      - agent_latency_ms
      - mean retrieval metric latency
      - mean generation metric latency
      - mean end-to-end metric latency
    """
    ret_lat = _group_latency_cols(df, list(RETRIEVAL_METRICS))
    gen_lat = _group_latency_cols(df, list(GENERATION_METRICS))
    e2e_lat = _group_latency_cols(df, list(END_TO_END_METRICS))

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append(
            {
                "row": i + 1,
                "agent_ms": float(row["agent_latency_ms"])
                if "agent_latency_ms" in row and pd.notna(row["agent_latency_ms"])
                else np.nan,
                "retrieval_ms": _row_group_mean_latency(row, ret_lat),
                "generation_ms": _row_group_mean_latency(row, gen_lat),
                "end_to_end_ms": _row_group_mean_latency(row, e2e_lat),
            }
        )
    return pd.DataFrame(records)


def build_type_means(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Mean of retrieval / generation / end-to-end *score* groups per question type.
    Returns long-form DataFrame suitable for a line/bar chart, or None.
    """
    if "type" not in df.columns or df["type"].isna().all():
        return None

    ret_m = available_metrics(df, RETRIEVAL_METRICS)
    gen_m = available_metrics(df, GENERATION_METRICS)
    e2e_m = available_metrics(df, END_TO_END_METRICS)

    rows = []
    for qtype, grp in df.groupby("type", dropna=True):
        def _mean_group(metrics: List[str]) -> float:
            if not metrics:
                return float("nan")
            vals = []
            for m in metrics:
                s = grp[m].dropna()
                if not s.empty:
                    vals.append(float(s.mean()))
            return float(np.mean(vals)) if vals else float("nan")

        rows.append(
            {
                "type": str(qtype),
                "retrieval": _mean_group(ret_m),
                "generation": _mean_group(gen_m),
                "end_to_end": _mean_group(e2e_m),
                "n": len(grp),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def render_latency_charts(df: pd.DataFrame, key_prefix: str = "live") -> None:
    """Line chart of latencies + mean/median stats underneath."""
    lat_df = build_latency_series(df)
    if lat_df.empty:
        return

    st.subheader("Latency over questions")
    fig = go.Figure()
    for col, name, color in [
        ("agent_ms", "Agent call", "#1f77b4"),
        ("retrieval_ms", "Retrieval metrics (mean)", "#2ca02c"),
        ("generation_ms", "Generation metrics (mean)", "#ff7f0e"),
        ("end_to_end_ms", "End-to-end metrics (mean)", "#d62728"),
    ]:
        if lat_df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=lat_df["row"],
                    y=lat_df[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                )
            )
    fig.update_layout(
        xaxis_title="Question # (processed order)",
        yaxis_title="Latency (ms)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_lat_line")

    # Mean / median table
    stats_rows = []
    for col, label in [
        ("agent_ms", "Agent call"),
        ("retrieval_ms", "Retrieval metrics"),
        ("generation_ms", "Generation metrics"),
        ("end_to_end_ms", "End-to-end metrics"),
    ]:
        s = lat_df[col].dropna()
        if s.empty:
            continue
        stats_rows.append(
            {
                "latency type": label,
                "mean (ms)": float(s.mean()),
                "median (ms)": float(s.median()),
                "n": int(s.count()),
            }
        )
    if stats_rows:
        st.caption("Mean and median latency (current sample)")
        st.dataframe(
            pd.DataFrame(stats_rows).style.format(
                {"mean (ms)": "{:.1f}", "median (ms)": "{:.1f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_type_metric_chart(df: pd.DataFrame, key_prefix: str = "live") -> None:
    """Line/bar chart of mean retrieval / generation / end-to-end scores by type."""
    type_df = build_type_means(df)
    if type_df is None or type_df.empty:
        return

    st.subheader("Mean scores by question type")
    long = type_df.melt(
        id_vars=["type", "n"],
        value_vars=["retrieval", "generation", "end_to_end"],
        var_name="group",
        value_name="mean_score",
    )
    fig = px.line(
        long,
        x="type",
        y="mean_score",
        color="group",
        markers=True,
        title="Mean metric scores by failure-mode type",
        labels={"mean_score": "Mean score", "type": "Question type", "group": "Group"},
        range_y=[0, 1.05],
    )
    fig.update_layout(xaxis_tickangle=-25, height=380)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_type_means")

    st.caption("Sample sizes per type")
    st.dataframe(
        type_df[["type", "n", "retrieval", "generation", "end_to_end"]].style.format(
            {"retrieval": "{:.3f}", "generation": "{:.3f}", "end_to_end": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_moe_panel(n: int, confidence: float, key_prefix: str = "live") -> None:
    """Show current worst-case margin of error for the selected confidence level."""
    moe = margin_of_error(n, confidence)
    st.subheader("Margin of error (worst-case σ = 0.5)")
    st.caption(
        "Assumes scores in [0, 1] and the most pessimistic standard deviation (0.5). "
        f"MoE = z · 0.5 / √n  at the selected confidence level."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Processed rows (n)", n)
    c2.metric("Confidence level", f"{confidence * 100:.0f}%")
    c3.metric("Margin of error (±)", f"{moe:.4f}" if not np.isnan(moe) else "—")


# ---------------------------------------------------------------------------
# Live evaluation runner
# ---------------------------------------------------------------------------
def run_live_evaluation(
    questions_df: pd.DataFrame,
    agent_cfg: AgentConfig,
    eval_cfg: EvaluationConfig,
    call_agent: bool,
    question_col: str,
    answer_col: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    n = len(questions_df)
    results: List[Dict[str, Any]] = []

    progress_bar = st.progress(0.0, text="Starting evaluation…")
    status = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    latency_placeholder = st.empty()
    type_placeholder = st.empty()
    moe_placeholder = st.empty()
    log_placeholder = st.empty()

    agent = AgentRunner(agent_cfg)
    evaluator = OnTheFlyEvaluator(agent=agent, eval_config=eval_cfg)

    t0 = time.perf_counter()

    for i, (_, row) in enumerate(questions_df.iterrows()):
        q = str(row.get(question_col) or "")
        ref = str(row.get(answer_col) or "") if answer_col in row.index else ""
        extra = {
            k: row[k]
            for k in ("type", "source", "chapter")
            if k in row.index and pd.notna(row[k])
        }

        if call_agent:
            ev: RowEvaluation = evaluator.evaluate_row_sync(
                question=q,
                reference=ref,
                row_index=i,
                extra=extra,
            )
        else:
            from core.data_prep import extract_contexts, parse_generation_chunks

            gen_ans = str(row.get("generation_answer") or "")
            chunks = parse_generation_chunks(row.get("generation_chunks"))
            contexts = extract_contexts(chunks)
            ev = evaluator.evaluate_row_sync(
                question=q,
                reference=ref,
                precomputed_answer=gen_ans,
                precomputed_contexts=contexts,
                row_index=i,
                extra=extra,
            )

        results.append(ev.to_flat_dict())
        done = i + 1
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (n - done) / rate if rate > 0 else 0
        pct = done / n

        progress_bar.progress(
            pct,
            text=(
                f"{done}/{n}  ({pct * 100:.1f}%)  |  "
                f"elapsed {format_eta(elapsed)}  |  "
                f"ETA {format_eta(remaining)}"
            ),
        )
        status.info(
            f"**Last question:** {q[:80]}{'…' if len(q) > 80 else ''}\n\n"
            f"Agent success: {'✅' if ev.agent_success else '❌'}  ·  "
            f"Latency: {ev.agent_latency_ms:.0f} ms  ·  "
            f"Contexts: {len(ev.contexts)}"
        )

        live_df = pd.DataFrame(results)
        st.session_state["live_results"] = live_df

        metric_cols = [
            c
            for c in live_df.columns
            if c
            in list(RETRIEVAL_METRICS)
            + list(GENERATION_METRICS)
            + list(END_TO_END_METRICS)
            and live_df[c].notna().any()
        ]
        if metric_cols:
            means = live_df[metric_cols].mean(numeric_only=True).sort_values(
                ascending=False
            )
            with metrics_placeholder.container():
                st.subheader(f"Live metric means ({done} rows)")
                display = pd.DataFrame(
                    {
                        "metric": means.index,
                        "mean": means.values,
                        "confidence": [
                            live_df[f"{m}_confidence"].mean()
                            if f"{m}_confidence" in live_df
                            else np.nan
                            for m in means.index
                        ],
                        "margin_of_error": [
                            live_df[f"{m}_margin_of_error"].mean()
                            if f"{m}_margin_of_error" in live_df
                            else np.nan
                            for m in means.index
                        ],
                    }
                )
                st.dataframe(
                    display.style.format(
                        {
                            "mean": "{:.3f}",
                            "confidence": "{:.2f}",
                            "margin_of_error": "±{:.3f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with chart_placeholder.container():
                fig = px.bar(
                    display,
                    x="metric",
                    y="mean",
                    error_y="margin_of_error",
                    title="Current mean scores (± margin of error)",
                    color="mean",
                    color_continuous_scale="RdYlGn",
                    range_y=[0, 1.05],
                )
                fig.update_layout(
                    xaxis_tickangle=-35, height=380, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"live_bar_{done}")

        # Latency line chart + mean/median
        with latency_placeholder.container():
            render_latency_charts(live_df, key_prefix=f"live_{done}")

        # Scores by question type
        with type_placeholder.container():
            render_type_metric_chart(live_df, key_prefix=f"live_{done}")

        # Worst-case MoE for selected confidence
        with moe_placeholder.container():
            render_moe_panel(done, confidence, key_prefix=f"live_{done}")

        with log_placeholder.container():
            st.caption("Recent answers")
            tail = live_df[["question", "generation_answer", "agent_success"]].tail(5)
            st.dataframe(tail, use_container_width=True, hide_index=True)

    progress_bar.progress(1.0, text=f"Done – {n}/{n} questions evaluated")
    status.success(f"Evaluation finished in {format_eta(time.perf_counter() - t0)}")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Post-run visualisation
# ---------------------------------------------------------------------------
def render_results_dashboard(df: pd.DataFrame, confidence: float = 0.95) -> None:
    ret_m = available_metrics(df, RETRIEVAL_METRICS)
    gen_m = available_metrics(df, GENERATION_METRICS)
    e2e_m = available_metrics(df, END_TO_END_METRICS)

    h_ret = health_score(df, ret_m)
    h_gen = health_score(df, gen_m)
    h_e2e = health_score(df, e2e_m)

    st.subheader("Category Health Scores")
    st.caption(
        "Higher is better (0–1). Noise sensitivity is inverted. "
        "Error bars reflect average margin of error when available."
    )

    c1, c2, c3 = st.columns(3)
    for col, label, score, metrics in [
        (c1, "Retrieval", h_ret, ret_m),
        (c2, "Generation", h_gen, gen_m),
        (c3, "End-to-end", h_e2e, e2e_m),
    ]:
        color = health_color(score)
        score_display = f"{score:.2f}" if not np.isnan(score) else "N/A"
        with col:
            st.markdown(
                f"""
                <div class="health-card" style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%);">
                    <div class="health-label">{label} Health</div>
                    <div class="health-value">{score_display}</div>
                    <div class="health-sub">{len(metrics)} metrics averaged</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Metric summary (mean ± margin of error, mean confidence)")
    all_m = ret_m + gen_m + e2e_m
    rows = []
    for m in all_m:
        mean = float(df[m].mean()) if m in df else np.nan
        conf = (
            float(df[f"{m}_confidence"].mean())
            if f"{m}_confidence" in df
            else np.nan
        )
        moe = (
            float(df[f"{m}_margin_of_error"].mean())
            if f"{m}_margin_of_error" in df
            else np.nan
        )
        rows.append({"metric": m, "mean": mean, "± margin": moe, "confidence": conf})
    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(
            summary_df.style.format(
                {"mean": "{:.3f}", "± margin": "±{:.3f}", "confidence": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(
            summary_df,
            x="metric",
            y="mean",
            error_y="± margin",
            title="Mean scores with margin of error",
            color="mean",
            color_continuous_scale="RdYlGn",
            range_y=[0, 1.05],
        )
        fig.update_layout(xaxis_tickangle=-40, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Latency charts (same as live view)
    render_latency_charts(df, key_prefix="results")

    # Scores by question type
    render_type_metric_chart(df, key_prefix="results")

    # Worst-case MoE panel
    render_moe_panel(len(df), confidence, key_prefix="results")

    if "agent_latency_ms" in df.columns:
        st.subheader("Agent latency distribution")
        st.metric("Mean agent latency (ms)", f"{df['agent_latency_ms'].mean():.0f}")
        fig_lat = px.histogram(
            df, x="agent_latency_ms", nbins=20, title="Agent latency distribution (ms)"
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    st.subheader("Full results")
    st.dataframe(df, use_container_width=True, height=400)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="rag_evaluation_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(questions_path: Optional[str] = None) -> None:
    st.markdown(
        '<p class="main-header">📊 RAG On-the-Fly Evaluation Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Call the agent live · evaluate enabled metrics · '
        "track progress, confidence & margin of error</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Configuration")

    default_q = questions_path or str(PROJECT_ROOT / "sample_questions.csv")
    q_path = st.sidebar.text_input("Questions CSV path", value=default_q)
    agent_yaml = st.sidebar.text_input(
        "Agent config", value=str(PROJECT_ROOT / "configs" / "agent.yaml")
    )
    eval_yaml = st.sidebar.text_input(
        "Evaluation config", value=str(PROJECT_ROOT / "configs" / "evaluation.yaml")
    )

    call_agent = st.sidebar.checkbox("Call agent on-the-fly", value=True)
    limit = st.sidebar.number_input("Limit rows (0 = all)", min_value=0, value=0, step=1)
    continue_eval = st.sidebar.checkbox(
        "Continue from existing output",
        value=False,
        help=(
            "If an output file with the same name as the input already exists "
            "and has fewer rows, resume from the next missing row. "
            "When set, Limit applies to the number of *new* rows to process."
        ),
    )
    output_dir = st.sidebar.text_input(
        "Output directory (optional)",
        value="",
        help=(
            "Directory where the result CSV is written using the same filename "
            "as the input. Leave empty to only offer a download button."
        ),
    )
    question_col = st.sidebar.text_input("Question column", value="question")
    answer_col = st.sidebar.text_input("Gold answer column", value="answer")

    # Confidence level for worst-case MoE
    conf_options = list(CONFIDENCE_Z.keys())
    conf_labels = [f"{int(c * 100)}%" for c in conf_options]
    conf_idx = conf_labels.index("95%") if "95%" in conf_labels else 2
    selected_label = st.sidebar.selectbox(
        "Confidence level (MoE)",
        options=conf_labels,
        index=conf_idx,
        help=(
            "Used for the worst-case margin of error (σ = 0.5) shown while "
            "rows accumulate and in the results dashboard."
        ),
    )
    confidence = conf_options[conf_labels.index(selected_label)]

    try:
        eval_cfg = EvaluationConfig.from_yaml(eval_yaml, base_dir=PROJECT_ROOT)
    except Exception as e:
        st.sidebar.error(f"Could not load evaluation config: {e}")
        eval_cfg = None

    # ---- Metric checkboxes (initialised from evaluation.yaml flags) ----
    st.sidebar.markdown("**Metrics**")
    st.sidebar.caption(
        "Unchecked metrics are skipped. Labels marked *heavy* use more LLM / embedding calls. "
        "Defaults come from the metrics flags in evaluation.yaml."
    )
    selected_metrics: Dict[str, bool] = {}
    for key, label, hint, default_on in METRIC_CATALOG:
        display = label if not hint else f"{label}  ·  _{hint}_"
        # Prefer the value from evaluation.yaml when available
        if eval_cfg is not None and key in eval_cfg.metrics:
            initial = bool(eval_cfg.metrics[key])
        else:
            initial = default_on
        selected_metrics[key] = st.sidebar.checkbox(
            display, value=initial, key=f"metric_{key}"
        )

    # Apply sidebar metric selection onto the eval config
    if eval_cfg is not None:
        for k, on in selected_metrics.items():
            eval_cfg.metrics[k] = on

    run_btn = st.sidebar.button("▶ Run evaluation", type="primary")

    if "live_results" not in st.session_state:
        st.session_state["live_results"] = None

    if run_btn:
        if not Path(q_path).exists():
            st.error(f"Questions file not found: {q_path}")
            st.stop()
        if eval_cfg is None:
            st.error("Evaluation config could not be loaded.")
            st.stop()
        try:
            agent_cfg = AgentConfig.from_yaml(agent_yaml)
        except Exception as e:
            st.error(f"Could not load agent config: {e}")
            st.stop()

        df_full = pd.read_csv(q_path)
        if question_col not in df_full.columns:
            st.error(f"Missing required column '{question_col}'")
            st.stop()

        # Resolve output path (same convention as CLI)
        out_path: Optional[Path] = None
        if output_dir.strip():
            out_dir = Path(output_dir.strip())
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(q_path).name
        else:
            # Still support continue against a sibling evaluated file
            candidate = Path(q_path).with_name(Path(q_path).stem + "_evaluated.csv")
            if continue_eval and candidate.exists():
                out_path = candidate

        start_idx = 0
        existing_df: Optional[pd.DataFrame] = None
        if continue_eval and out_path is not None and out_path.exists():
            try:
                existing_df = pd.read_csv(out_path)
                start_idx = len(existing_df)
                if start_idx >= len(df_full):
                    st.success(
                        f"Output already complete ({start_idx} rows). Nothing to do."
                    )
                    st.session_state["live_results"] = existing_df
                    st.stop()
                st.info(
                    f"Continuing from row {start_idx} "
                    f"({len(df_full) - start_idx} remaining) …"
                )
            except Exception as e:
                st.warning(f"Could not read existing output for continue: {e}")
                existing_df = None
                start_idx = 0

        # Slice to process: from start_idx, optionally limited
        df_q = df_full.iloc[start_idx:].copy()
        if limit and limit > 0:
            df_q = df_q.head(int(limit))

        if len(df_q) == 0:
            st.success("No new rows to process.")
            if existing_df is not None:
                st.session_state["live_results"] = existing_df
            st.stop()

        st.info(f"Evaluating {len(df_q)} questions (starting at input row {start_idx}) …")
        new_df = run_live_evaluation(
            questions_df=df_q,
            agent_cfg=agent_cfg,
            eval_cfg=eval_cfg,
            call_agent=call_agent,
            question_col=question_col,
            answer_col=answer_col,
            confidence=confidence,
        )

        if existing_df is not None and len(existing_df) > 0:
            result_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            result_df = new_df

        if out_path is not None:
            try:
                result_df.to_csv(out_path, index=False, quoting=1)
                st.success(f"Wrote results → {out_path}")
            except Exception as e:
                st.warning(f"Could not write output to {out_path}: {e}")

        st.session_state["live_results"] = result_df

    if st.session_state.get("live_results") is not None:
        st.markdown("---")
        st.header("Results")
        render_results_dashboard(
            st.session_state["live_results"], confidence=confidence
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", "-q", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(questions_path=args.questions)
