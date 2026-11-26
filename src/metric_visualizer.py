"""
Streamlit Dashboard for RAG Evaluation Metrics
Works with the unified FAISS evaluation CSV (single file containing
generation metrics, retrieval metrics, LLM-as-a-Judge scores, and
string-similarity scores).

Health scores are computed for three failure-mode categories:
  1. Retrieval
  2. Generation
  3. End-to-end

Usage:
  streamlit run metric_visualizer.py -- --csv path/to/evaluation.csv
  # or
  python metric_visualizer.py --csv path/to/evaluation.csv
"""

import argparse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Metric Visualizer",
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
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metric groupings (exactly as requested)
# ---------------------------------------------------------------------------
RETRIEVAL_METRICS = {
    "context_precision": "Context Precision (irrelevant chunks)",
    "context_recall": "Context Recall (missing relevant chunks)",
    "context_entities_recall": "Context Entities Recall (missing key entities)",
    "context_relevance": "Context Relevance (overall)",
    "corpus_mrr": "MRR (ranking of first relevant doc)",
    "corpus_map": "MAP (mean average precision)",
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
    "faithfulness": "Faithfulness (hallucination / unsupported claims)",
    "response_groundedness": "Response Groundedness",
    "answer_accuracy": "Answer Accuracy",
    "factual_correctness": "Factual Correctness",
    "metric_bleu": "BLEU",
    "metric_chrf": "CHRF",
    "metric_rouge_l": "ROUGE-L",
    # also expose ragas variants if present
    "ragas_bleu": "RAGAS BLEU",
    "ragas_chrf": "RAGAS CHRF",
    "ragas_rouge": "RAGAS ROUGE",
}

END_TO_END_METRICS = {
    "noise_sensitivity": "Noise Sensitivity (impact of irrelevant context)",
    "response_relevancy": "Response Relevancy",
    "nonllm_string_similarity": "Semantic / String Similarity",
}

# Metrics where *lower* is better → invert for health score
INVERT_FOR_HEALTH = {"noise_sensitivity"}

CATEGORY_COLORS = {
    "Retrieval": "#2ecc71",
    "Generation": "#3498db",
    "End-to-end": "#9b59b6",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Coerce every metric-like column to numeric
    metric_like = [
        c
        for c in df.columns
        if any(
            c.startswith(p)
            for p in (
                "metric_",
                "corpus_",
                "context_",
                "noise_",
                "response_",
                "faithfulness",
                "answer_",
                "factual_",
                "nonllm_",
                "ragas_",
                "generation_",
                "input_token",
                "output_token",
                "elapsed_time",
            )
        )
        or c
        in (
            "input_token_count",
            "output_token_count",
            "elapsed_time",
            "generation_llm_calls",
            "generation_input_tokens",
            "generation_output_tokens",
            "generation_latency_s",
        )
    ]
    for c in metric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep type as categorical with observed order
    if "type" in df.columns:
        order = df["type"].value_counts().index.tolist()
        df["type"] = pd.Categorical(df["type"], categories=order, ordered=True)

    return df


def available_metrics(df: pd.DataFrame, mapping: dict) -> list[str]:
    """Return only columns that exist and have at least one non-null value."""
    return [m for m in mapping if m in df.columns and df[m].notna().any()]


def health_score(df: pd.DataFrame, metrics: list[str]) -> float:
    """
    Mean of available metrics (after optional inversion).
    Returns NaN if nothing is usable.
    """
    if not metrics:
        return float("nan")
    vals = []
    for m in metrics:
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


def mean_median_table(df: pd.DataFrame, cols: list[str], group_col: str = "type") -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg = {c: ["mean", "median"] for c in present}
    summary = df.groupby(group_col, observed=True)[present].agg(agg).round(3)
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def create_bar_by_type(df: pd.DataFrame, metric: str, title: str):
    means = df.groupby("type", observed=True)[metric].mean().reset_index()
    fig = px.bar(
        means,
        x="type",
        y=metric,
        title=title,
        color="type",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f",
    )
    fig.update_layout(
        xaxis_title="Question Type",
        yaxis_title=metric.replace("_", " ").title(),
        showlegend=False,
        height=380,
        xaxis_tickangle=-30,
    )
    fig.update_traces(textposition="outside")
    return fig


def create_box_by_type(df: pd.DataFrame, metrics: list[str], title: str):
    present = [m for m in metrics if m in df.columns]
    if not present:
        return go.Figure()
    melted = df.melt(id_vars=["type"], value_vars=present, var_name="metric", value_name="value")
    fig = px.box(
        melted,
        x="type",
        y="value",
        color="metric",
        title=title,
        points="outliers",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        xaxis_title="Question Type",
        yaxis_title="Score",
        height=420,
        legend_title="Metric",
        xaxis_tickangle=-30,
    )
    return fig


def create_heatmap(df: pd.DataFrame, metrics: list[str], title: str):
    present = [m for m in metrics if m in df.columns]
    if not present:
        return go.Figure()
    means = df.groupby("type", observed=True)[present].mean().T
    fig = px.imshow(
        means,
        labels=dict(x="Question Type", y="Metric", color="Mean Score"),
        x=means.columns.astype(str),
        y=means.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=title,
        text_auto=".2f",
        zmin=0,
        zmax=1,
    )
    fig.update_layout(height=max(320, 28 * len(present)))
    return fig


def create_radar(df: pd.DataFrame, metrics: list[str], title: str):
    present = [m for m in metrics if m in df.columns][:8]  # keep readable
    if not present:
        return go.Figure()
    means = df.groupby("type", observed=True)[present].mean()
    fig = go.Figure()
    for t in means.index:
        r = means.loc[t].values.tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=present + [present[0]],
                fill="toself",
                name=str(t),
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        height=520,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main(csv_path: str | None = None):
    st.markdown(
        '<p class="main-header">📊 RAG Evaluation Metrics Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Health scores & detailed analysis for Retrieval, Generation, and End-to-end failure modes</p>',
        unsafe_allow_html=True,
    )

    # ---- Sidebar: file + filters ----
    st.sidebar.header("Data & Filters")

    # Prefer CLI / function argument; fall back to common locations
    default_candidates = [
        Path(csv_path) if csv_path else None,
        Path("/home/workdir/attachments/faiss_evaluated_202611260954_202611261139.csv"),
        Path("faiss_evaluated.csv"),
        Path("../data/evaluations/faiss_evaluated.csv"),
        Path("data/evaluations/faiss_evaluated.csv"),
    ]
    existing = [p for p in default_candidates if p is not None and p.exists()]
    default_path = str(existing[0]) if existing else (csv_path or "")

    csv_path_ui = st.sidebar.text_input("Path to evaluation CSV", value=default_path)
    if not csv_path_ui or not Path(csv_path_ui).exists():
        st.error(
            "Please provide a valid path to the evaluation CSV.\n\n"
            "Pass it as a CLI argument:\n"
            "  streamlit run metric_visualizer.py -- --csv path/to/file.csv"
        )
        st.stop()

    df = load_data(csv_path_ui)

    all_types = list(df["type"].cat.categories) if hasattr(df["type"], "cat") else sorted(df["type"].unique())
    selected_types = st.sidebar.multiselect(
        "Question types",
        options=all_types,
        default=all_types,
    )
    if not selected_types:
        st.warning("Select at least one question type.")
        st.stop()

    df_f = df[df["type"].isin(selected_types)].copy()

    show_raw = st.sidebar.checkbox("Show full raw tables in explorer", value=False)

    # ---- Resolve which metrics actually exist ----
    ret_m = available_metrics(df_f, RETRIEVAL_METRICS)
    gen_m = available_metrics(df_f, GENERATION_METRICS)
    e2e_m = available_metrics(df_f, END_TO_END_METRICS)

    # Prefer the non-ragas string metrics for generation when both exist
    # (already handled by order in the dict; both can still appear)

    # ---- Health scores (top of page) ----
    h_ret = health_score(df_f, ret_m)
    h_gen = health_score(df_f, gen_m)
    h_e2e = health_score(df_f, e2e_m)

    st.subheader("Category Health Scores")
    st.caption(
        "Higher is better (0–1). Noise sensitivity is inverted (1 − score) so that higher still means healthier."
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

    st.markdown("")  # spacer

    # Quick aggregates
    st.subheader("Run Overview (selected types)")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Questions", len(df_f))
    with k2:
        st.metric(
            "Avg Input Tokens",
            f"{df_f['input_token_count'].mean():.0f}" if "input_token_count" in df_f else "—",
        )
    with k3:
        st.metric(
            "Avg Output Tokens",
            f"{df_f['output_token_count'].mean():.0f}" if "output_token_count" in df_f else "—",
        )
    with k4:
        st.metric(
            "Avg Latency (s)",
            f"{df_f['generation_latency_s'].mean():.2f}" if "generation_latency_s" in df_f else "—",
        )
    with k5:
        st.metric(
            "Avg LLM Calls",
            f"{df_f['generation_llm_calls'].mean():.1f}" if "generation_llm_calls" in df_f else "—",
        )

    # ---- Tabs ----
    tab_overview, tab_ret, tab_gen, tab_e2e, tab_gen_cost, tab_explorer = st.tabs(
        [
            "📈 Overview by Type",
            "🔍 Retrieval Failure Modes",
            "🧠 Generation Failure Modes",
            "🔗 End-to-End",
            "⚙️ Generation Cost & Latency",
            "📋 Data Explorer",
        ]
    )

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    with tab_overview:
        st.header("Average Scores by Question Type")

        # Compact health per type
        st.subheader("Health Score per Question Type")
        rows = []
        for t in selected_types:
            sub = df_f[df_f["type"] == t]
            rows.append(
                {
                    "type": t,
                    "Retrieval": health_score(sub, ret_m),
                    "Generation": health_score(sub, gen_m),
                    "End-to-end": health_score(sub, e2e_m),
                    "n": len(sub),
                }
            )
        health_by_type = pd.DataFrame(rows).set_index("type").round(3)
        st.dataframe(health_by_type, use_container_width=True)

        # Heatmap of the three health scores
        fig_h = px.imshow(
            health_by_type[["Retrieval", "Generation", "End-to-end"]].T,
            labels=dict(x="Question Type", y="Category", color="Health"),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Health Scores by Type × Category",
            text_auto=".2f",
            zmin=0,
            zmax=1,
        )
        fig_h.update_layout(height=280)
        st.plotly_chart(fig_h, use_container_width=True, key="overview_health_heatmap")

        # Key metric bars
        st.subheader("Selected Key Metrics")
        key_metrics = [
            m
            for m in [
                "context_precision",
                "context_recall",
                "faithfulness",
                "answer_accuracy",
                "response_relevancy",
                "metric_bleu",
            ]
            if m in df_f.columns
        ]
        cols = st.columns(3)
        for i, m in enumerate(key_metrics):
            with cols[i % 3]:
                st.plotly_chart(
                    create_bar_by_type(df_f, m, m.replace("_", " ").title()),
                    use_container_width=True,
                    key=f"overview_bar_{m}",
                )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    with tab_ret:
        st.header("Retrieval Failure Modes")
        st.markdown(
            """
            | Metric | What a low score indicates |
            |--------|---------------------------|
            | **Context precision** | Too many irrelevant chunks retrieved |
            | **Context recall** | Missing relevant chunks |
            | **Context entities recall** | Missing key entities from the ground-truth answer |
            | **Context relevance** | Overall low relevance of retrieved context |
            | **MRR / MAP / nDCG@k** | Poor ranking quality of relevant documents |
            """
        )

        if not ret_m:
            st.warning("No retrieval metrics found in the CSV.")
        else:
            st.subheader("Mean / Median by Type")
            st.dataframe(mean_median_table(df_f, ret_m), use_container_width=True, hide_index=True)

            st.plotly_chart(
                create_heatmap(df_f, ret_m, "Mean Retrieval Scores by Type × Metric"),
                use_container_width=True,
                key="ret_heatmap",
            )

            # Core vs ranking split for readability
            core_ret = [m for m in ret_m if m.startswith("context_")]
            rank_ret = [m for m in ret_m if m.startswith("corpus_")]

            if core_ret:
                st.subheader("Context Quality Metrics")
                st.plotly_chart(
                    create_box_by_type(df_f, core_ret, "Context Metrics Distribution"),
                    use_container_width=True,
                    key="ret_context_box",
                )
                st.plotly_chart(
                    create_radar(df_f, core_ret, "Radar – Context Metrics by Type"),
                    use_container_width=True,
                    key="ret_context_radar",
                )

            if rank_ret:
                st.subheader("Ranking Metrics (corpus-level – constant across rows)")
                # Show once
                rank_vals = df_f[rank_ret].iloc[0]
                rank_df = pd.DataFrame({"Metric": rank_vals.index, "Value": rank_vals.values.round(4)})
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
                fig_rank = px.bar(
                    rank_df,
                    x="Metric",
                    y="Value",
                    title="Corpus Ranking Metrics",
                    color="Value",
                    color_continuous_scale="Viridis",
                    text_auto=".3f",
                )
                fig_rank.update_layout(xaxis_tickangle=-35, height=360, showlegend=False)
                st.plotly_chart(fig_rank, use_container_width=True, key="ret_rank_bar")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    with tab_gen:
        st.header("Generation Failure Modes")
        st.markdown(
            """
            | Metric | What a low score indicates |
            |--------|---------------------------|
            | **Faithfulness** | Answer contains claims not supported by the context (hallucination) |
            | **Response groundedness** | Answer is poorly grounded in the provided context |
            | **Answer accuracy / Factual correctness** | Factual errors vs. ground truth |
            | **BLEU / CHRF / ROUGE** | Low n-gram / character / LCS overlap with reference |
            """
        )

        if not gen_m:
            st.warning("No generation metrics found in the CSV.")
        else:
            st.subheader("Mean / Median by Type")
            st.dataframe(mean_median_table(df_f, gen_m), use_container_width=True, hide_index=True)

            st.plotly_chart(
                create_heatmap(df_f, gen_m, "Mean Generation Scores by Type × Metric"),
                use_container_width=True,
                key="gen_heatmap",
            )

            llm_judge = [
                m
                for m in gen_m
                if m
                in (
                    "faithfulness",
                    "response_groundedness",
                    "answer_accuracy",
                    "factual_correctness",
                )
            ]
            string_m = [m for m in gen_m if m not in llm_judge]

            if llm_judge:
                st.subheader("LLM-as-a-Judge Metrics")
                st.plotly_chart(
                    create_box_by_type(df_f, llm_judge, "LLM-Judge Distributions"),
                    use_container_width=True,
                    key="gen_llm_box",
                )
                st.plotly_chart(
                    create_radar(df_f, llm_judge, "Radar – LLM-Judge Metrics"),
                    use_container_width=True,
                    key="gen_llm_radar",
                )

            if string_m:
                st.subheader("String / N-gram Overlap Metrics")
                st.plotly_chart(
                    create_box_by_type(df_f, string_m, "String Similarity Distributions"),
                    use_container_width=True,
                    key="gen_string_box",
                )

            # Individual bars
            st.subheader("Per-Metric Breakdown")
            cols = st.columns(3)
            for i, m in enumerate(gen_m):
                with cols[i % 3]:
                    st.plotly_chart(
                        create_bar_by_type(df_f, m, GENERATION_METRICS.get(m, m)),
                        use_container_width=True,
                        key=f"gen_bar_{m}",
                    )

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    with tab_e2e:
        st.header("End-to-End Failure Modes")
        st.markdown(
            """
            | Metric | What a low / high score indicates |
            |--------|-----------------------------------|
            | **Noise sensitivity** | High → system is strongly affected by noisy/irrelevant context |
            | **Response relevancy** | Low → answer is not relevant to the question |
            | **Semantic / String similarity** | Low → answer diverges from reference meaning |
            """
        )

        if not e2e_m:
            st.warning("No end-to-end metrics found in the CSV.")
        else:
            st.subheader("Mean / Median by Type")
            st.dataframe(mean_median_table(df_f, e2e_m), use_container_width=True, hide_index=True)

            st.plotly_chart(
                create_heatmap(df_f, e2e_m, "Mean End-to-End Scores by Type × Metric"),
                use_container_width=True,
                key="e2e_heatmap",
            )
            st.plotly_chart(
                create_box_by_type(df_f, e2e_m, "End-to-End Metric Distributions"),
                use_container_width=True,
                key="e2e_box",
            )
            st.plotly_chart(
                create_radar(df_f, e2e_m, "Radar – End-to-End Metrics"),
                use_container_width=True,
                key="e2e_radar",
            )

            cols = st.columns(3)
            for i, m in enumerate(e2e_m):
                with cols[i % 3]:
                    st.plotly_chart(
                        create_bar_by_type(df_f, m, END_TO_END_METRICS.get(m, m)),
                        use_container_width=True,
                        key=f"e2e_bar_{m}",
                    )

    # ------------------------------------------------------------------
    # Generation cost / latency
    # ------------------------------------------------------------------
    with tab_gen_cost:
        st.header("Generation Cost & Latency")

        cost_cols = [
            c
            for c in [
                "input_token_count",
                "output_token_count",
                "generation_input_tokens",
                "generation_output_tokens",
                "generation_latency_s",
                "generation_llm_calls",
                "elapsed_time",
            ]
            if c in df_f.columns
        ]

        if cost_cols:
            st.subheader("Mean / Median by Type")
            st.dataframe(mean_median_table(df_f, cost_cols), use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                if "generation_latency_s" in df_f:
                    st.plotly_chart(
                        create_bar_by_type(df_f, "generation_latency_s", "Mean Generation Latency (s)"),
                        use_container_width=True,
                        key="cost_latency_bar",
                    )
            with c2:
                if "generation_input_tokens" in df_f:
                    st.plotly_chart(
                        create_bar_by_type(
                            df_f, "generation_input_tokens", "Mean Generation Input Tokens"
                        ),
                        use_container_width=True,
                        key="cost_tokens_bar",
                    )

            if {"generation_input_tokens", "generation_latency_s"}.issubset(df_f.columns):
                fig_sc = px.scatter(
                    df_f,
                    x="generation_input_tokens",
                    y="generation_latency_s",
                    color="type",
                    size="generation_llm_calls" if "generation_llm_calls" in df_f else None,
                    hover_data=["question"] if "question" in df_f else None,
                    title="Latency vs Generation Input Tokens (size = # LLM calls)",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_sc.update_layout(height=450)
                st.plotly_chart(fig_sc, use_container_width=True, key="cost_scatter")
        else:
            st.info("No cost / latency columns found.")

    # ------------------------------------------------------------------
    # Data explorer
    # ------------------------------------------------------------------
    with tab_explorer:
        st.header("Data Explorer")

        display_cols = [
            c
            for c in [
                "type",
                "source",
                "chapter",
                "question",
                "generation_answer",
                "answer",
                "input_token_count",
                "output_token_count",
                "generation_latency_s",
                "generation_llm_calls",
                "context_precision",
                "context_recall",
                "context_entities_recall",
                "context_relevance",
                "faithfulness",
                "response_groundedness",
                "answer_accuracy",
                "factual_correctness",
                "metric_bleu",
                "metric_chrf",
                "metric_rouge_l",
                "noise_sensitivity",
                "response_relevancy",
                "nonllm_string_similarity",
            ]
            if c in df_f.columns
        ]

        st.dataframe(
            df_f[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
        )

        if show_raw:
            st.subheader("Full table")
            st.dataframe(df_f, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Health scores = unweighted mean of available metrics in each category "
        "(noise_sensitivity is inverted). Data source: unified FAISS evaluation CSV."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Metrics Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  streamlit run metric_visualizer.py -- --csv path/to/eval.csv\n"
            "  python metric_visualizer.py --csv path/to/eval.csv\n"
        ),
    )
    parser.add_argument(
        "--csv",
        "-c",
        dest="csv_path",
        type=str,
        default=None,
        help="Path to the evaluation CSV file",
    )
    # When launched via `streamlit run ... -- --csv ...`, Streamlit leaves the
    # extra args in sys.argv; argparse will pick them up.
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(csv_path=args.csv_path)
