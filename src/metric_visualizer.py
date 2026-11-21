"""
Streamlit Dashboard for Laudon MIS Evaluation Metrics
Visualizes both LLM-as-a-Judge and Non-LLM metrics from the two evaluation CSVs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast

# Page config
st.set_page_config(
    page_title="Metric Visualizer Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and lightly clean both evaluation CSVs."""
    df_llm = pd.read_csv("../data/evaluations/laudon_evaluated.csv")
    df_nollm = pd.read_csv("../data/evaluations/laudon_evaluated_nollm.csv")

    # Ensure numeric columns
    numeric_cols_llm = [
        "input_token_count", "output_token_count", "elapsed_time",
        "generation_llm_calls", "generation_input_tokens",
        "generation_output_tokens", "generation_latency_s",
        "context_precision", "context_recall", "context_entities_recall",
        "noise_sensitivity", "response_relevancy", "faithfulness",
        "answer_accuracy", "context_relevance", "response_groundedness",
        "factual_correctness", "nonllm_string_similarity", "bleu", "chrf", "rouge"
    ]
    for col in numeric_cols_llm:
        if col in df_llm.columns:
            df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")

    numeric_cols_nollm = [
        "input_token_count", "output_token_count", "elapsed_time",
        "generation_llm_calls", "generation_input_tokens",
        "generation_output_tokens", "generation_latency_s",
        "metric_bleu", "metric_chrf", "metric_rouge_l",
        "corpus_mrr", "corpus_map", "corpus_ndcg@1", "corpus_ndcg@3",
        "corpus_ndcg@5", "corpus_ndcg@10", "corpus_map@1", "corpus_map@3",
        "corpus_map@5", "corpus_map@10", "corpus_bleu_score",
        "corpus_chrf_score", "corpus_rouge_score"
    ]
    for col in numeric_cols_nollm:
        if col in df_nollm.columns:
            df_nollm[col] = pd.to_numeric(df_nollm[col], errors="coerce")

    # Normalize type names for display
    type_order = ["basic", "complexity", "information_integration", "negative_unsupported_questions"]
    df_llm["type"] = pd.Categorical(df_llm["type"], categories=type_order, ordered=True)
    df_nollm["type"] = pd.Categorical(df_nollm["type"], categories=type_order, ordered=True)

    return df_llm, df_nollm


def mean_median_table(df, cols, group_col="type"):
    """Return a nice mean / median table grouped by type."""
    agg = {}
    for c in cols:
        agg[c] = ["mean", "median"]
    summary = df.groupby(group_col, observed=True)[cols].agg(agg).round(3)
    # Flatten multi-index columns
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def create_bar_by_type(df, metric, title, color="#636EFA"):
    """Bar chart of mean metric by type."""
    means = df.groupby("type", observed=True)[metric].mean().reset_index()
    fig = px.bar(
        means, x="type", y=metric,
        title=title,
        color="type",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".3f"
    )
    fig.update_layout(
        xaxis_title="Question Type",
        yaxis_title=metric.replace("_", " ").title(),
        showlegend=False,
        height=400,
    )
    fig.update_traces(textposition="outside")
    return fig


def create_box_by_type(df, metrics, title):
    """Box plots of several metrics grouped by type."""
    melted = df.melt(
        id_vars=["type"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )
    fig = px.box(
        melted, x="type", y="value", color="metric",
        title=title,
        points="outliers",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        xaxis_title="Question Type",
        yaxis_title="Score",
        height=450,
        legend_title="Metric",
    )
    return fig


def create_heatmap(df, metrics, title):
    """Heatmap of mean scores by type × metric."""
    means = df.groupby("type", observed=True)[metrics].mean().T
    fig = px.imshow(
        means,
        labels=dict(x="Question Type", y="Metric", color="Mean Score"),
        x=means.columns.astype(str),
        y=means.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(height=500)
    return fig


def create_radar_by_type(df, metrics, title):
    """Radar / spider chart comparing mean metrics across types."""
    means = df.groupby("type", observed=True)[metrics].mean()
    fig = go.Figure()
    for t in means.index:
        fig.add_trace(go.Scatterpolar(
            r=means.loc[t].values.tolist() + [means.loc[t].values[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=str(t),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        height=550,
        showlegend=True,
    )
    return fig


def main():
    st.markdown('<p class="main-header">📊 Laudon MIS Evaluation Metrics Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Interactive analysis of LLM-as-a-Judge and Non-LLM metrics across question types</p>',
        unsafe_allow_html=True
    )

    df_llm, df_nollm = load_data()

    # Sidebar filters
    st.sidebar.header("Filters & Options")
    all_types = list(df_llm["type"].cat.categories)
    selected_types = st.sidebar.multiselect(
        "Select question types",
        options=all_types,
        default=all_types,
    )
    if not selected_types:
        st.warning("Please select at least one question type.")
        return

    df_llm_f = df_llm[df_llm["type"].isin(selected_types)].copy()
    df_nollm_f = df_nollm[df_nollm["type"].isin(selected_types)].copy()

    show_raw = st.sidebar.checkbox("Show raw data tables", value=False)

    # Key metrics at top
    st.subheader("Key Aggregates (selected types)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Questions", len(df_llm_f))
    with col2:
        st.metric("Avg Input Tokens", f"{df_llm_f['input_token_count'].mean():.0f}")
    with col3:
        st.metric("Avg Output Tokens", f"{df_llm_f['output_token_count'].mean():.0f}")
    with col4:
        st.metric("Avg Latency (s)", f"{df_llm_f['generation_latency_s'].mean():.2f}")
    with col5:
        st.metric("Avg LLM Calls", f"{df_llm_f['generation_llm_calls'].mean():.1f}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview by Type",
        "⚙️ Generation Metrics",
        "🧠 LLM-as-a-Judge",
        "📏 Non-LLM Metrics",
        "🔀 Side-by-Side Comparison",
        "📋 Data Explorer",
    ])

    # ------------------------------------------------------------------
    # TAB 1 – Overview by Type
    # ------------------------------------------------------------------
    with tab1:
        st.header("Average Metrics by Question Type")

        # Generation summary
        gen_cols = [
            "input_token_count", "output_token_count", "elapsed_time",
            "generation_llm_calls", "generation_input_tokens",
            "generation_output_tokens", "generation_latency_s"
        ]
        st.subheader("Generation Cost & Latency (Mean / Median)")
        gen_summary = mean_median_table(df_llm_f, gen_cols)
        st.dataframe(gen_summary, use_container_width=True, hide_index=True)

        # Bar charts for generation
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                create_bar_by_type(df_llm_f, "generation_latency_s", "Mean Generation Latency (s) by Type"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                create_bar_by_type(df_llm_f, "generation_input_tokens", "Mean Generation Input Tokens by Type"),
                use_container_width=True
            )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                create_bar_by_type(df_llm_f, "input_token_count", "Mean Question Input Tokens by Type"),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                create_bar_by_type(df_llm_f, "generation_llm_calls", "Mean LLM Calls by Type"),
                use_container_width=True
            )

    # ------------------------------------------------------------------
    # TAB 2 – Generation Metrics detail
    # ------------------------------------------------------------------
    with tab2:
        st.header("Question & Agent Generation Metrics")

        st.markdown("#### Distribution of Token Counts and Latency")
        fig_box = create_box_by_type(
            df_llm_f,
            ["input_token_count", "output_token_count", "generation_input_tokens", "generation_output_tokens"],
            "Token Counts Distribution by Type"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Latency scatter
        st.markdown("#### Latency vs Tokens")
        fig_scatter = px.scatter(
            df_llm_f,
            x="generation_input_tokens",
            y="generation_latency_s",
            color="type",
            size="generation_llm_calls",
            hover_data=["question", "generation_output_tokens"],
            title="Generation Latency vs Input Tokens (size = # LLM calls)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Line-style comparison of means (ordered by type)
        st.markdown("#### Mean Metrics Across Types (line view)")
        mean_gen = df_llm_f.groupby("type", observed=True)[
            ["input_token_count", "generation_input_tokens", "generation_latency_s", "generation_llm_calls"]
        ].mean().reset_index()
        # Normalize for visual comparison
        for col in ["input_token_count", "generation_input_tokens", "generation_latency_s", "generation_llm_calls"]:
            mean_gen[f"{col}_norm"] = (mean_gen[col] - mean_gen[col].min()) / (mean_gen[col].max() - mean_gen[col].min() + 1e-9)

        fig_line = go.Figure()
        for col, name in [
            ("input_token_count_norm", "Input Tokens (norm)"),
            ("generation_input_tokens_norm", "Gen Input Tokens (norm)"),
            ("generation_latency_s_norm", "Latency (norm)"),
            ("generation_llm_calls_norm", "LLM Calls (norm)"),
        ]:
            fig_line.add_trace(go.Scatter(
                x=mean_gen["type"].astype(str),
                y=mean_gen[col],
                mode="lines+markers",
                name=name,
            ))
        fig_line.update_layout(
            title="Normalized Mean Generation Metrics by Type",
            xaxis_title="Type",
            yaxis_title="Normalized value (0-1)",
            height=420,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3 – LLM-as-a-Judge metrics
    # ------------------------------------------------------------------
    with tab3:
        st.header("LLM-as-a-Judge Metrics (from laudon_evaluated.csv)")

        llm_metrics = [
            "context_precision", "context_recall", "context_entities_recall",
            "noise_sensitivity", "response_relevancy", "faithfulness",
            "answer_accuracy", "context_relevance", "response_groundedness",
            "factual_correctness",
        ]
        # Also the string similarity ones that appear in the same file
        extra_sim = ["nonllm_string_similarity", "bleu", "chrf", "rouge"]

        st.subheader("Mean Scores by Type")
        llm_summary = mean_median_table(df_llm_f, llm_metrics)
        st.dataframe(llm_summary, use_container_width=True, hide_index=True)

        # Heatmap
        st.plotly_chart(
            create_heatmap(df_llm_f, llm_metrics, "Mean LLM-Judge Scores by Type × Metric"),
            use_container_width=True
        )

        # Radar
        st.plotly_chart(
            create_radar_by_type(df_llm_f, llm_metrics[:6], "Radar: Core LLM-Judge Metrics by Type"),
            use_container_width=True
        )

        # Individual bar charts in a grid
        st.subheader("Individual Metric Breakdowns")
        cols = st.columns(3)
        for i, m in enumerate(llm_metrics):
            with cols[i % 3]:
                st.plotly_chart(
                    create_bar_by_type(df_llm_f, m, m.replace("_", " ").title()),
                    use_container_width=True
                )

        # String similarity from the same file
        st.subheader("String Similarity Metrics (also in LLM file)")
        st.plotly_chart(
            create_box_by_type(df_llm_f, extra_sim, "String Similarity Scores by Type"),
            use_container_width=True
        )

    # ------------------------------------------------------------------
    # TAB 4 – Non-LLM metrics
    # ------------------------------------------------------------------
    with tab4:
        st.header("Non-LLM Metrics (from laudon_evaluated_nollm.csv)")

        nollm_metrics = ["metric_bleu", "metric_chrf", "metric_rouge_l"]

        st.subheader("Per-Question String Metrics (Mean / Median)")
        nollm_summary = mean_median_table(df_nollm_f, nollm_metrics)
        st.dataframe(nollm_summary, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                create_bar_by_type(df_nollm_f, "metric_bleu", "Mean BLEU by Type"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                create_bar_by_type(df_nollm_f, "metric_rouge_l", "Mean ROUGE-L by Type"),
                use_container_width=True
            )

        st.plotly_chart(
            create_box_by_type(df_nollm_f, nollm_metrics, "Non-LLM Metric Distributions by Type"),
            use_container_width=True
        )

        # Corpus-level metrics (they are constant, so just display once)
        st.subheader("Corpus-Level Retrieval Metrics")
        corpus_cols = [
            "corpus_mrr", "corpus_map", "corpus_ndcg@1", "corpus_ndcg@3",
            "corpus_ndcg@5", "corpus_ndcg@10", "corpus_map@1", "corpus_map@3",
            "corpus_map@5", "corpus_map@10", "corpus_bleu_score",
            "corpus_chrf_score", "corpus_rouge_score"
        ]
        # Take first non-null row
        corpus_vals = df_nollm_f[corpus_cols].iloc[0]
        corpus_df = pd.DataFrame({
            "Metric": corpus_vals.index,
            "Value": corpus_vals.values.round(4)
        })
        st.dataframe(corpus_df, use_container_width=True, hide_index=True)

        # Nice bar for corpus metrics
        fig_corpus = px.bar(
            corpus_df,
            x="Metric", y="Value",
            title="Corpus-Level Metrics (same for all rows)",
            color="Value",
            color_continuous_scale="Viridis",
            text_auto=".3f"
        )
        fig_corpus.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
        st.plotly_chart(fig_corpus, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 5 – Side-by-side comparison
    # ------------------------------------------------------------------
    with tab5:
        st.header("Side-by-Side: LLM-Judge vs Non-LLM String Metrics")

        # Merge on a key if possible (question + type)
        # Both files share the same questions in the same order
        df_llm_f = df_llm_f.reset_index(drop=True)
        df_nollm_f = df_nollm_f.reset_index(drop=True)

        comparison = pd.DataFrame({
            "type": df_llm_f["type"],
            "question": df_llm_f["question"].str[:80] + "...",
            "llm_bleu": df_llm_f["bleu"],
            "nollm_bleu": df_nollm_f["metric_bleu"],
            "llm_chrf": df_llm_f["chrf"],
            "nollm_chrf": df_nollm_f["metric_chrf"],
            "llm_rouge": df_llm_f["rouge"],
            "nollm_rouge": df_nollm_f["metric_rouge_l"],
            "answer_accuracy": df_llm_f["answer_accuracy"],
            "faithfulness": df_llm_f["faithfulness"],
        })

        st.subheader("Correlation: LLM string scores vs Non-LLM metrics")
        corr_cols = ["llm_bleu", "nollm_bleu", "llm_chrf", "nollm_chrf", "llm_rouge", "nollm_rouge", "answer_accuracy", "faithfulness"]
        corr = comparison[corr_cols].corr().round(3)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            aspect="auto",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter BLEU vs answer_accuracy
        st.subheader("Non-LLM BLEU vs LLM Answer Accuracy")
        fig_sc = px.scatter(
            comparison,
            x="nollm_bleu",
            y="answer_accuracy",
            color="type",
            hover_data=["question"],
            title="Non-LLM BLEU vs LLM Answer Accuracy",
            trendline="ols",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Mean comparison table
        st.subheader("Mean Scores Side-by-Side by Type")
        side_means = comparison.groupby("type", observed=True)[
            ["llm_bleu", "nollm_bleu", "llm_chrf", "nollm_chrf", "llm_rouge", "nollm_rouge", "answer_accuracy", "faithfulness"]
        ].mean().round(3)
        st.dataframe(side_means, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 6 – Data Explorer
    # ------------------------------------------------------------------
    with tab6:
        st.header("Raw Data Explorer")

        view = st.radio("Which file?", ["LLM-as-a-Judge (laudon_evaluated.csv)", "Non-LLM (laudon_evaluated_nollm.csv)"], horizontal=True)

        if "LLM-as-a-Judge" in view:
            st.dataframe(
                df_llm_f[["type", "chapter", "question", "generation_answer",
                          "input_token_count", "output_token_count", "generation_latency_s",
                          "context_precision", "context_recall", "faithfulness",
                          "answer_accuracy", "response_relevancy"]].reset_index(drop=True),
                use_container_width=True,
                height=500,
            )
        else:
            st.dataframe(
                df_nollm_f[["type", "chapter", "question", "generation_answer",
                           "metric_bleu", "metric_chrf", "metric_rouge_l",
                           "generation_latency_s"]].reset_index(drop=True),
                use_container_width=True,
                height=500,
            )

        if show_raw:
            st.subheader("Full tables")
            st.write("LLM file")
            st.dataframe(df_llm_f, use_container_width=True)
            st.write("Non-LLM file")
            st.dataframe(df_nollm_f, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Data sources: laudon_evaluated.csv (LLM-as-a-Judge) & laudon_evaluated_nollm.csv (string + corpus metrics)")


if __name__ == "__main__":
    main()
