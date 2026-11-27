"""On-the-fly evaluator: call agent then score enabled metrics per row."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .agent_runner import AgentResponse, AgentRunner
from .config import EvaluationConfig
from .data_prep import extract_contexts, is_relevant_chunk, parse_generation_chunks
from .metrics import (
    HAS_OPENAI,
    HAS_RAGAS,
    MetricResult,
    RankingMetrics,
    RagasMetricRunner,
    compute_bleu,
    compute_chrf,
    compute_rouge_l,
    timed_lexical,
)


@dataclass
class RowEvaluation:
    """All metric results for a single question/answer pair."""

    question: str
    reference: str
    generation_answer: str
    contexts: List[str]
    agent_latency_ms: float = 0.0
    agent_success: bool = True
    agent_error: Optional[str] = None
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    row_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question": self.question,
            "answer": self.reference,
            "generation_answer": self.generation_answer,
            "agent_latency_ms": self.agent_latency_ms,
            "agent_success": self.agent_success,
            "agent_error": self.agent_error,
            "n_contexts": len(self.contexts),
        }
        d.update(self.extra)
        for name, mr in self.metrics.items():
            d[name] = mr.value
            d[f"{name}_latency_ms"] = mr.latency_ms
            d[f"{name}_input_tokens"] = mr.input_tokens
            d[f"{name}_output_tokens"] = mr.output_tokens
            d[f"{name}_confidence"] = mr.confidence
            d[f"{name}_margin_of_error"] = mr.margin_of_error
            if mr.error:
                d[f"{name}_error"] = mr.error
        return d


class OnTheFlyEvaluator:
    """
    Loop over questions, invoke the agent, evaluate enabled metrics, and
    optionally push live progress via a callback.
    """

    def __init__(
        self,
        agent: AgentRunner,
        eval_config: EvaluationConfig,
        progress_callback: Optional[Callable[[int, int, RowEvaluation, float], None]] = None,
    ):
        self.agent = agent
        self.cfg = eval_config
        self.progress_callback = progress_callback
        self._ragas: Optional[RagasMetricRunner] = None
        self._ranking = RankingMetrics(
            k_values=eval_config.k_values,
            relevance_threshold=eval_config.relevance_threshold,
        )

    def _ensure_ragas(self) -> Optional[RagasMetricRunner]:
        if not self.cfg.use_llm:
            return None
        if self._ragas is not None:
            return self._ragas
        if not (HAS_RAGAS and HAS_OPENAI):
            return None

        # Credentials / model from EvaluationConfig (already resolved from llm.yaml)
        api_key = getattr(self.cfg, "llm_api_key", None)
        base_url = getattr(self.cfg, "llm_base_url", None)
        model = getattr(self.cfg, "llm_model", None) or "gpt-4o"

        # Fallback: load llm.yaml now if from_yaml did not resolve it
        if self.cfg.llm_config_path and not api_key:
            from .config import load_llm_config

            resolved = load_llm_config(self.cfg.llm_config_path, self.cfg.llm_entry)
            api_key = resolved.get("api_key") or api_key
            base_url = resolved.get("base_url") or base_url
            model = resolved.get("model") or model

        if not api_key:
            raise RuntimeError(
                "LLM metrics are enabled but no API key was found. "
                "Set llm.config_path and llm.entry in evaluation.yaml to point "
                "at an entry in configs/llm.yaml that contains api_key, "
                "or export OPENAI_API_KEY."
            )

        enabled = {
            k: v
            for k, v in self.cfg.metrics.items()
            if k
            not in (
                "metric_bleu",
                "metric_chrf",
                "metric_rouge_l",
            )
            and not k.startswith("corpus_")
        }
        self._ragas = RagasMetricRunner(
            model_name=model,
            embedding_model=self.cfg.embedding_model,
            enabled_metrics=enabled,
            api_key=api_key,
            base_url=base_url,
        )
        return self._ragas

    def evaluate_lexical(
        self, pred: str, ref: str
    ) -> Dict[str, MetricResult]:
        out: Dict[str, MetricResult] = {}
        if self.cfg.is_enabled("metric_bleu"):
            out["metric_bleu"] = timed_lexical(compute_bleu, pred, ref)
        if self.cfg.is_enabled("metric_chrf"):
            out["metric_chrf"] = timed_lexical(compute_chrf, pred, ref)
        if self.cfg.is_enabled("metric_rouge_l"):
            out["metric_rouge_l"] = timed_lexical(compute_rouge_l, pred, ref)
        return out

    async def evaluate_row_async(
        self,
        question: str,
        reference: str = "",
        precomputed_answer: Optional[str] = None,
        precomputed_contexts: Optional[List[str]] = None,
        row_index: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RowEvaluation:
        """
        Call the agent (unless answer is already provided) then score.
        """
        if precomputed_answer is not None:
            answer = precomputed_answer
            contexts = precomputed_contexts or []
            agent_lat = 0.0
            success = True
            error = None
            agent_extra: Dict[str, Any] = {}
            agent_chunks: list = []
        else:
            resp: AgentResponse = self.agent.invoke(question)
            answer = resp.answer
            contexts = resp.contexts
            agent_lat = resp.latency_ms
            success = resp.success
            error = resp.error
            agent_extra = dict(resp.extra or {})
            agent_chunks = list(getattr(resp, "chunks", None) or [])

        merged_extra = dict(extra or {})
        # Promote common agent telemetry into the row
        for key in (
            "agent",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "latency_s",
            "model",
        ):
            if key in agent_extra and key not in merged_extra:
                merged_extra[key] = agent_extra[key]
        if agent_chunks:
            merged_extra["generation_chunks"] = agent_chunks
        # Keep any other extras under agent_* prefix if not already taken
        for k, v in agent_extra.items():
            if k not in merged_extra:
                merged_extra[k] = v

        row = RowEvaluation(
            question=question,
            reference=reference or "",
            generation_answer=answer,
            contexts=contexts,
            agent_latency_ms=agent_lat,
            agent_success=success,
            agent_error=error,
            row_index=row_index,
            extra=merged_extra,
        )

        # Lexical (always cheap)
        row.metrics.update(self.evaluate_lexical(answer, reference or ""))

        # Ragas / LLM metrics
        ragas = self._ensure_ragas()
        if ragas is not None:
            llm_scores = await ragas.score_row(
                question=question,
                response=answer,
                reference=reference or "",
                contexts=contexts,
            )
            # Filter to enabled only (runner already filters, but double-check)
            for name, mr in llm_scores.items():
                if self.cfg.is_enabled(name):
                    row.metrics[name] = mr

        return row

    def evaluate_row_sync(
        self,
        question: str,
        reference: str = "",
        precomputed_answer: Optional[str] = None,
        precomputed_contexts: Optional[List[str]] = None,
        row_index: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RowEvaluation:
        """Synchronous wrapper (runs lexical + agent; skips async LLM if no loop)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop (e.g. Streamlit) – create task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.evaluate_row_async(
                            question,
                            reference,
                            precomputed_answer,
                            precomputed_contexts,
                            row_index,
                            extra,
                        ),
                    )
                    return future.result()
            return loop.run_until_complete(
                self.evaluate_row_async(
                    question,
                    reference,
                    precomputed_answer,
                    precomputed_contexts,
                    row_index,
                    extra,
                )
            )
        except RuntimeError:
            return asyncio.run(
                self.evaluate_row_async(
                    question,
                    reference,
                    precomputed_answer,
                    precomputed_contexts,
                    row_index,
                    extra,
                )
            )

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = "question",
        answer_col: str = "answer",
        call_agent: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate every row. If call_agent=True the agent is invoked per row;
        otherwise existing generation_answer / generation_chunks columns are used.
        """
        if limit is not None:
            df = df.head(limit).copy()
        n = len(df)
        results: List[RowEvaluation] = []
        t_start = time.perf_counter()

        for i, row in df.iterrows():
            q = str(row.get(question_col) or "")
            ref = str(row.get(answer_col) or "")
            extra = {
                k: row[k]
                for k in ("type", "source", "chapter")
                if k in row and pd.notna(row[k])
            }

            if call_agent:
                ev = self.evaluate_row_sync(
                    question=q,
                    reference=ref,
                    row_index=int(i) if isinstance(i, (int, np.integer)) else 0,
                    extra=extra,
                )
            else:
                gen_ans = str(row.get("generation_answer") or "")
                chunks = parse_generation_chunks(row.get("generation_chunks"))
                contexts = extract_contexts(chunks)
                # Also accept plain list of strings
                if not contexts and isinstance(row.get("generation_chunks"), list):
                    contexts = [
                        str(c) for c in row["generation_chunks"] if c
                    ]
                ev = self.evaluate_row_sync(
                    question=q,
                    reference=ref,
                    precomputed_answer=gen_ans,
                    precomputed_contexts=contexts,
                    row_index=int(i) if isinstance(i, (int, np.integer)) else 0,
                    extra=extra,
                )

            results.append(ev)
            elapsed = time.perf_counter() - t_start
            if self.progress_callback:
                self.progress_callback(len(results), n, ev, elapsed)

        # Corpus ranking (once, after all rows)
        if self.cfg.use_ranking and any(
            self.cfg.is_enabled(f"corpus_{m}")
            for m in (
                "mrr",
                "map",
                "ndcg@1",
                "ndcg@3",
                "ndcg@5",
                "ndcg@10",
                "map@1",
                "map@3",
                "map@5",
                "map@10",
            )
        ):
            rows_as_dicts = []
            for ev in results:
                # Prefer structured chunks from the agent (with real scores / ids)
                structured = ev.extra.get("generation_chunks") if ev.extra else None
                if structured and isinstance(structured, list):
                    gen_chunks = structured
                else:
                    gen_chunks = [
                        {"text": c, "score": 1.0, "chunk_id": str(i)}
                        for i, c in enumerate(ev.contexts)
                    ]
                rows_as_dicts.append(
                    {
                        "answer": ev.reference,
                        "chunks": " ".join(ev.contexts),
                        "generation_chunks": gen_chunks,
                    }
                )
            ranking = self._ranking.compute(rows_as_dicts, is_relevant_chunk)
            for ev in results:
                for name, mr in ranking.items():
                    key = f"corpus_{name}"
                    if self.cfg.is_enabled(key) or self.cfg.is_enabled(name):
                        ev.metrics[key] = mr

        flat = [r.to_flat_dict() for r in results]
        return pd.DataFrame(flat)
