"""YAML configuration loaders for agent and evaluation settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Defaults used when evaluation.yaml has no metrics block (UI overrides these).
DEFAULT_METRICS = {'context_precision': False, 'context_recall': True, 'context_entities_recall': True, 'context_relevance': True, 'faithfulness': True, 'response_groundedness': True, 'answer_accuracy': True, 'factual_correctness': True, 'noise_sensitivity': False, 'response_relevancy': True, 'nonllm_string_similarity': True, 'ragas_bleu': False, 'ragas_chrf': False, 'ragas_rouge': False, 'metric_bleu': True, 'metric_chrf': True, 'metric_rouge_l': True, 'corpus_mrr': True, 'corpus_map': True, 'corpus_ndcg@1': True, 'corpus_ndcg@3': True, 'corpus_ndcg@5': True, 'corpus_ndcg@10': True, 'corpus_map@1': True, 'corpus_map@3': True, 'corpus_map@5': True, 'corpus_map@10': True}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_llm_config(
    config_path: Optional[str],
    entry: str,
) -> Dict[str, Any]:
    """
    Load one named entry from llm.yaml.

    Returns a dict with at least:
      provider, model, api_key, base_url

    Also injects the API key into the process environment so libraries that
    only read env vars (OpenAI SDK, Ragas helpers, etc.) can find it.
    """
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if entry not in config:
        raise KeyError(
            f"Entry '{entry}' not found in {config_path}. "
            f"Available keys: {list(config.keys())}"
        )

    entry_cfg = config[entry] or {}
    provider = entry_cfg.get("provider") or ""
    if not provider:
        raise ValueError(f"No 'provider' key under '{entry}' in config.")

    api_key = entry_cfg.get("api_key") or entry_cfg.get("api-key") or ""
    base_url = entry_cfg.get("base_url") or entry_cfg.get("base-url") or entry_cfg.get("api_base")
    model = entry_cfg.get("model")

    # Derive model from provider string when missing, e.g. "anthropic/claude-haiku-5-5"
    if not model and "/" in str(provider):
        model = str(provider).split("/", 1)[1]
    if not model:
        model = str(provider)

    # Inject credentials into the environment for SDKs that only look there.
    # Always set OPENAI_API_KEY as well: Ragas uses the OpenAI-compatible client.
    if api_key:
        provider_l = str(provider).lower()
        if "anthropic" in provider_l or "claude" in provider_l:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        # Ragas / AsyncOpenAI always read OPENAI_API_KEY unless api_key= is passed
        os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", str(base_url))

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "raw": entry_cfg,
    }


@dataclass
class AgentConfig:
    """How to invoke the RAG agent (HTTP service or legacy CLI)."""

    # Transport: "http" or "cli"
    mode: str = "http"

    # --- HTTP ---
    url: Optional[str] = "http://localhost:8000/agent"
    method: str = "POST"
    question_field: str = "question"
    request_body: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    # --- Shared ---
    timeout_s: float = 120.0
    answer_key: str = "answer"
    contexts_key: str = "chunks"
    contexts_key_fallback: str = "contexts"
    plain_text_fallback: bool = True

    # --- CLI (legacy; kept so UI / CLI code can still set cwd) ---
    entrypoint: str = "python agents/dummy_agent.py"
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    question_mode: str = "arg"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        data = load_yaml(path)
        # Support nested under "agent:" or flat
        if "agent" in data and isinstance(data["agent"], dict):
            data = data["agent"]
        headers = data.get("headers") or {}
        # Drop commented-style nulls / non-string values
        headers = {str(k): str(v) for k, v in headers.items() if v is not None}
        return cls(
            mode=str(data.get("mode", "http")).lower(),
            url=data.get("url", cls.url),
            method=str(data.get("method", "POST")).upper(),
            question_field=data.get("question_field", "question"),
            request_body=dict(data.get("request_body") or {}),
            headers=headers,
            timeout_s=float(data.get("timeout_s", 120.0)),
            answer_key=data.get("answer_key", "answer"),
            contexts_key=data.get("contexts_key", "chunks"),
            contexts_key_fallback=data.get("contexts_key_fallback", "contexts"),
            plain_text_fallback=bool(data.get("plain_text_fallback", True)),
            entrypoint=data.get("entrypoint", cls.entrypoint),
            extra_args=list(data.get("extra_args") or []),
            env=dict(data.get("env") or {}),
            cwd=data.get("cwd"),
            question_mode=data.get("question_mode", "arg"),
        )


@dataclass
class EvaluationConfig:
    """Which metrics to run and how."""

    # metric_name -> enabled
    metrics: Dict[str, bool] = field(default_factory=dict)
    # Ranking
    relevance_threshold: float = 0.25
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    # LLM judge
    llm_config_path: Optional[str] = None
    llm_entry: str = "claude_5_5"
    embedding_model: str = "text-embedding-3-small"
    # Resolved from llm.yaml (filled by from_yaml / apply_llm_credentials)
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    use_llm: bool = True
    use_ranking: bool = True
    use_lexical: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path, base_dir: str | Path | None = None) -> "EvaluationConfig":
        path = Path(path)
        data = load_yaml(path)
        metrics = data.get("metrics") or {}
        # Normalise to bool; fall back to package defaults when the YAML
        # has no metrics block (selection is done in the Streamlit UI).
        if metrics:
            metrics = {k: bool(v) for k, v in metrics.items()}
        else:
            metrics = dict(DEFAULT_METRICS)
        ranking = data.get("ranking") or {}
        llm = data.get("llm") or {}
        llm_path = llm.get("config_path")
        # Resolve relative LLM config paths against base_dir (project root)
        # or the directory of the evaluation yaml.
        if llm_path:
            lp = Path(llm_path)
            if not lp.is_absolute():
                root = Path(base_dir) if base_dir else path.parent.parent
                candidate = root / lp
                if candidate.exists():
                    llm_path = str(candidate)
                else:
                    # also try relative to the eval yaml directory
                    alt = path.parent / lp
                    if alt.exists():
                        llm_path = str(alt)
                    else:
                        llm_path = str(candidate)
        llm_entry = llm.get("entry", "claude_5_5")
        embedding_model = llm.get("embedding_model", "text-embedding-3-small")

        # Eagerly resolve credentials from llm.yaml so the evaluator never
        # constructs AsyncOpenAI without an api_key.
        llm_api_key = None
        llm_base_url = None
        llm_model = None
        llm_provider = None
        if llm_path:
            try:
                resolved = load_llm_config(llm_path, llm_entry)
                llm_api_key = resolved.get("api_key") or None
                llm_base_url = resolved.get("base_url") or None
                llm_model = resolved.get("model") or None
                llm_provider = resolved.get("provider") or None
            except Exception as exc:
                # Keep going; _ensure_ragas will surface a clearer error later
                print(f"[warn] Could not load LLM config '{llm_path}' entry '{llm_entry}': {exc}")

        return cls(
            metrics=metrics,
            relevance_threshold=float(
                ranking.get("relevance_threshold", 0.25)
            ),
            k_values=list(ranking.get("k_values") or [1, 3, 5, 10]),
            llm_config_path=llm_path,
            llm_entry=llm_entry,
            embedding_model=embedding_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_provider=llm_provider,
            use_llm=bool(data.get("use_llm", True)),
            use_ranking=bool(data.get("use_ranking", True)),
            use_lexical=bool(data.get("use_lexical", True)),
        )

    def apply_metric_selection(self, selected: Dict[str, bool]) -> None:
        """Override enabled metrics from the UI checkboxes.

        Also tweaks master switches so we do not spin up LLM / ranking
        machinery when nothing in that family is selected.
        """
        self.metrics = {k: bool(v) for k, v in selected.items()}
        lexical = {"metric_bleu", "metric_chrf", "metric_rouge_l"}
        ranking_prefixes = ("corpus_",)
        llm_keys = [
            k for k in self.metrics
            if k not in lexical and not k.startswith("corpus_")
        ]
        self.use_lexical = any(self.metrics.get(k) for k in lexical)
        self.use_ranking = any(
            self.metrics.get(k) for k in self.metrics if k.startswith("corpus_")
        )
        self.use_llm = any(self.metrics.get(k) for k in llm_keys)

    def is_enabled(self, name: str) -> bool:
        """Return True if the metric should be computed."""
        # Lexical always-on aliases
        if name in ("metric_bleu", "metric_chrf", "metric_rouge_l"):
            return self.use_lexical and self.metrics.get(name, True)
        if name.startswith("corpus_"):
            return self.use_ranking and self.metrics.get(name, True)
        return self.metrics.get(name, False)

    def enabled_metric_names(self) -> List[str]:
        return [k for k, v in self.metrics.items() if v]
