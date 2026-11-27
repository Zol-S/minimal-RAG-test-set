"""Invoke an external RAG agent via HTTP (or legacy CLI entrypoint)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import AgentConfig


@dataclass
class AgentResponse:
    """Structured result returned by the agent."""

    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    # Full chunk objects when the agent returns structured chunks
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    # Optional extra fields the agent may return (llm_calls, tokens, …)
    extra: Dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    """
    Call the agent defined in agent.yaml.

    Supported modes
    ---------------
    http  – POST JSON ``{question_field: question, ...}`` to ``url``.
            Expected response (example)::

                {
                  "agent": "basic",
                  "answer": "...",
                  "chunks": [
                    {"chunk_id": 7032, "title": "...", "score": 0.267, "text": "..."},
                    ...
                  ],
                  "llm_calls": 2,
                  "input_tokens": 762,
                  "output_tokens": 29,
                  "latency_s": 18.91
                }

    cli   – legacy subprocess entrypoint (plain text or JSON on stdout).
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def invoke(self, question: str) -> AgentResponse:
        mode = (getattr(self.config, "mode", None) or "http").lower()
        if mode == "http":
            return self._invoke_http(question)
        return self._invoke_cli(question)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _invoke_http(self, question: str) -> AgentResponse:
        cfg = self.config
        url = getattr(cfg, "url", None)
        if not url:
            return AgentResponse(
                success=False,
                error="agent.url is not configured in agent.yaml",
            )

        body: Dict[str, Any] = dict(getattr(cfg, "request_body", None) or {})
        body[getattr(cfg, "question_field", "question")] = question

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        extra_headers = getattr(cfg, "headers", None) or {}
        headers.update(extra_headers)

        payload = json.dumps(body).encode("utf-8")
        method = (getattr(cfg, "method", None) or "POST").upper()
        req = Request(url, data=payload, headers=headers, method=method)

        t0 = time.perf_counter()
        try:
            with urlopen(req, timeout=cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", None) or resp.getcode()
            latency = (time.perf_counter() - t0) * 1000.0

            if status and int(status) >= 400:
                return AgentResponse(
                    raw_stdout=raw,
                    latency_ms=latency,
                    success=False,
                    error=f"HTTP {status}: {raw[:500]}",
                )

            return self._parse_json_payload(raw, latency)

        except HTTPError as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(exc)
            return AgentResponse(
                raw_stdout=err_body,
                latency_ms=latency,
                success=False,
                error=f"HTTP {exc.code}: {err_body[:500]}",
            )
        except URLError as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                latency_ms=latency,
                success=False,
                error=f"Connection error: {exc.reason}",
            )
        except TimeoutError:
            latency = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                latency_ms=latency,
                success=False,
                error=f"Agent timed out after {cfg.timeout_s}s",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                latency_ms=latency,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_json_payload(self, raw: str, latency: float) -> AgentResponse:
        """Parse the HTTP / JSON agent response into AgentResponse."""
        cfg = self.config
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            if getattr(cfg, "plain_text_fallback", True) and raw.strip():
                return AgentResponse(
                    answer=raw.strip(),
                    raw_stdout=raw,
                    latency_ms=latency,
                    success=True,
                )
            return AgentResponse(
                raw_stdout=raw,
                latency_ms=latency,
                success=False,
                error=f"Invalid JSON from agent: {exc}",
            )

        if not isinstance(data, dict):
            return AgentResponse(
                raw_stdout=raw,
                latency_ms=latency,
                success=False,
                error="Agent JSON root is not an object",
            )

        answer_key = getattr(cfg, "answer_key", "answer")
        contexts_key = getattr(cfg, "contexts_key", "chunks")
        contexts_key_fallback = getattr(cfg, "contexts_key_fallback", "contexts")

        answer = str(data.get(answer_key, "") or "")

        chunks_raw = data.get(contexts_key)
        if chunks_raw is None and contexts_key_fallback:
            chunks_raw = data.get(contexts_key_fallback)

        contexts: List[str] = []
        chunks: List[Dict[str, Any]] = []

        if isinstance(chunks_raw, list):
            for c in chunks_raw:
                if isinstance(c, str):
                    if c.strip():
                        contexts.append(c.strip())
                elif isinstance(c, dict):
                    chunks.append(c)
                    text = c.get("text") or c.get("content") or c.get("page_content")
                    if text:
                        contexts.append(str(text).strip())

        known_extra = (
            "agent",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "latency_s",
            "model",
        )
        extra = {k: data[k] for k in known_extra if k in data}
        for k, v in data.items():
            if k in (answer_key, contexts_key, contexts_key_fallback):
                continue
            if k not in extra:
                extra[k] = v

        if "latency_s" in data and data["latency_s"] is not None:
            try:
                latency = float(data["latency_s"]) * 1000.0
            except (TypeError, ValueError):
                pass

        return AgentResponse(
            answer=answer,
            contexts=contexts,
            chunks=chunks,
            raw_stdout=raw,
            latency_ms=latency,
            success=True,
            extra=extra,
        )

    def _parse_output(
        self, stdout: str, stderr: str, latency: float
    ) -> AgentResponse:
        """CLI path: try JSON first, then plain-text fallback."""
        cfg = self.config
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                result = self._parse_json_payload(stdout, latency)
                result.raw_stderr = stderr
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        if getattr(cfg, "plain_text_fallback", True) and stdout:
            return AgentResponse(
                answer=stdout,
                raw_stdout=stdout,
                raw_stderr=stderr,
                latency_ms=latency,
                success=True,
            )

        return AgentResponse(
            raw_stdout=stdout,
            raw_stderr=stderr,
            latency_ms=latency,
            success=False,
            error="Could not parse agent output",
        )