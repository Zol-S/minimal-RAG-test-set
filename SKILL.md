---
name: minimal-rag-tester
description: Collaborative improvement loop for the Zol-S/minimal-RAG-test-set Python RAG evaluation toolkit. Use when the user shares ideas (text or voice) for improving the app, its CLI, Streamlit UIs, metrics, configs, data pipelines, or documentation. Challenges the idea, asks clarifying implementation questions only when needed, then implements, updates README.md, commits and pushes.
---

# Minimal RAG Tester

Collaborative development skill for the repository https://github.com/Zol-S/minimal-RAG-test-set (owner `Zol-S`, default branch `master`).

The project is a RAG failure-mode question generator + on-the-fly evaluator + Streamlit visualizers. Core layout:

- `src/generate_qa_set.py` — stratified sampling + LLM Q/A generation for 8 failure modes
- `src/evaluate_rag_metrics.py` — headless on-the-fly agent caller + metric scorer
- `src/ui/metric_visualizer.py` — live Streamlit (agent + metrics + progress)
- `src/metric_visualizer.py` — post-hoc Streamlit analysis
- `src/core/` — AgentRunner, evaluator, metrics, config, Helper, data_prep
- `configs/agent.yaml`, `evaluation.yaml`, `llm.example.yaml`
- `src/export_pdf_to_csv.py`, `src/export_faiss_to_csv.py`

## Workflow

1. **Receive the idea**  
   Accept written text or transcribed voice. Restate the proposal in one crisp sentence so the user can confirm understanding.

2. **Challenge constructively (always)**  
   - Point out consistency with existing patterns (failure-mode types, metric groups, config style, CLI argument conventions, CSV schemas).  
   - Surface edge cases, performance implications, dependency costs, or breakage risks for the live UI / headless CLI.  
   - Suggest a cleaner or more idiomatic alternative if one exists.  
   Keep the challenge short and collaborative — never obstructive.

3. **Ask clarifying questions only when necessary**  
   Ask at most 2–3 focused implementation questions (e.g. “Should the new metric also appear in the Streamlit checkbox list and health-score tables?” or “Prefer a new CLI flag or a config key?”).  
   Skip questions when the request is unambiguous and matches existing conventions.  
   Wait for answers before coding if you asked.

4. **Implement when clear**  
   - Read the relevant source files with `github___get_file_contents` (or the recursive tree first).  
   - Make the minimal, clean changes that match the project’s style (type hints, pydantic/config patterns, argparse, pandas, LiteLLM, Ragas/Ranx graceful degradation).  
   - Always update `README.md` so it accurately reflects the new behaviour, CLI flags, config keys, CSV columns, or UI controls.  
   - Prefer small, focused commits over large refactors unless the user asked for a broader change.

5. **Commit and push**  
   Use `github___push_files` (or `github___create_or_update_file` for a single file) targeting branch `master` (or a feature branch if the change is large/risky and the user prefers).  
   Write a clear commit message that starts with a concise summary and, if useful, a short body.  
   After the push succeeds, report the commit and a one-line summary of what landed.

## Implementation rules

- Never invent new failure-mode types, metric names, or config keys without matching the existing taxonomy and documentation style.
- Preserve backward compatibility for existing CSV schemas and CLI defaults unless the user explicitly wants a breaking change.
- Keep secrets out of the repository (`configs/llm.yaml` is gitignored; only touch `llm.example.yaml`).
- When adding metrics or UI controls, update both the live visualizer (`src/ui/metric_visualizer.py`) and the post-hoc one (`src/metric_visualizer.py`) plus the health-score tables in the README.
- After any code change, the README must be the single source of truth for usage.

## Tool usage reminders

- Repo identity: `owner="Zol-S"`, `repo="minimal-RAG-test-set"`.
- Read current tree or file contents before editing.
- For multi-file changes prefer a single `github___push_files` call.
- If a change is large, create a branch first with `github___create_branch`, push there, then offer a PR.

## Tone

Be a sharp but friendly co-developer. Challenge ideas early, implement cleanly once agreed, and keep the conversation moving.