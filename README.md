# 🧪 RAG Failure-Mode Question Generator (CLI)

A small toolkit that turns documents (PDF or FAISS knowledge bases) into chunked CSV files, generates targeted question–answer sets for evaluating RAG systems on common failure modes, **calls your RAG agent on the fly**, scores answers with a unified metric suite, and visualizes results in Streamlit.

## 📄 Paper

The relevant paper is available here *(link to be added)*.

---

## ⚙️ Installation

From the project root:

```bash
uv add -r requirements.txt
```

Main dependencies include `pymupdf`, `langchain-text-splitters`, `pandas`, `litellm`, `faiss-cpu` (or `faiss-gpu`), `pydantic`, `tqdm`, and `pyyaml`.

**Evaluation extras** (optional, depending on which metrics you need):

```bash
uv add ragas datasets openai tiktoken ranx nltk sentence-transformers
```

**Visualization extras** (Streamlit dashboards):

```bash
uv add streamlit plotly
```

---

## 🎯 What it does

1. **Extract & chunk** text from different sources and save it as a standardized CSV.
2. **Sample** the chunks according to several failure-mode strategies (single-hop, multi-hop, negative/unsupported, linguistic complexity, document maximization, cross-section synthesis, ambiguous, terminological variation).
3. **Generate** question–answer pairs with an LLM that can later be used to test RAG systems.
4. **Call the agent on the fly** (HTTP service configured in `configs/agent.yaml`) for every question.
5. **Evaluate** answers with a unified evaluator (LLM / Ragas metrics, lexical scores, ranking metrics) controlled by `configs/evaluation.yaml` and, in the UI, by metric checkboxes.
6. **Visualize** either:
   - **live** progress while the agent is called and metrics are scored (`src/ui/metric_visualizer.py`), or
   - **post-hoc** analysis of a finished evaluation CSV (`src/metric_visualizer.py`).

---

## 📁 Layout (relevant parts)

```text
project/
├── configs/
│   ├── agent.yaml          # How to call the RAG agent (HTTP or CLI)
│   ├── evaluation.yaml     # LLM / ranking / lexical switches (no per-metric flags)
│   ├── llm.yaml            # API keys + model entries (gitignored secrets)
│   └── llm.example.yaml
├── src/
│   ├── core/               # Shared classes (AgentRunner, OnTheFlyEvaluator, metrics, …)
│   ├── evaluate_rag_metrics.py   # Headless on-the-fly evaluator CLI
│   ├── ui/
│   │   └── metric_visualizer.py  # Live Streamlit: agent + metrics + progress
│   ├── metric_visualizer.py      # Static Streamlit: explore a scored CSV
│   ├── export_pdf_to_csv.py
│   ├── export_faiss_to_csv.py
│   └── generate_qa_set.py
└── data/…
```

---

## 🚀 Quick start

### 1. Prepare LLM config

```bash
cp configs/llm.example.yaml configs/llm.yaml
```

Edit `configs/llm.yaml`, paste your API key(s) and model entries.  
Ragas uses an **OpenAI-compatible** client: prefer an OpenAI-style entry (e.g. `gpt_6_luna`) unless you expose Claude behind a compatible `base_url`.

### 2. Extract chunks

**PDF → CSV:**

```bash
uv run python src/export_pdf_to_csv.py ../data/Laudon_MIS_17th.pdf \
  --output ../data/export/laudon_export.csv \
  --chunk-size 300 --chunk-overlap 0
```

**FAISS → CSV:**

```bash
uv run python src/export_faiss_to_csv.py \
  --source faiss \
  --kb-dir ../../agent-harness-comparison/kb/44d5fde8e5 \
  --output ../data/export/faiss_export.csv
```

### 3. Generate questions

```bash
uv run python src/generate_qa_set.py \
  --input path/to/chunks.csv \
  --output-dir ./output \
  --num-entries 10 \
  --llm-config configs/llm.yaml \
  --llm-entry claude_5_5
```

### 4. Point the evaluator at your running agent

Edit `configs/agent.yaml`:

```yaml
agent:
  mode: http
  url: "http://localhost:8000/"    # your agent endpoint
  method: POST
  question_field: question
  answer_key: answer
  contexts_key: chunks
  timeout_s: 120.0
```

The agent must accept:

```json
{"question": "<text>"}
```

and return JSON similar to:

```json
{
  "agent": "basic",
  "answer": "…",
  "chunks": [
    {"chunk_id": 7032, "title": "…", "score": 0.267, "text": "…"}
  ],
  "llm_calls": 2,
  "input_tokens": 762,
  "output_tokens": 29,
  "latency_s": 18.91
}
```

(`contexts` is accepted as a fallback key for the chunk list.)

### 5. Evaluate on the fly (CLI)

`src/evaluate_rag_metrics.py` loads a **questions** CSV (needs at least `question`; `answer` is used as the gold reference), calls the agent for every row, scores enabled metrics, and writes an enriched CSV + summary JSON.

```bash
# From project root — agent must already be running
uv run python src/evaluate_rag_metrics.py \
  -i data/generated_questions/qa_set_faiss_202611260954.csv \
  --limit 5

# Custom configs / output
uv run python src/evaluate_rag_metrics.py \
  -i questions.csv \
  -o scored.csv \
  --agent-config configs/agent.yaml \
  --eval-config configs/evaluation.yaml \
  --limit 10

# Score an existing CSV that already has generation_answer / generation_chunks
uv run python src/evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --no-agent
```

Default output: `<input>_evaluated.csv` (and a timestamped copy when `Helper.Log` is available) plus `<input>_evaluated.summary.json`.

Every metric also records companion columns where applicable:

- `<metric>_latency_ms`
- `<metric>_input_tokens` / `<metric>_output_tokens`
- `<metric>_confidence` / `<metric>_margin_of_error`

### 6. Live Streamlit evaluation (`src/ui/metric_visualizer.py`)

Calls the agent, scores metrics, and updates progress / charts as each row finishes.

```bash
uv run streamlit run src/ui/metric_visualizer.py

# Optional default questions file
uv run streamlit run src/ui/metric_visualizer.py -- \
  --questions data/generated_questions/qa_set_faiss_202611260954.csv
```

**Sidebar**

| Control | Purpose |
|---------|---------|
| Questions CSV path | Input file (`question` column required; `answer` optional gold) |
| Agent / evaluation config paths | Defaults under `configs/` |
| **Metrics** | Checkboxes — all on by default except **Context Precision** and **Noise Sensitivity** (heavy). Hints mark LLM-heavy metrics. |
| Call agent on-the-fly | Uncheck to score existing `generation_answer` columns |
| Limit rows | Dry-run on a subset |

Progress shows **% done, elapsed time, and ETA**. Live mean scores (± margin of error) and charts refresh after each agent call.

### 7. Post-hoc visualization (`src/metric_visualizer.py`)

Loads a **finished** evaluation CSV (from step 5) for multi-tab analysis and category health scores.

```bash
uv run streamlit run src/metric_visualizer.py -- --csv path/to/scored.csv
```

**Health scores**

| Category | Metrics averaged (higher = healthier) |
|----------|----------------------------------------|
| **Retrieval** | `context_precision`, `context_recall`, `context_entities_recall`, `context_relevance`, `corpus_mrr`, `corpus_map`, `corpus_map@k`, `corpus_ndcg@k` |
| **Generation** | `faithfulness`, `response_groundedness`, `answer_accuracy`, `factual_correctness`, BLEU / CHRF / ROUGE |
| **End-to-end** | `noise_sensitivity` *(inverted)*, `response_relevancy`, `nonllm_string_similarity` |

---

## ⚙️ Configuration reference

### `configs/agent.yaml`

| Key | Description |
|-----|-------------|
| `mode` | `http` (default) or `cli` |
| `url` | Agent endpoint for HTTP mode |
| `method` | Usually `POST` |
| `question_field` | JSON field name for the question (default `question`) |
| `request_body` | Extra static fields merged into every request |
| `headers` | Optional HTTP headers |
| `timeout_s` | Per-call timeout |
| `answer_key` / `contexts_key` | Response keys (`chunks` by default) |
| `entrypoint` / `cwd` / … | Only used when `mode: cli` |

### `configs/evaluation.yaml`

Controls **global** switches and LLM settings. **Per-metric on/off is chosen in the Streamlit UI** (CLI uses package defaults: everything on except `context_precision` and `noise_sensitivity`).

```yaml
use_llm: true
use_ranking: true
use_lexical: true

llm:
  config_path: "configs/llm.yaml"
  entry: "gpt_6_luna"              # OpenAI-compatible entry recommended for Ragas
  embedding_model: "text-embedding-3-small"

ranking:
  relevance_threshold: 0.25
  k_values: [1, 3, 5, 10]
```

### `configs/llm.yaml`

Named entries, for example:

```yaml
gpt_6_luna:
  provider: openai/gpt-6-luna
  model: gpt-6-luna
  api_key: sk-...
  # base_url: "https://…"   # optional OpenAI-compatible gateway

claude_5_5:
  provider: anthropic/claude-haiku-5-5
  api_key: sk-ant-...
  # base_url required if used with Ragas (OpenAI-compatible proxy)
```

---

## 🛠️ Scripts & CLI arguments

### 📄 `src/export_pdf_to_csv.py`

Extracts chapters from a PDF, splits text with LangChain’s `RecursiveCharacterTextSplitter`, writes CSV columns `source`, `chunks`, `chapter`.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_filename` | ✅ | — | Input PDF path |
| `-o`, `--output` | ✅ | — | Output CSV path or directory |
| `--chunk-size` | | `500` | Max characters per chunk (≥ 50) |
| `--chunk-overlap` | | `0` | Overlap between chunks |

---

### 🗄️ `src/export_faiss_to_csv.py`

Exports every chunk from a FAISS KB directory into a CSV.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | ✅ | — | Value for the `source` column |
| `--kb-dir` | ✅ | — | Directory with `chunks.jsonl` and `index.faiss` |
| `-o`, `--output` | | `export.csv` | Output CSV path |

---

### 🤖 `src/generate_qa_set.py`

Stratified sampling for **eight** failure-mode types and optional LLM Q/A generation.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Chunks CSV (`source`, `chunks`, `chapter`) |
| `-o`, `--output-dir` | ✅ | — | Output directory |
| `-n`, `--num-entries` | | `10` | Samples **per** failure-mode type |
| `--llm-config` | | `None` | Path to `llm.yaml` |
| `--llm-entry` | | `claude_5_5` | Entry key inside the YAML |
| `--model` | | `None` | Override model (LiteLLM format) |
| `--max-tokens` | | `1024` | Max tokens per completion |
| `--skip-generation` | | `False` | Sample only; skip LLM |
| `--seed` | | `None` | Reproducible sampling |

#### Failure-mode types

| Type | Description |
|------|-------------|
| `single-hop` | Answerable from a single chunk |
| `multi-hop` | Needs two passages |
| `negative_unsupported_questions` | Answer not in context (rejection expected) |
| `linguistic_complexity` | Hard-to-read / low Flesch passages |
| `document_maximization` | Needs evidence from many chunks (≈ 5) |
| `cross_section_synthesis` | Joins facts across chapters |
| `ambiguous` | Underspecified / vague |
| `terminological_variation` | Synonyms / paraphrases of source wording |

---

### 📊 `src/evaluate_rag_metrics.py` (on-the-fly evaluator)

Calls the agent (unless `--no-agent`), then scores metrics.

**Input CSV** needs at least:

| Column | Required | Description |
|--------|----------|-------------|
| `question` | ✅ | User question |
| `answer` | recommended | Gold / reference answer |
| `generation_answer` | only with `--no-agent` | Precomputed agent answer |
| `generation_chunks` | only with `--no-agent` | Precomputed retrieved chunks |

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Questions CSV |
| `-o`, `--output` | | `<input>_evaluated.csv` | Output path |
| `--agent-config` | | `configs/agent.yaml` | Agent invocation config |
| `--eval-config` | | `configs/evaluation.yaml` | Evaluation / LLM settings |
| `--limit` | | `None` | First N rows only |
| `--no-agent` | | `False` | Do not call the agent; use existing answer columns |
| `--question-col` | | `question` | Question column name |
| `--answer-col` | | `answer` | Gold answer column name |

**Metric groups**

| Group | Examples |
|-------|----------|
| LLM / Ragas | `context_recall`, `context_relevance`, `faithfulness`, `answer_accuracy`, `response_relevancy`, … |
| Lexical | `metric_bleu`, `metric_chrf`, `metric_rouge_l` |
| Ranking (Ranx) | `corpus_mrr`, `corpus_map`, `corpus_ndcg@k`, `corpus_map@k` |

Graceful degradation: missing optional packages (`ragas`, `ranx`, `nltk`, `openai`) simply disable the corresponding groups.

---

### 📊 `src/ui/metric_visualizer.py` (live dashboard)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--questions`, `-q` | | `None` | Default questions CSV path (also editable in the sidebar) |

```bash
uv run streamlit run src/ui/metric_visualizer.py -- --questions path/to/questions.csv
```

---

### 📊 `src/metric_visualizer.py` (post-hoc dashboard)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv`, `-c` | | `None` | Path to a finished evaluation CSV |

```bash
uv run streamlit run src/metric_visualizer.py -- --csv path/to/scored.csv
```

---

## 📊 CSV schemas

### Input for `generate_qa_set.py`

| Column | Type | Description |
|--------|------|-------------|
| `source` | str | Document source |
| `chunks` | str | Text chunk |
| `chapter` | str | Chapter / section id |

### Output of `generate_qa_set.py`

| Column | Description |
|--------|-------------|
| `type` | Failure-mode type |
| `source` | Source document |
| `chapter` | Chapter(s) |
| `question` | Generated question |
| `chunks` | Context used for generation |
| `answer` | Expected answer |
| `model` | Model that produced the Q/A |
| `input_token_count` / `output_token_count` | Token counts |
| `elapsed_time` | Seconds for the LLM call |

### Input for on-the-fly evaluation

| Column | Description |
|--------|-------------|
| `question` | User question |
| `answer` | Gold answer (optional but recommended) |
| `type` / `source` / `chapter` | Optional metadata carried through |

When using `--no-agent`, also provide `generation_answer` and preferably `generation_chunks`.

### Evaluation output

All input columns are preserved. New columns include agent telemetry (`agent_success`, `agent_latency_ms`, `llm_calls`, …), selected metric scores, and per-metric latency / token / confidence / margin-of-error companions.

---

## 📝 Notes

- **Agent must be running** before CLI or live UI evaluation when agent calls are enabled.
- Ragas metrics need an **OpenAI-compatible** API key/endpoint. An Anthropic-only key (`sk-ant-…`) will fail against `api.openai.com` unless you set a compatible `base_url`.
- Embeddings for Response Relevancy (`text-embedding-3-small`) likewise need an OpenAI-compatible embedding endpoint.
- Heavy metrics (especially **Noise Sensitivity** and **Context Precision**) are unchecked by default in the UI; enable them only when you accept the extra cost/latency.
- Ranking relevance is a heuristic (token overlap ≥ threshold with gold context or gold answer). Tune `ranking.relevance_threshold` in `evaluation.yaml` if needed.
- For multi-chunk generation types, sampled chunks are joined with `|` before being sent to the LLM.
- API keys are read from `configs/llm.yaml` (and injected into the environment for SDKs that only look there).
