# 🧪 RAG Failure-Mode Question Generator (CLI)

A small toolkit that turns documents (PDF or FAISS knowledge bases) into chunked CSV files and then generates targeted question–answer sets for evaluating RAG systems on common failure modes.

## 📄 Paper

The relevant paper is available here *(link to be added)*.

---

## ⚙️ Installation

```bash
uv add -r requirements.txt
```

Main dependencies include `pymupdf`, `langchain-text-splitters`, `pandas`, `litellm`, `faiss-cpu` (or `faiss-gpu`), `pydantic`, `tqdm`, and `pyyaml`.

**Evaluation extras** (optional, depending on which metrics you need):

```bash
uv add ragas datasets openai tiktoken ranx nltk sentence-transformers
```

**Visualization extras** (for the Streamlit dashboard):

```bash
uv add streamlit plotly
```

---

## 🎯 What it does

1. **Extract & chunk** text from different sources and save it as a standardized CSV.
2. **Sample** the chunks according to several failure-mode strategies (single-hop, multi-hop, negative/unsupported, linguistic complexity, document maximization, cross-section synthesis, ambiguous, terminological variation).
3. **Generate** question–answer pairs with an LLM (optional) that can later be used to test RAG systems.
4. **Evaluate** agent answers with a **unified** evaluator that produces both LLM-based RAGAS metrics and offline lexical / ranking metrics in a single CSV.
5. **Visualize** the evaluation results in an interactive Streamlit dashboard (`metric_visualizer.py`) with category health scores.

---

## 🚀 Quick start

### 1. Prepare a config

```bash
cp llm.example.yaml llm.yaml
```

Edit `llm.yaml`, paste your API key and select a model of your choice.

### 2. Extract the chunks

There are several connectors implemented:

- 📄 **PDF to CSV** – Detects chapters, extracts text, chunks it, and writes a CSV.
- 🗄️ **FAISS to CSV** – Reads all chunks from a FAISS-backed knowledge base and writes a CSV.

**Example – PDF to CSV export:**

```bash
uv run python export_pdf_to_csv.py ../data/Laudon_MIS_17th.pdf --output ../data/export/laudon_export.csv --chunk-size 300 --chunk-overlap 0
```

**Example – FAISS to CSV export:**

```bash
uv run python export_faiss_to_csv.py --source faiss --kb-dir ../../agent-harness-comparison/kb/44d5fde8e5 --output ../data/export/faiss_export.csv
```

### 3. Sample from exported chunks and generate questions

```bash
uv run python generate_qa_set.py --input path/to/chunks.csv \
  --output-dir ./output \
  --num-entries 10 \
  --llm-config configs/llm.yaml \
  --llm-entry claude_5_5
```

### 4. Send the generated questions to the agent

The output should be saved as CSV and should have the following columns appended to the input CSV:

| Column | Description |
|--------|-------------|
| `agent` | Name of the agent for identification |
| `generation_answer` | The answer outputted by the agent |
| `generation_chunks` | All the chunks returned by the retriever (list of dicts with at least `text` and preferably `score` / `chunk_id`) |
| `generation_llm_calls` | Number of LLM calls |
| `generation_input_tokens` | Total number of input tokens |
| `generation_output_tokens` | Total number of output tokens |
| `generation_latency_s` | Generation latency (s) |

### 5. Evaluate the output

A **single unified script** (`evaluate_rag_metrics.py`) computes both LLM-based and no-LLM metrics and writes them into one CSV.

| Mode | Flag | Metrics |
|------|------|---------|
| LLM + lexical + ranking | default (with `--llm-config`) | RAGAS collections (Context Precision/Recall, Faithfulness, Answer Accuracy, …) + BLEU/CHRF/ROUGE + Ranx ranking |
| Lexical + ranking only | `--no-llm` | Per-row BLEU / CHRF / ROUGE-L + corpus MRR / MAP / nDCG@k |
| Lexical only | `--no-llm --no-ranking` | BLEU / CHRF / ROUGE-L |

**Required columns:** `question`, `answer`, `generation_answer`, `generation_chunks`.  
(`chunks` is optional but improves ranking relevance judgments.)

```bash
# Full evaluation (LLM + lexical + ranking)
uv run python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5

# Custom output path + limit to first 5 rows
uv run python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  -o scored.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5 \
  --limit 5

# Offline only (no LLM calls)
uv run python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --no-llm

# Skip ranking metrics
uv run python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --no-llm --no-ranking
```

Default output is `<input>_all_metrics.csv` plus a companion `<input>_all_metrics.summary.json`.

Every metric also records companion columns:

- `<metric>_latency_ms`
- `<metric>_input_tokens`
- `<metric>_output_tokens`

### 6. Visualize metrics (`metric_visualizer.py`)

Interactive Streamlit dashboard that loads the **unified evaluation CSV** produced in step 5 and provides multi-tab analysis with **category health scores**.

**Launch**

```bash
# Pass the evaluation CSV as a CLI argument
uv run streamlit run metric_visualizer.py -- --csv path/to/scored.csv

# Or open the app and type the path in the sidebar
uv run streamlit run metric_visualizer.py
```

**Expected input**

A single CSV produced by `evaluate_rag_metrics.py` containing the original question columns plus the metric columns listed below.

**Health scores (top of dashboard)**

| Category | Metrics averaged (higher = healthier) |
|----------|----------------------------------------|
| **Retrieval** | `context_precision`, `context_recall`, `context_entities_recall`, `context_relevance`, `corpus_mrr`, `corpus_map`, `corpus_map@k`, `corpus_ndcg@k` |
| **Generation** | `faithfulness`, `response_groundedness`, `answer_accuracy`, `factual_correctness`, `metric_bleu` / `ragas_bleu`, `metric_chrf` / `ragas_chrf`, `metric_rouge_l` / `ragas_rouge` |
| **End-to-end** | `noise_sensitivity` *(inverted)*, `response_relevancy`, `nonllm_string_similarity` |

`noise_sensitivity` is inverted (`1 − score`) so that a higher health score always means better behaviour.

**Dashboard tabs**

| Tab | What you can explore |
|-----|----------------------|
| **Overview by Type** | Health score per question type, heatmap of the three categories, bar charts for key metrics |
| **Retrieval Failure Modes** | Context quality + ranking metrics, heatmaps, box plots, radar charts |
| **Generation Failure Modes** | LLM-as-a-Judge scores and string/n-gram overlap metrics |
| **End-to-End** | Noise sensitivity, response relevancy, semantic similarity |
| **Generation Cost & Latency** | Token counts, latency, scatter of latency vs. input tokens |
| **Data Explorer** | Filterable table of questions, answers, and key scores |

**Sidebar controls**

- Path to the evaluation CSV (pre-filled from `--csv` when provided)
- Multi-select filter for question types
- Toggle to show the full raw DataFrame

**Key aggregates** (always visible at the top): number of questions, average input/output tokens, average generation latency, average number of LLM calls.

---

## 🛠️ Scripts & CLI arguments

### 📄 `export_pdf_to_csv.py`

Extracts chapters from a PDF, splits the text with LangChain’s `RecursiveCharacterTextSplitter`, and writes a CSV with columns `source`, `chunks`, `chapter`.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_filename` | ✅ | — | Path to the input PDF file. |
| `-o`, `--output` | ✅ | — | Output CSV path **or** a directory. If a directory is given, the file is named `<pdf_stem>_chunks.csv`. |
| `--chunk-size` | | `500` | Maximum characters per chunk (must be ≥ 50). |
| `--chunk-overlap` | | `0` | Character overlap between consecutive chunks (must be ≥ 0 and < `chunk-size`). |

**Notes**
- Chapter headings are detected with several regex patterns (`CHAPTER 1`, `1. Title`, Roman numerals, etc.).
- If no chapters are found, the whole document is treated as a single section named `Full Document`.

---

### 🗄️ `export_faiss_to_csv.py`

Exports every chunk stored in a FAISS knowledge-base directory (produced by `build_kb.py`) into a CSV.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | ✅ | — | Value written into the `source` column of every row (e.g. `faiss`). |
| `--kb-dir` | ✅ | — | Directory that contains `chunks.jsonl` and `index.faiss`. |
| `-o`, `--output` | | `export.csv` | Path of the CSV file to write. Parent directories are created automatically. |

**Notes**
- The script verifies that the number of vectors in the FAISS index matches the number of records in `chunks.jsonl` and prints a warning if they differ.
- Output columns: `source`, `chunks`, `chapter` (the latter comes from the `dataset` field of each chunk).

---

### 🤖 `generate_qa_set.py`

Performs stratified sampling for **eight** failure-mode types and (optionally) calls an LLM to produce question–answer pairs.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Path to the input CSV (must contain columns `source`, `chunks`, `chapter`). |
| `-o`, `--output-dir` | ✅ | — | Directory where the resulting QA CSV will be written. |
| `-n`, `--num-entries` | | `10` | Number of samples to draw **for each** failure-mode type. |
| `--llm-config` | | `None` | Path to a YAML config file that contains model name and API key. If omitted, the `LLM_MODEL` environment variable is used. |
| `--llm-entry` | | `claude_5_5` | Key inside the YAML config that selects the model / API key. |
| `--model` | | `None` | Override the model name (LiteLLM format). Takes precedence over the config file. |
| `--max-tokens` | | `1024` | Maximum tokens for each LLM completion. |
| `--skip-generation` | | `False` | Only perform sampling + merge; do **not** call the LLM. |
| `--seed` | | `None` | Random seed for reproducible sampling. |

#### Failure-mode types

| Type | Description |
|------|-------------|
| `single-hop` | Question answerable from a single chunk (stratified by chapter) |
| `multi-hop` | Question that requires combining two passages |
| `negative_unsupported_questions` | Question whose answer is **not** in the context (rejection / “I don’t know” expected) |
| `linguistic_complexity` | Question based on hard-to-read / low Flesch-score passages |
| `document_maximization` | Question that needs evidence from many chunks (≈ 5) |
| `cross_section_synthesis` | Question that joins facts from different chapters / sections |
| `ambiguous` | Intentionally underspecified / vague question |
| `terminological_variation` | Question that uses synonyms / paraphrases instead of the source wording |

#### Useful flags (quick reference)

| Flag | Description |
|------|-------------|
| `--skip-generation` | Only sample + merge; do not call the LLM |
| `--model MODEL` | Override the model name (litellm format) |
| `--seed N` | Make sampling reproducible |
| `--max-tokens N` | Max tokens per completion (default 1024) |

---

### 📊 `evaluate_rag_metrics.py` (unified evaluator)

Computes **all** metrics in one pass and writes a single enriched CSV.

**Per-row metrics (LLM / Ragas collections, when an LLM is configured)**

| Metric column | Description |
|---------------|-------------|
| `context_precision` | Context Precision (too many irrelevant chunks) |
| `context_recall` | Context Recall (missing relevant chunks) |
| `context_entities_recall` | Context Entity Recall (missing key entities) |
| `noise_sensitivity` | Noise Sensitivity (impact of irrelevant context) |
| `response_relevancy` | Answer / Response Relevancy (needs embeddings) |
| `faithfulness` | Faithfulness (hallucination / unsupported claims) |
| `answer_accuracy` | Answer Accuracy |
| `context_relevance` | Context Relevance |
| `response_groundedness` | Response Groundedness |
| `factual_correctness` | Factual Correctness |
| `nonllm_string_similarity` | Non-LLM string similarity (Levenshtein) |
| `ragas_bleu` / `ragas_chrf` / `ragas_rouge` | Ragas collection BLEU / CHRF / ROUGE |

**Per-row metrics (always, pure lexical)**

| Metric column | Description |
|---------------|-------------|
| `metric_bleu` | BLEU |
| `metric_chrf` | CHRF |
| `metric_rouge_l` | ROUGE-L |

**Corpus-level metrics (when `ranx` is available)**

| Metric column | Description |
|---------------|-------------|
| `corpus_mrr` | Mean Reciprocal Rank |
| `corpus_map` | Mean Average Precision |
| `corpus_ndcg@1/3/5/10` | Normalized Discounted Cumulative Gain at k |
| `corpus_map@1/3/5/10` | MAP at k |

For every metric the script also records:

- `<metric>_latency_ms`
- `<metric>_input_tokens`
- `<metric>_output_tokens`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Path to the input CSV to evaluate. |
| `-o`, `--output` | | `<input>_all_metrics.csv` | Path for the output CSV. |
| `--llm-config` | | `None` | Path to `llm_config.yaml` (required for LLM-based Ragas metrics). |
| `--llm-entry` | | `claude_5_5` | Key inside the YAML config that selects the model / API key. |
| `--embedding-model` | | `text-embedding-3-small` | OpenAI embedding model used by Response Relevancy. |
| `--limit` | | `None` | Only evaluate the first N rows (useful for dry-runs). |
| `--no-llm` | | `False` | Skip all LLM-based Ragas metrics. |
| `--relevance-threshold` | | `0.25` | Token-overlap threshold used to build qrels for ranking metrics. |
| `--no-ranking` | | `False` | Skip ranking metrics (MRR / NDCG / MAP). |

**Notes**
- Required columns: `question`, `generation_chunks`, `generation_answer`, `answer`.
- `generation_chunks` is expected to be a stringified list of dicts containing at least a `text` field (and optionally `score` / `chunk_id`).
- Optional column `chunks` (gold context) improves the quality of the relevance judgments used for ranking metrics.
- Ranking metrics are written as constant columns (`corpus_mrr`, `corpus_ndcg@5`, …) so every row carries the corpus-level score.
- A JSON summary (`*.summary.json`) with means and ranking aggregates is written next to the CSV.
- Graceful degradation: missing optional packages (`ranx`, `ragas`, `nltk`, `sentence-transformers`, `openai`) simply disable the corresponding metric groups.

---

### 📊 `metric_visualizer.py` (Streamlit dashboard)

Interactive multi-tab dashboard for exploring the unified evaluation CSV.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv`, `-c` | | `None` | Path to the evaluation CSV. Can also be typed in the sidebar. |

**Launch examples**

```bash
streamlit run metric_visualizer.py -- --csv path/to/eval.csv
python metric_visualizer.py --csv path/to/eval.csv   # opens Streamlit if configured
```

**Notes**
- Expects a **single** CSV produced by `evaluate_rag_metrics.py`.
- Sidebar lets you change the CSV path, filter by failure-mode type, and toggle full raw tables.
- Uses Plotly for interactive charts (hover, zoom, export). Every chart has a unique `key` to avoid Streamlit ID collisions.
- Health scores are the unweighted mean of the metrics present in each category (`noise_sensitivity` is inverted).
- See [§6 Visualize metrics](#6-visualize-metrics-metric_visualizerpy) for a full feature overview.

---

## 📊 CSV schemas

### Input CSV schema (required by `generate_qa_set.py`)

| Column    | Type | Description |
|-----------|------|-------------|
| `source`  | str  | Source of the document |
| `chunks`  | str  | Text chunk from the preprocessed document |
| `chapter` | str  | Sub-unit / chapter identifier |

### Output CSV schema (produced by `generate_qa_set.py`)

| Column               | Description |
|----------------------|-------------|
| `type`               | Failure-mode type (see table above) |
| `source`             | Source document name |
| `chapter`            | Chapter(s) the chunk(s) belong to |
| `question`           | Generated question |
| `chunks`             | Original context (string or list of strings) |
| `answer`             | Expected answer |
| `model`              | Model that produced the Q/A |
| `input_token_count`  | Prompt tokens |
| `output_token_count` | Completion tokens |
| `elapsed_time`       | Seconds taken by the LLM call |

### Evaluation input schema (`evaluate_rag_metrics.py`)

| Column | Description |
|--------|-------------|
| `question` | User question |
| `answer` | Gold / reference answer |
| `generation_answer` | Answer produced by the RAG agent |
| `generation_chunks` | Retrieved chunks (stringified list of dicts with `text`, optionally `score` / `chunk_id`) |
| `chunks` | *(optional)* Gold context text used for ranking relevance judgments |

### Evaluation output schema (produced by `evaluate_rag_metrics.py`)

All input columns are preserved. New columns include:

- Lexical: `metric_bleu`, `metric_chrf`, `metric_rouge_l` (+ latency / token companions)
- Ranking: `corpus_mrr`, `corpus_map`, `corpus_ndcg@k`, `corpus_map@k` (+ latency / token companions)
- LLM / Ragas (when enabled): `context_precision`, `context_recall`, `context_entities_recall`, `noise_sensitivity`, `response_relevancy`, `faithfulness`, `answer_accuracy`, `context_relevance`, `response_groundedness`, `factual_correctness`, `nonllm_string_similarity`, `ragas_bleu`, `ragas_chrf`, `ragas_rouge` (+ latency / token companions)

---

## 📝 Notes

- For multi-chunk types (`multi-hop`, `document_maximization`, `cross_section_synthesis`) the sampled chunks are joined with `|` before being sent to the LLM.
- The negative prompt explicitly instructs the model to produce questions that **cannot** be answered from the given context (the expected answer should be a rejection / “insufficient information”).
- API keys are read from the YAML config (or from the usual environment variables that LiteLLM / OpenAI clients respect).
- Sampling strategies:
  - **single-hop** – stratified by chapter
  - **multi-hop** – pairs of chunks
  - **negative_unsupported_questions** – questions whose answers are deliberately absent
  - **linguistic_complexity** – focuses on harder-to-read passages
  - **document_maximization** – larger groups of chunks (≈ 5)
  - **cross_section_synthesis** – prefers chunks from different chapters
  - **ambiguous** – underspecified questions
  - **terminological_variation** – synonym / paraphrase questions
- Ranking relevance in the evaluator is a heuristic (token overlap ≥ threshold with gold context or gold answer). Adjust `--relevance-threshold` if the default is too strict or too lenient for your data.
