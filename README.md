# 🧪 RAG Failure-Mode Question Generator (CLI)

A small toolkit that turns documents (PDF or FAISS knowledge bases) into chunked CSV files and then generates targeted question–answer sets for evaluating RAG systems on common failure modes.

## 📄 Paper

The relevant paper is available here *(link to be added)*.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Main dependencies include `pymupdf`, `langchain-text-splitters`, `pandas`, `litellm`, `faiss-cpu` (or `faiss-gpu`), `pydantic`, `tqdm`, and `pyyaml`.

**Evaluation extras** (optional, depending on which metrics you need):

```bash
# LLM-based RAGAS evaluation
pip install ragas openai pyyaml tqdm

# No-LLM evaluation (ranking + lexical)
pip install ranx nltk
# optional: RAGAS non-LLM metrics + semantic similarity
pip install ragas datasets sentence-transformers
```

---

## 🎯 What it does

1. **Extract & chunk** text from different sources and save it as a standardized CSV.
2. **Sample** the chunks according to several failure-mode strategies (basic, multi-hop, negative/unsupported, complexity).
3. **Generate** question–answer pairs with an LLM (optional) that can later be used to test RAG systems.
4. **Evaluate** agent answers with either LLM-based RAGAS metrics or a fully offline (no-LLM) suite of ranking + lexical metrics.

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
py export_pdf_to_csv.py ../data/Laudon_MIS_17th.pdf --output ../data/export/laudon_export.csv --chunk-size 300 --chunk-overlap 0
```

**Example – FAISS to CSV export:**

```bash
py export_faiss_to_csv.py --source faiss --kb-dir ..\..\agent-harness-comparison\kb\44d5fde8e5 --output ../data/export/faiss_export.csv
```

### 3. Sample from exported chunks and generate questions

```bash
py generate_qa_set.py --input path/to/chunks.csv \
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

Two evaluation scripts are provided:

| Script | Requires LLM? | Main metrics |
|--------|---------------|--------------|
| `evaluate_rag_metrics.py` | Yes (OpenAI-compatible) | RAGAS collections: Context Precision/Recall, Faithfulness, Answer Relevancy, Answer Accuracy, Factual Correctness, … + BLEU/CHRF/ROUGE |
| `evaluate_rag_metrics_nollm.py` | No | Per-row BLEU / CHRF / ROUGE-L, Ranx ranking (MRR, MAP, NDCG@k), optional non-LLM RAGAS scores |

**Required columns for both scripts:** `question`, `answer`, `generation_answer`, `generation_chunks`.  
(`chunks` is optional but used by the no-LLM script for relevance judgments.)

#### LLM-based evaluation (`evaluate_rag_metrics.py`)

```bash
python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5

# Custom output path + limit to first 5 rows
python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  -o scored.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5 \
  --limit 5

# Use a different embedding model for Answer Relevancy
python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --llm-config llm.yaml \
  --embedding-model text-embedding-3-large
```

Scores are written as new columns; default output is `<input>_scored.csv`.

#### No-LLM evaluation (`evaluate_rag_metrics_nollm.py`)

```bash
python evaluate_rag_metrics_nollm.py -i questions_with_answers.csv

# Limit + custom output
python evaluate_rag_metrics_nollm.py -i input.csv -o metrics.csv --limit 20

# Skip ranking metrics
python evaluate_rag_metrics_nollm.py -i input.csv --no-ranking

# Adjust relevance threshold used to build qrels
python evaluate_rag_metrics_nollm.py -i input.csv --relevance-threshold 0.3

# Skip RAGAS batch metrics entirely
python evaluate_rag_metrics_nollm.py -i input.csv --no-ragas
```

Default output is `<input>_nollm.csv` plus a companion `<input>_nollm.summary.json`.

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

Performs stratified sampling for four failure-mode types and (optionally) calls an LLM to produce question–answer pairs.

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

#### Useful flags (quick reference)

| Flag | Description |
|------|-------------|
| `--skip-generation` | Only sample + merge; do not call the LLM |
| `--model MODEL` | Override the model name (litellm format) |
| `--seed N` | Make sampling reproducible |
| `--max-tokens N` | Max tokens per completion (default 1024) |

---

### 📊 `evaluate_rag_metrics.py` (LLM-based RAGAS)

Evaluates agent outputs with the Ragas **collections** API. Metrics are computed **per row** and attached as new columns.

**Metrics computed**

| Metric column | Description |
|---------------|-------------|
| `context_precision` | Context Precision |
| `context_recall` | Context Recall |
| `context_entities_recall` | Context Entity Recall |
| `noise_sensitivity` | Noise Sensitivity |
| `response_relevancy` | Answer / Response Relevancy (needs embeddings) |
| `faithfulness` | Faithfulness |
| `answer_accuracy` | Answer Accuracy |
| `context_relevance` | Context Relevance |
| `response_groundedness` | Response Groundedness |
| `factual_correctness` | Factual Correctness |
| `nonllm_string_similarity` | Non-LLM string similarity (Levenshtein) |
| `bleu` | BLEU |
| `chrf` | CHRF |
| `rouge` | ROUGE |

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Path to the input CSV to evaluate. |
| `-o`, `--output` | | `<input>_scored.csv` | Path for the output CSV. |
| `--llm-config` | | `None` | Path to `llm_config.yaml` (provider + API key + model). |
| `--llm-entry` | | `claude_5_5` | Key inside the YAML config that selects the model / API key. |
| `--embedding-model` | | `text-embedding-3-small` | OpenAI embedding model used by Response Relevancy. |
| `--limit` | | `None` | Only evaluate the first N rows (useful for dry-runs). |

**Notes**
- Required columns: `question`, `generation_chunks`, `generation_answer`, `answer`.
- `generation_chunks` is expected to be a stringified list of dicts containing at least a `text` field (and optionally `score`).
- The script creates an OpenAI-compatible client via `ragas.llms.llm_factory` / `embedding_factory`. API keys are taken from the YAML config (or from the usual environment variables).
- A mean summary of every metric is printed at the end.

---

### 📉 `evaluate_rag_metrics_nollm.py` (no-LLM)

Fully offline evaluation suite. Computes:

1. **Per-row lexical metrics** – BLEU, CHRF, ROUGE-L (always available; NLTK used when present).
2. **Ranking metrics** (via Ranx) – MRR, MAP, NDCG@1/3/5/10, MAP@k. Relevance judgments are built with a simple token-overlap heuristic against the gold `chunks` / `answer`.
3. **Optional non-LLM RAGAS metrics** – BleuScore, CHRFScore, RougeScore, and SemanticSimilarity (when `sentence-transformers` is installed).

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-i`, `--input` | ✅ | — | Input CSV (must contain `question`, `answer`, `generation_answer`, `generation_chunks`). |
| `-o`, `--output` | | `<input>_nollm.csv` | Output CSV path. |
| `--limit` | | `None` | Evaluate only the first N rows. |
| `--metrics` | | all available | Subset of RAGAS metrics to run (e.g. `bleu_score chrf_score rouge_score semantic_similarity`). |
| `--relevance-threshold` | | `0.25` | Token-overlap threshold used to build qrels for ranking metrics. |
| `--no-ranking` | | `False` | Skip ranking metrics (MRR / NDCG / MAP). |
| `--no-ragas` | | `False` | Skip RAGAS batch metrics. |

**Notes**
- Optional column `chunks` (gold context) improves the quality of the relevance judgments used for ranking metrics.
- Ranking metrics are written as constant columns (`corpus_mrr`, `corpus_ndcg@5`, …) so every row carries the corpus-level score.
- A JSON summary (`*.summary.json`) with means and ranking/RAGAS aggregates is written next to the CSV.
- Graceful degradation: missing optional packages (`ranx`, `ragas`, `nltk`, `sentence-transformers`) simply disable the corresponding metric groups.

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
| `type`               | Failure-mode type (`basic`, `information_integration`, `negative_unsupported_questions`, `complexity`) |
| `source`             | Source document name |
| `chapter`            | Chapter(s) the chunk(s) belong to |
| `question`           | Generated question |
| `chunks`             | Original context (string or list of strings) |
| `answer`             | Expected answer |
| `model`              | Model that produced the Q/A |
| `input_token_count`  | Prompt tokens |
| `output_token_count` | Completion tokens |
| `elapsed_time`       | Seconds taken by the LLM call |

### Evaluation input schema (both `evaluate_*.py` scripts)

| Column | Description |
|--------|-------------|
| `question` | User question |
| `answer` | Gold / reference answer |
| `generation_answer` | Answer produced by the RAG agent |
| `generation_chunks` | Retrieved chunks (stringified list of dicts with `text`, optionally `score` / `chunk_id`) |
| `chunks` | *(optional)* Gold context text used for ranking relevance judgments |

---

## 📝 Notes

- For the `information_integration` type the two sampled chunks are joined with `|` before being sent to the LLM.
- The negative-prompt explicitly instructs the model to produce questions that **cannot** be answered from the given context (the expected answer should be a rejection / “insufficient information”).
- API keys are read from the YAML config (or from the usual environment variables that LiteLLM / OpenAI clients respect).
- Sampling strategies:
  - **basic** – stratified by chapter
  - **information_integration** – pairs of chunks (multi-hop)
  - **negative_unsupported_questions** – questions whose answers are deliberately absent
  - **complexity** – focuses on harder-to-read passages
- Ranking relevance in the no-LLM evaluator is a heuristic (token overlap ≥ threshold with gold context or gold answer). Adjust `--relevance-threshold` if the default is too strict or too lenient for your data.
