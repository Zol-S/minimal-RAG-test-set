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

---

## 🎯 What it does

1. **Extract & chunk** text from different sources and save it as a standardized CSV.
2. **Sample** the chunks according to several failure-mode strategies (basic, multi-hop, negative/unsupported, complexity).
3. **Generate** question–answer pairs with an LLM (optional) that can later be used to test RAG systems.

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
| `generation_chunks` | All the chunks returned by the retriever |
| `generation_llm_calls` | Number of LLM calls |
| `generation_input_tokens` | Total number of input tokens |
| `generation_output_tokens` | Total number of output tokens |
| `generation_latency_s` | Generation latency (s) |

### 5. Evaluate the output
The script calculates the following metrics for the input CSV

| Category | Metrics | Library / method |
| --- | --- | --- |
| Retrieval ranking | MRR, MAP, NDCG@1/3/5/10, MAP@k | Ranx (relevance via token-overlap heuristic against gold chunks / answer) |
| Lexical generation | BLEU, CHRF, ROUGE-L | Built-in implementations (always available) |
| RAGAS | Context Precision, Context Recall, Context Entity Recall, Context Relevance, Faithfulness, Answer/Response Relevancy, Answer Accuracy, Factual Correctness, Response Groundedness, Noise Sensitivity, Semantic Similarity, BLEU/CHRF/ROUGE (RAGAS versions) | RAGAS (LLM-based ones need an API key) |

```bash
python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5

# Override model, still use key from config
python evaluate_rag_metrics.py \
  -i questions_with_answers.csv \
  --llm-config llm.yaml \
  --llm-entry claude_5_5 \
  --model anthropic/claude-3-5-sonnet

# Quick test on first N rows
python evaluate_rag_metrics.py -i input.csv -o out.csv --limit 5 --no-llm
```

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

---

## 📝 Notes

- For the `information_integration` type the two sampled chunks are joined with `|` before being sent to the LLM.
- The negative-prompt explicitly instructs the model to produce questions that **cannot** be answered from the given context (the expected answer should be a rejection / “insufficient information”).
- API keys are read from the YAML config (or from the usual environment variables that LiteLLM respects).
- Sampling strategies:
  - **basic** – stratified by chapter
  - **information_integration** – pairs of chunks (multi-hop)
  - **negative_unsupported_questions** – questions whose answers are deliberately absent
  - **complexity** – focuses on harder-to-read passages
