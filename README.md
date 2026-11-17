# RAG Failure-Mode Question Generator (CLI)

## What it does
1. **Loads** a CSV that must contain the columns `source`, `chunks` and `chapter`.
2. **Samples** content according to four strategies:
   - **basic** – stratified by chapter
   - **information_integration** – pairs of chunks (multi-hop)
   - **negative_unsupported_questions** – questions whose answer is *not* in the context
   - **complexity** – questions derived from (hard-to-read) passages
3. **Merges** all sampled rows into a single DataFrame.
4. **Generates** a question + expected answer for every row via an LLM (LiteLLM + Claude / any supported model).
5. **Saves** the resulting QA set as a CSV in the folder you specify.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Prepare a config (optional – you can also set LLM_MODEL env var)
cp llm_config.example.yaml llm_config.yaml
# edit llm.yaml and put your API key

# 2. Run
python generate_qa_set.py \
  --input path/to/chunks.csv \
  --output-dir ./output \
  --num-entries 10 \
  --llm-config llm_config.yaml \
  --llm-entry claude_5_5
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--skip-generation` | Only sample + merge; do not call the LLM |
| `--model MODEL` | Override the model name (litellm format) |
| `--seed N` | Make sampling reproducible |
| `--max-tokens N` | Max tokens per completion (default 1024) |

## Input CSV schema

| Column   | Type | Description                          |
|----------|------|--------------------------------------|
| `source` | str  | Source of the document |
| `chunks` | str  | Text chunk from the preprocessed document |
| `chapter`| str  | Sub-unit / chapter identifier        |

## Output CSV schema

| Column              | Description |
|---------------------|-------------|
| `type`              | Failure-mode type (`basic`, `information_integration`, …) |
| `source`            | Source document name you supplied |
| `chapter`           | Chapter(s) the chunk(s) belong to |
| `question`          | Generated question |
| `chunks`            | Original context (string or list of strings) |
| `answer`            | Expected answer |
| `model`             | Model that produced the Q/A |
| `input_token_count` | Prompt tokens |
| `output_token_count`| Completion tokens |
| `elapsed_time`      | Seconds taken by the LLM call |

## Notes

- For the `information_integration` type the two sampled chunks are joined with `|` before being sent to the LLM.
- The negative-prompt explicitly instructs the model to produce questions that **cannot** be answered from the given context (the expected answer should be a rejection / “insufficient information”).
- API keys are read from the YAML config (or from the usual environment variables that LiteLLM respects).
