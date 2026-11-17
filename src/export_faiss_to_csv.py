#!/usr/bin/env python3
"""
CLI app to export FAISS-backed knowledge base chunks to a CSV file.

Expects a KB directory produced by build_kb.py containing:
  - chunks.jsonl
  - index.faiss  (used to verify count)

CSV columns:
  - source   : value passed via --source
  - chunks   : the text of each chunk
  - chapter  : the dataset field of each chunk
  
Example invocation:
python -m export_faiss_to_csv
    --source faiss
    --kb-dir C:\Users\Zol\Documents\agent-harness-comparison\kb\44d5fde8e5
    --output ../data/knowledge_base_export/faiss_export.csv
"""

import argparse
import csv
import json
import os
import sys

import faiss


def load_chunks(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FAISS knowledge-base chunks to CSV"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Value written into the 'source' column of every row",
    )
    parser.add_argument(
        "--kb-dir",
        required=True,
        help="Directory that contains chunks.jsonl and index.faiss "
             "(the hash subdirectory produced by build_kb.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="export.csv",
        help="Path of the CSV file to write (default: export.csv)",
    )
    args = parser.parse_args()

    chunks_path = os.path.join(args.kb_dir, "chunks.jsonl")
    index_path = os.path.join(args.kb_dir, "index.faiss")

    if not os.path.isfile(chunks_path):
        print(f"Error: {chunks_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(index_path):
        print(f"Error: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    index = faiss.read_index(index_path)

    if index.ntotal != len(chunks):
        print(
            f"Warning: FAISS index has {index.ntotal} vectors but "
            f"chunks.jsonl has {len(chunks)} records. "
            "Proceeding with chunks.jsonl order.",
            file=sys.stderr,
        )

    # Ensure parent directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "chunks", "chapter"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for c in chunks:
            writer.writerow(
                {
                    "source": args.source,
                    "chunks": c["text"],
                    "chapter": c["dataset"],
                }
            )

    print(f"Exported {len(chunks)} rows to {args.output}")


if __name__ == "__main__":
    main()
