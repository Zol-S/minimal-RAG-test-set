#!/usr/bin/env python3
"""
PDF to Chunked CSV/TSV Exporter

Accepts a PDF, detects chapters, extracts text per chapter, splits the text
into chunks using langchain's RecursiveCharacterTextSplitter, and writes a
tab-separated file (with timestamped filename via Log.save_file) with columns:

    source, chunks, chapter

Usage examples:
    python export_pdf_to_csv.py book.pdf -o output/
    python export_pdf_to_csv.py book.pdf -o results/chunks.csv --chunk-size 500 --chunk-overlap 0
"""

import re
import argparse
from pathlib import Path

try:
    import pymupdf  # PyMuPDF
except ImportError:
    raise SystemExit("PyMuPDF is required. Install it with:  pip install pymupdf")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise SystemExit(
        "langchain-text-splitters is required. Install it with:\n"
        "  pip install langchain-text-splitters"
    )

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required. Install it with:  pip install pandas")

from Helper import Log


# ----------------------------------------------------------------------
# Chapter detection
# ----------------------------------------------------------------------
CHAPTER_PATTERNS = [
    # CHAPTER 1  /  Chapter 1: Title  /  CHAPTER I
    r'^(?:CHAPTER|Chapter|CHAPT\.?)\s+([IVXLCDM]+|\d+)[\.\:\-\s]*(.*)$',
    # 1. Title  or  1 Title
    r'^(\d+)[\.\:\-\s]+(.+)$',
    # I. Title
    r'^([IVXLCDM]+)[\.\:\-\s]+(.+)$',
]


def detect_chapter(line: str):
    """Return (number, title) if the line looks like a chapter heading."""
    line = line.strip()
    if not line or len(line) > 120:
        return None
    for pat in CHAPTER_PATTERNS:
        m = re.match(pat, line, re.IGNORECASE)
        if m:
            num = m.group(1).strip()
            title = m.group(2).strip() if m.lastindex >= 2 else ""
            return num, title
    return None


def discover_structure(pdf_path: Path):
    """
    Analyse the PDF and return a list of sections:
    [
        {"title": "...", "page_start": int, "page_end": int},
        ...
    ]
    """
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    sections = []
    seen_titles = set()

    for page_idx in range(total_pages):
        page_num = page_idx + 1  # 1-based
        text = doc[page_idx].get_text("text")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines:
            detected = detect_chapter(line)
            if detected:
                num, raw_title = detected
                if raw_title:
                    title = f"Chapter {num}: {raw_title}"
                else:
                    title = f"Chapter {num}"

                if title in seen_titles:
                    continue

                sections.append({
                    "title": title,
                    "page_start": page_num
                })
                seen_titles.add(title)

    doc.close()

    # Compute page_end
    for i, sec in enumerate(sections):
        if i + 1 < len(sections):
            sec["page_end"] = sections[i + 1]["page_start"] - 1
        else:
            sec["page_end"] = total_pages

        if sec["page_end"] < sec["page_start"]:
            sec["page_end"] = sec["page_start"]

    return sections


# ----------------------------------------------------------------------
# Text extraction & chunking
# ----------------------------------------------------------------------
def extract_text_for_range(doc, page_start: int, page_end: int) -> str:
    """Extract plain text from a 1-based inclusive page range."""
    total_pages = len(doc)
    start_idx = max(0, page_start - 1)
    end_idx = min(total_pages - 1, page_end - 1)

    if start_idx > end_idx:
        return ""

    texts = []
    for page_idx in range(start_idx, end_idx + 1):
        texts.append(doc[page_idx].get_text("text"))
    return "\n".join(texts).strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 0) -> list[str]:
    """
    Split text using langchain's RecursiveCharacterTextSplitter
    with the requested separators and parameters.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=[".\n", ". ", "!", "?"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def process_pdf(
    pdf_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[dict]:
    """
    Detect chapters, extract text, chunk it, and return rows for CSV.
    Each row is a dict with keys: source, chunks, chapter
    """
    source_name = pdf_path.name
    sections = discover_structure(pdf_path)

    doc = pymupdf.open(pdf_path)
    rows = []

    if not sections:
        # Fallback: treat the whole document as a single chapter
        print("No chapter headings detected – treating entire PDF as one section.")
        full_text = extract_text_for_range(doc, 1, len(doc))
        for chunk in chunk_text(full_text, chunk_size, chunk_overlap):
            rows.append({
                "source": source_name,
                "chunks": chunk,
                "chapter": "Full Document",
            })
    else:
        print(f"Detected {len(sections)} chapter(s). Chunking…")
        for sec in sections:
            title = sec["title"]
            page_start = sec["page_start"]
            page_end = sec["page_end"]

            text = extract_text_for_range(doc, page_start, page_end)
            if not text:
                print(f"  Skipping empty chapter: {title}")
                continue

            chapter_chunks = chunk_text(text, chunk_size, chunk_overlap)
            print(f"  {title}: {len(chapter_chunks)} chunk(s)  (pages {page_start}-{page_end})")

            for chunk in chapter_chunks:
                rows.append({
                    "source": source_name,
                    "chunks": chunk,
                    "chapter": title,
                })

    doc.close()
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract chapters from a PDF, chunk the text with "
                    "RecursiveCharacterTextSplitter, and save as TSV "
                    "(columns: source, chunks, chapter) using Log.save_file"
    )
    parser.add_argument(
        "input_filename",
        help="Input PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (or directory). A timestamp will be added to the "
             "filename via Log.save_file; if a directory is given the base name "
             "will be <pdf_stem>_chunks.csv"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum characters per chunk (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Character overlap between consecutive chunks (default: 0)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_filename)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise SystemExit("Input must be a PDF file.")

    if args.chunk_size < 50:
        raise SystemExit("--chunk-size must be at least 50")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be >= 0 and < chunk-size")

    output_arg = Path(args.output)
    if output_arg.suffix.lower() == ".csv":
        # User gave an explicit CSV path
        csv_path = output_arg
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Treat as directory
        output_arg.mkdir(parents=True, exist_ok=True)
        csv_path = output_arg / f"{input_path.stem}_chunks.csv"

    print(f"Processing {input_path.name} …")
    rows = process_pdf(
        pdf_path=input_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not rows:
        print("No text could be extracted. Nothing written.")
        return

    pdf_data = pd.DataFrame(rows)

    with Log.save_file(csv_path) as filename:
        pdf_data.to_csv(filename, sep=",", index=False)

    print(f"\nWrote {len(rows)} chunk(s)")

if __name__ == "__main__":
    main()