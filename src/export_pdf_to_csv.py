#!/usr/bin/env python3
"""
PDF Structure Discoverer

Usage:
    python pdf_to_csv.py book.pdf -d  -o output_folder

When the -d/--discover flag is passed, the script analyses the PDF and
writes a CSV named <pdf_stem>_structure.csv containing:

    title,page_start,page_end
"""

import re
import csv
import argparse
from pathlib import Path

try:
    import pymupdf  # PyMuPDF
except ImportError:
    raise SystemExit("PyMuPDF is required. Install it with:  pip install pymupdf")

def safe_filename(title: str) -> str:
    """
    Convert a section title into a safe filename:
    - lowercased
    - spaces → _
    - remove/replace characters that are illegal in filenames
    """
    name = title.lower().strip()
    name = name.replace(" ", "_")
    # Remove characters that are problematic in filenames
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def extract_sections_from_structure(
    structure_csv: Path,
    pdf_path: Path,
    output_dir: Path,
) -> int:
    """
    Read a *_structure.csv file and extract the corresponding page ranges
    from the PDF into individual text files.

    Parameters
    ----------
    structure_csv : Path
        Path to the CSV produced by the discover step
        (columns: title, page_start, page_end)
    pdf_path : Path
        Path to the original PDF
    output_dir : Path
        Folder where the extracted .txt files will be written
    """
    if not structure_csv.exists():
        raise FileNotFoundError(f"Structure CSV not found: {structure_csv}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the structure
    with structure_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sections = list(reader)

    if not sections:
        print("No sections found in the CSV.")
        return

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    exported_files_count = 0

    print(f"Extracting {len(sections)} sections from {pdf_path.name} …")

    for sec in sections:
        title = sec["title"].strip()
        try:
            page_start = int(sec["page_start"])
            page_end = int(sec["page_end"])
        except (KeyError, ValueError) as e:
            print(f"  Skipping invalid row: {sec} ({e})")
            continue

        # Clamp to valid page range (1-based → 0-based)
        start_idx = max(0, page_start - 1)
        end_idx = min(total_pages - 1, page_end - 1)

        if start_idx > end_idx:
            print(f"  Skipping '{title}': invalid page range {page_start}-{page_end}")
            continue

        # Extract text from the page range
        texts = []
        for page_idx in range(start_idx, end_idx + 1):
            texts.append(doc[page_idx].get_text("text"))

        full_text = "\n".join(texts).strip()

        # Build safe filename
        filename = safe_filename(title) + ".txt"
        out_path = output_dir / filename

        out_path.write_text(full_text, encoding="utf-8")
        print(f"  Saved: {out_path.name}  (pages {page_start}-{page_end})")
        exported_files_count += 1

    doc.close()
    return exported_files_count

# ----------------------------------------------------------------------
# Detection patterns
# ----------------------------------------------------------------------
CHAPTER_PATTERNS = [
    # CHAPTER 1  /  Chapter 1: Title  /  CHAPTER I
    r'^(?:CHAPTER|Chapter|CHAPT\.?)\s+([IVXLCDM]+|\d+)[\.\:\-\s]*(.*)$',
    # 1. Title  or  1 Title
    r'^(\d+)[\.\:\-\s]+(.+)$',
    # I. Title
    r'^([IVXLCDM]+)[\.\:\-\s]+(.+)$',
]

'''
TOC_PATTERN = re.compile(
    r'^(?:brief\s+)?(?:table\s+of\s+)?contents?$',
    re.IGNORECASE
)
'''

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

# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
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

    sections = []          # temporary list with only page_start
    toc_found = False
    seen_titles = set()    # avoid duplicates

    for page_idx in range(total_pages):
        page_num = page_idx + 1          # 1-based
        text = doc[page_idx].get_text("text")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines:
            # ---- Table of Contents ----
            '''if not toc_found and TOC_PATTERN.match(line):
                title = "Table of Contents"
                if title not in seen_titles:
                    sections.append({
                        "title": title,
                        "page_start": page_num
                    })
                    seen_titles.add(title)
                    toc_found = True
                continue
            '''

            # ---- Chapters ----
            detected = detect_chapter(line)
            if detected:
                num, raw_title = detected
                if raw_title:
                    title = f"Chapter {num}: {raw_title}"
                else:
                    title = f"Chapter {num}"

                # Skip if we already recorded this exact title
                if title in seen_titles:
                    continue

                # Heuristic: ignore very late occurrences that are probably
                # just mentions inside the text (optional safety)
                sections.append({
                    "title": title,
                    "page_start": page_num
                })
                seen_titles.add(title)

    doc.close()

    # ---- Compute page_end ----
    for i, sec in enumerate(sections):
        if i + 1 < len(sections):
            sec["page_end"] = sections[i + 1]["page_start"] - 1
        else:
            sec["page_end"] = total_pages

        # Safety: never let page_end < page_start
        if sec["page_end"] < sec["page_start"]:
            sec["page_end"] = sec["page_start"]

    return sections

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Discover the structure (ToC + chapters) of a PDF"
    )
    parser.add_argument(
        "input_filename",
        help="Input PDF file"
    )
    parser.add_argument(
        "-d", "--discover",
        action="store_true",
        help="Discovery mode reads the PDF and generates a CSV of its structure"
    )
    parser.add_argument(
        "-e", "--extract",
        action="store_true",
        help="Extracts and converts the page from the input PDF based on what is specified in the structure CSV"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Folder where the structure CSV will be written"
    )
    args = parser.parse_args()

    input_path = Path(args.input_filename)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise SystemExit("Input must be a PDF file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.discover and args.extract:
        parser.error("Pass only one mode: either discover or extract.")

    if not args.discover and not args.extract:
        parser.error("Pass at least one mode: either discover or extract")

    # Discover
    if args.discover:
        print(f"Analysing structure of {input_path.name} …")
        sections = discover_structure(input_path)

        if not sections:
            print("No sections (ToC or chapters) were detected.")
            return

        csv_path = output_dir / f"{input_path.stem}_structure.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "page_start", "page_end"])
            writer.writeheader()
            for sec in sections:
                writer.writerow(sec)

        print(f"Found {len(sections)} sections")
        print(f"Saved → {csv_path}")
    
    # Extract
    if args.extract:
        csv_path = output_dir / f"{input_path.stem}_structure.csv"

        sections_exported = extract_sections_from_structure(
            structure_csv=csv_path,
            pdf_path=Path(args.input_filename),
            output_dir=Path(args.output_dir),
        )
        
        print(f"Number of sections exported: {sections_exported}")

if __name__ == "__main__":
    main()