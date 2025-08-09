#!/usr/bin/env python3
"""
Lightweight PDF-to-text utilities for FinanceBench PDFs.

Provided functions:
- pdf_to_text(pdf_path): returns the extracted text as a UTF-8 string
- parse_stock_and_year(pdf_path): returns (stock_name, year|None) parsed from filename

Examples:
    text = pdf_to_text("datasets/financebench/pdfs/AMD_2017_10K.pdf")
    stock, year = parse_stock_and_year("datasets/financebench/pdfs/AMD_2017_10K.pdf")
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    print(
        "PyMuPDF (package 'PyMuPDF') is required. Please install it with: pip install PyMuPDF",
        file=sys.stderr,
    )
    raise


def parse_stock_and_year(pdf_path: str | Path) -> tuple[str, int | None]:
    """Parse the stock/company name and year from a FinanceBench PDF filename.

    Heuristics:
    - stock/company name is the leading token before the first underscore.
    - year is the first 4-digit number in the filename (e.g., 2015, 2023).
    Returns (stock_name, year_or_None).
    """
    stem = Path(pdf_path).stem
    # Stock/company name: up to first underscore, fallback to whole stem
    stock = stem.split("_", 1)[0] if "_" in stem else stem

    # First 4-digit year occurrence, prioritizing 20xx then 19xx
    m = re.search(r"(20\d{2}|19\d{2})", stem)
    year: int | None = int(m.group(1)) if m else None

    return stock, year


def pdf_to_text(pdf_path: str | Path) -> str:
    """Extract text from a single PDF file using PyMuPDF.

    Adds page separators between pages to preserve structure during later analysis.
    Raises ValueError for encrypted PDFs that require a password.
    """
    pdf_path = Path(pdf_path)
    text_parts: list[str] = []

    with fitz.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError(f"Encrypted PDF requires a password: {pdf_path.name}")

        total_pages = len(doc)
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            header = f"\n\n===== Page {i}/{total_pages} : {pdf_path.name} =====\n\n"
            text_parts.append(header)
            text_parts.append(page_text)

    return "".join(text_parts)
