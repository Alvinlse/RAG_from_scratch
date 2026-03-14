"""
PDF parser using PyMuPDF (fitz) with section detection.

Extracts text grouped by sections (detected via font-size headings)
so downstream RAG chunks carry semantic context rather than raw pages.

No external server required — works fully offline.
"""

import os
import re
import unicodedata
from pathlib import Path
import torch
import fitz  # PyMuPDF
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


# ── Text helpers ──────────────────────────────────────────────────────────────

def text_formatter(text: str) -> str:
    # NFKC decomposes ligatures (ﬁ→fi, ﬂ→fl, ﬀ→ff, etc.) and normalises unicode
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\n", " ").strip()
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned


def _chunk_stats(text: str, section: str, chunk_index: int) -> dict:
    return {
        "section": section,
        "chunk_index": chunk_index,
        "char_count": len(text),
        "word_count": len(text.split()),
        "token_count": len(text) / 4,
        "text": text,
    }


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, min_heading_size: float | None = None) -> list[dict]:
    """
    Parse a PDF into section-level chunks using PyMuPDF.

    Strategy:
      1. Collect every text span with its font size.
      2. Treat spans whose font size is noticeably larger than the median body
         font as section headings.
      3. Accumulate body text under each heading; yield one chunk per section.

    Args:
        pdf_path:         Path to the PDF file.
        min_heading_size: Override the auto-detected heading size threshold.

    Returns:
        List of dicts with keys: section, chunk_index, char_count,
        word_count, token_count, text.
    """
    doc = fitz.open(pdf_path)

    # ── Pass 1: collect all spans to find body font size ─────────────────────
    all_sizes: list[float] = []
    page_spans: list[list[dict]] = []  # per-page list of span dicts

    # Pattern that matches figure captions: "Figure 1", "Fig. 2", "FIGURE 3", etc.
    _FIGURE_RE = re.compile(r"^(Figure|Fig\.?)\s*\d+", re.IGNORECASE)

    for page in doc:
        # Collect table bounding boxes — text inside tables is still skipped
        table_rects: list[fitz.Rect] = []
        try:
            for table in page.find_tables().tables:
                table_rects.append(fitz.Rect(table.bbox))
        except Exception:
            pass  # find_tables not available on this page/version

        def _in_table(bbox) -> bool:
            rect = fitz.Rect(bbox)
            return any(rect.intersects(t) for t in table_rects)

        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:      # skip raw image blocks
                continue
            if _in_table(block["bbox"]):
                continue                # skip table text

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    # Mark figure captions so Pass 2 can use them as headings
                    is_caption = bool(_FIGURE_RE.match(text))
                    spans.append({"text": text, "size": span["size"], "is_caption": is_caption})
                    if not is_caption:
                        all_sizes.append(span["size"])
        page_spans.append(spans)

    if not all_sizes:
        return []

    # Median font size ≈ body text
    all_sizes.sort()
    median_size = all_sizes[len(all_sizes) // 2]
    heading_threshold = min_heading_size or (median_size * 1.15)

    # ── Pass 2: group spans into sections ────────────────────────────────────
    chunks: list[dict] = []
    current_section = "preamble"
    current_text_parts: list[str] = []
    chunk_index = 0

    def flush():
        nonlocal chunk_index, current_text_parts
        raw = " ".join(current_text_parts)
        text = text_formatter(raw)
        if text:
            chunks.append(_chunk_stats(text, current_section, chunk_index))
            chunk_index += 1
        current_text_parts = []

    for spans in page_spans:
        for span in spans:
            size = span["size"]
            text = span["text"]

            if span["is_caption"]:
                # Figure caption → new section named after the caption
                flush()
                current_section = text_formatter(text)
            elif size >= heading_threshold and len(text) < 120:
                # Larger font → regular section heading
                flush()
                current_section = text_formatter(text)
            else:
                current_text_parts.append(text)

    flush()  # last section
    doc.close()
    return chunks


# ── Batch helper ──────────────────────────────────────────────────────────────

def parse_all_pdfs(pdf_paths: list[str]) -> dict[str, list[dict]]:
    """Parse multiple PDFs; returns {path: [chunks]}."""
    results = {}
    for path in tqdm(pdf_paths, desc="Parsing PDFs"):
        print(f"  → {os.path.basename(path)}")
        chunks = parse_pdf(path)
        results[path] = chunks
        print(f"     {len(chunks)} chunks extracted")
    return results


# ── Default paths used by the rest of the pipeline ───────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
DEFAULT_PDF = os.path.join(DOCS_DIR, "Learning to Optimize Tensor Programs.pdf")


def get_chunks(pdf_path: str = DEFAULT_PDF) -> list[dict]:
    """Return parsed chunks for a single PDF (drop-in for the pipeline)."""
    return parse_pdf(pdf_path)


pdf_path = r'C:\Users\layki\Downloads\RAG_from_scratch\docs\Learning to Optimize Tensor Programs.pdf'
chunk_info = parse_pdf(pdf_path)

