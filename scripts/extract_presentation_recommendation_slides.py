#!/usr/bin/env python
"""Export recommendation slides from the final presentation PDF as PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import fitz  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "PyMuPDF is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


DEFAULT_PAGES_1_BASED = [21, 22, 23, 24, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        default="reports/final-presentation.pdf",
        help="Input presentation PDF",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/images/presentation",
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Render zoom factor (2.0 gives high-quality slide images)",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        type=int,
        default=DEFAULT_PAGES_1_BASED,
        help="1-based page numbers to export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Suppress non-blocking PDF structure-tree warnings from this deck.
    fitz.TOOLS.mupdf_display_errors(False)

    doc = fitz.open(pdf_path)

    for idx, page_num in enumerate(args.pages, start=1):
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Invalid page {page_num}; PDF has {len(doc)} pages")

        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom), alpha=False)
        out_path = out_dir / f"recommendation_slide_{idx:02d}_page_{page_num}.png"
        pix.save(out_path)

        preview = (page.get_text() or "").replace("\n", " ")[:120]
        print(f"Exported {out_path} | {preview}")

    doc.close()


if __name__ == "__main__":
    main()
