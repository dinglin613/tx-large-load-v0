from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from pypdf import PdfReader


def iter_page_hits(pdf_path: Path, pattern: re.Pattern[str]) -> Iterable[Tuple[int, List[str]]]:
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text:
            continue
        lines = text.splitlines()
        hits = [ln.strip() for ln in lines if pattern.search(ln)]
        if hits:
            # 1-index page numbers (more human-friendly)
            yield i + 1, hits[:10]


def main() -> int:
    ap = argparse.ArgumentParser(description="Search text inside a PDF and report page hits")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--re", dest="regex", required=True, help="Regex pattern (case-insensitive by default)")
    ap.add_argument("--case-sensitive", action="store_true", help="Enable case-sensitive search")
    args = ap.parse_args()

    # Best-effort UTF-8 output on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py>=3.7
    except Exception:
        pass

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    flags = 0 if args.case_sensitive else re.IGNORECASE
    pat = re.compile(args.regex, flags=flags)

    any_hit = False
    for page_num, lines in iter_page_hits(pdf_path, pat):
        any_hit = True
        print(f"\n--- page {page_num} ---")
        for ln in lines:
            # Avoid hard crashes on terminals with legacy encodings.
            try:
                print(ln)
            except UnicodeEncodeError:
                print(ln.encode("utf-8", errors="backslashreplace").decode("utf-8"))

    if not any_hit:
        print("No matches.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
