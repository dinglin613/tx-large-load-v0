from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = REPO_ROOT / "rules" / "published.jsonl"
DEFAULT_REGISTRY = REPO_ROOT / "registry" / "doc_registry.json"


def _best_effort_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py>=3.7
    except Exception:
        pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield i, json.loads(s)


def load_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reg = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for d in reg or []:
        if isinstance(d, dict) and d.get("doc_id"):
            out[str(d["doc_id"])] = d
    return out


def pick_pdf_artifact(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    arts = [a for a in (doc.get("artifacts") or []) if isinstance(a, dict) and a.get("kind") == "pdf" and a.get("path")]
    if not arts:
        return None

    # Prefer canonical artifact (matches doc-level hash) when available.
    canonical_sha = str(doc.get("hash") or "").strip().lower()
    if canonical_sha:
        matched = [
            a
            for a in arts
            if a.get("sha256") and str(a.get("sha256") or "").strip().lower() == canonical_sha
        ]
    else:
        matched = []

    # Prefer the most recently retrieved artifact when multiple versions exist.
    def _key(a: Dict[str, Any]) -> tuple[str, str]:
        # ISO dates sort lexicographically
        return (str(a.get("retrieved_date") or ""), str(a.get("path") or ""))

    pool = matched or arts
    return sorted(pool, key=_key)[-1]


def normalize_text(s: str) -> str:
    s = s.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\s+", " ", s, flags=re.M).strip()
    return s.lower()


PAGE_RE = re.compile(
    r"\bpp?\.\s*(?P<a>\d+)\s*(?:[\u2013\-]\s*(?P<b>\d+))?",
    flags=re.IGNORECASE,
)

# Matches "Slide 5", "Slide 12-13", "Slide 12–13" (used in PPTX-backed rules)
SLIDE_RE = re.compile(
    r"\bslide\s+(?P<a>\d+)\s*(?:[\u2013\-]\s*(?P<b>\d+))?",
    flags=re.IGNORECASE,
)


def parse_page_range(loc: str) -> Optional[Tuple[int, int]]:
    # Try standard p./pp. first
    m = PAGE_RE.search(loc or "")
    if not m:
        # Fall back to Slide N / Slide N-M notation
        m = SLIDE_RE.search(loc or "")
    if not m:
        return None
    a = int(m.group("a"))
    b = int(m.group("b")) if m.group("b") else a
    if b < a:
        a, b = b, a
    return a, b


def candidate_anchors(loc: str) -> List[str]:
    out: List[str] = []

    # Quoted anchors are highest-signal when present.
    for pat in [
        r"\"([^\"]{12,})\"",
        r"\u201c([^\u201d]{12,})\u201d",
    ]:
        for m in re.finditer(pat, loc or ""):
            s = m.group(1).strip()
            if s:
                out.append(s)

    # Section/Sec patterns.
    for m in re.finditer(r"\b(?:section|sec)\s*([0-9]+(?:\.[0-9]+)*)", loc or "", flags=re.IGNORECASE):
        out.append(m.group(0).strip())
        out.append(m.group(1).strip())

    # Clause-like numeric patterns (e.g., 9.2.2, 3.6.1). Avoid short single numbers.
    for m in re.finditer(r"\b([0-9]+(?:\.[0-9]+){1,4})\b", loc or ""):
        out.append(m.group(1).strip())

    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        x2 = x.strip()
        if not x2:
            continue
        k = x2.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x2)
    return uniq


@dataclass
class Finding:
    rule_id: str
    doc_id: str
    loc: str
    status: str  # ok | warn | error
    page_start: Optional[int]
    page_end: Optional[int]
    pdf_path: Optional[str]
    pdf_sha256_expected: Optional[str]
    pdf_sha256_actual: Optional[str]
    anchor_used: Optional[str]
    anchor_found: bool
    anchor_page: Optional[int]
    message: str


def extract_page_text(reader: PdfReader, page_num_1_indexed: int) -> str:
    # PdfReader pages are 0-indexed
    idx = page_num_1_indexed - 1
    if idx < 0 or idx >= len(reader.pages):
        return ""
    t = reader.pages[idx].extract_text() or ""
    return normalize_text(t)


def find_anchor_in_pages(
    reader: PdfReader, page_start: int, page_end: int, anchors: Sequence[str]
) -> Tuple[bool, Optional[str], Optional[int]]:
    # Keep ranges small and predictable; most loc ranges are 1–2 pages.
    pages = list(range(page_start, page_end + 1))
    if len(pages) > 6:
        pages = pages[:3] + pages[-3:]

    for a in anchors:
        a_norm = normalize_text(a)
        if not a_norm:
            continue
        for p in pages:
            txt = extract_page_text(reader, p)
            if not txt:
                continue
            if a_norm in txt:
                return True, a, p
    return False, None, None


def audit_citations(
    *,
    rules_path: Path = DEFAULT_RULES,
    registry_path: Path = DEFAULT_REGISTRY,
    only_doc: str = "",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Audit published rule loc/doc_id citations against local PDF artifacts.

    Returns a JSON-serializable report dict matching CLI output.
    """
    registry = load_registry(registry_path)

    findings: List[Finding] = []
    error_count = 0
    warn_count = 0

    # Cache PdfReaders per doc_id
    readers: Dict[str, PdfReader] = {}

    def _get_reader(doc_id: str) -> Tuple[Optional[PdfReader], Optional[Path], Optional[str], Optional[str]]:
        doc = registry.get(doc_id)
        if not doc:
            return None, None, None, None
        art = pick_pdf_artifact(doc)
        if not art:
            return None, None, None, None
        rel = str(art.get("path"))
        expected = str(art.get("sha256") or "").lower() or None
        p = (REPO_ROOT / rel).resolve()
        if not p.exists():
            return None, p, expected, None
        actual = sha256_file(p).lower()
        if doc_id not in readers:
            readers[doc_id] = PdfReader(str(p))
        return readers[doc_id], p, expected, actual

    for line_no, rule in iter_jsonl(rules_path):
        if rule.get("review_status") != "published":
            continue

        rid = str(rule.get("rule_id") or "")
        doc_id = str(rule.get("doc_id") or "")
        loc = str(rule.get("loc") or "")

        if only_doc and doc_id != only_doc:
            continue

        where = f"{rules_path.name}:L{line_no}:{rid}"
        page_range = parse_page_range(loc)
        if not page_range:
            # Check whether the referenced doc is a non-paginated source (tracker,
            # form, web resource) — these legitimately have no page ranges.
            doc_entry = registry.get(doc_id)
            art_kinds = [
                a.get("kind")
                for a in ((doc_entry or {}).get("artifacts") or [])
                if isinstance(a, dict)
            ]
            has_pdf = any(k == "pdf" for k in art_kinds)
            if doc_entry and not has_pdf:
                # Non-paginated document — accept the citation as warn
                msg = (
                    f"{where}: non-paginated source ({', '.join(art_kinds) or 'no artifacts'}) — "
                    f"loc accepted without page-range verification"
                )
                findings.append(
                    Finding(
                        rule_id=rid,
                        doc_id=doc_id,
                        loc=loc,
                        status="warn",
                        page_start=None,
                        page_end=None,
                        pdf_path=None,
                        pdf_sha256_expected=None,
                        pdf_sha256_actual=None,
                        anchor_used=None,
                        anchor_found=False,
                        anchor_page=None,
                        message=msg,
                    )
                )
                warn_count += 1
                continue

            msg = f"{where}: unable to parse page range from loc"
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="error",
                    page_start=None,
                    page_end=None,
                    pdf_path=None,
                    pdf_sha256_expected=None,
                    pdf_sha256_actual=None,
                    anchor_used=None,
                    anchor_found=False,
                    anchor_page=None,
                    message=msg,
                )
            )
            error_count += 1
            continue

        page_start, page_end = page_range
        reader, pdf_path, expected_sha, actual_sha = _get_reader(doc_id)

        if reader is None or pdf_path is None:
            # Check whether the doc has a non-PDF artifact (pptx, html_snapshot).
            # These sources are valid for provenance but cannot be anchor-checked
            # via PdfReader.  Downgrade to warn instead of hard error.
            doc_entry = registry.get(doc_id)
            non_pdf_kinds = [
                a.get("kind")
                for a in (doc_entry.get("artifacts") or [] if doc_entry else [])
                if isinstance(a, dict) and a.get("kind") not in ("pdf",)
            ]
            if doc_entry and non_pdf_kinds:
                msg = (
                    f"{where}: non-pdf artifact ({', '.join(non_pdf_kinds)}) — "
                    f"slide/section citation accepted, anchor not machine-verifiable"
                )
                findings.append(
                    Finding(
                        rule_id=rid,
                        doc_id=doc_id,
                        loc=loc,
                        status="warn",
                        page_start=page_start,
                        page_end=page_end,
                        pdf_path=None,
                        pdf_sha256_expected=None,
                        pdf_sha256_actual=None,
                        anchor_used=None,
                        anchor_found=False,
                        anchor_page=None,
                        message=msg,
                    )
                )
                warn_count += 1
            else:
                msg = f"{where}: missing pdf artifact for doc_id={doc_id} (check registry + local docs/raw)"
                findings.append(
                    Finding(
                        rule_id=rid,
                        doc_id=doc_id,
                        loc=loc,
                        status="error",
                        page_start=page_start,
                        page_end=page_end,
                        pdf_path=str(pdf_path) if pdf_path else None,
                        pdf_sha256_expected=expected_sha,
                        pdf_sha256_actual=actual_sha,
                        anchor_used=None,
                        anchor_found=False,
                        anchor_page=None,
                        message=msg,
                    )
                )
                error_count += 1
            continue

        # sha256 mismatch => hard error (provenance broken)
        if expected_sha and actual_sha and expected_sha.lower() != actual_sha.lower():
            msg = f"{where}: pdf sha256 mismatch for {doc_id} (expected {expected_sha}, got {actual_sha})"
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="error",
                    page_start=page_start,
                    page_end=page_end,
                    pdf_path=str(pdf_path),
                    pdf_sha256_expected=expected_sha,
                    pdf_sha256_actual=actual_sha,
                    anchor_used=None,
                    anchor_found=False,
                    anchor_page=None,
                    message=msg,
                )
            )
            error_count += 1
            continue

        n_pages = len(reader.pages)
        if page_start < 1 or page_end < 1 or page_start > n_pages or page_end > n_pages:
            msg = f"{where}: loc pages out of range (p.{page_start}-{page_end}, pdf_pages={n_pages})"
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="error",
                    page_start=page_start,
                    page_end=page_end,
                    pdf_path=str(pdf_path),
                    pdf_sha256_expected=expected_sha,
                    pdf_sha256_actual=actual_sha,
                    anchor_used=None,
                    anchor_found=False,
                    anchor_page=None,
                    message=msg,
                )
            )
            error_count += 1
            continue

        anchors = candidate_anchors(loc)
        if not anchors:
            msg = f"{where}: no usable anchors parsed from loc; add a section number or short quote to loc"
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="warn",
                    page_start=page_start,
                    page_end=page_end,
                    pdf_path=str(pdf_path),
                    pdf_sha256_expected=expected_sha,
                    pdf_sha256_actual=actual_sha,
                    anchor_used=None,
                    anchor_found=False,
                    anchor_page=None,
                    message=msg,
                )
            )
            warn_count += 1
            continue

        found, anchor, anchor_page = find_anchor_in_pages(reader, page_start, page_end, anchors)
        if found:
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="ok",
                    page_start=page_start,
                    page_end=page_end,
                    pdf_path=str(pdf_path),
                    pdf_sha256_expected=expected_sha,
                    pdf_sha256_actual=actual_sha,
                    anchor_used=anchor,
                    anchor_found=True,
                    anchor_page=anchor_page,
                    message="anchor_found_on_cited_page",
                )
            )
        else:
            msg = (
                f"{where}: anchor not found on cited page(s) p.{page_start}-{page_end}. "
                f"Consider tightening loc with an exact short quote from the PDF."
            )
            findings.append(
                Finding(
                    rule_id=rid,
                    doc_id=doc_id,
                    loc=loc,
                    status="warn",
                    page_start=page_start,
                    page_end=page_end,
                    pdf_path=str(pdf_path),
                    pdf_sha256_expected=expected_sha,
                    pdf_sha256_actual=actual_sha,
                    anchor_used=anchors[0] if anchors else None,
                    anchor_found=False,
                    anchor_page=None,
                    message=msg,
                )
            )
            warn_count += 1

    return {
        "ok": (error_count == 0) and (warn_count == 0 or not strict),
        "rules_path": str(rules_path),
        "registry_path": str(registry_path),
        "only_doc": only_doc or None,
        "error_count": error_count,
        "warn_count": warn_count,
        "findings": [f.__dict__ for f in findings],
    }


def main() -> int:
    _best_effort_utf8_stdout()

    ap = argparse.ArgumentParser(description="Audit rule citations against local PDF artifacts")
    ap.add_argument("--rules", default=str(DEFAULT_RULES), help="Rules JSONL (default: rules/published.jsonl)")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Doc registry JSON (default: registry/doc_registry.json)")
    ap.add_argument("--only-doc", default="", help="Only audit a single doc_id (optional)")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors (exit non-zero)")
    ap.add_argument("--out", default="", help="Write full JSON report to a file (optional)")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    registry_path = Path(args.registry)
    report = audit_citations(
        rules_path=rules_path,
        registry_path=registry_path,
        only_doc=args.only_doc,
        strict=bool(args.strict),
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if int(report.get("error_count") or 0) > 0:
        return 2
    if args.strict and int(report.get("warn_count") or 0) > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

