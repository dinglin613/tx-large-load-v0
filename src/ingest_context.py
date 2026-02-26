"""
src/ingest_context.py
=====================
Context pipeline for tx-large-load-v0.

Purpose
-------
Fetches and archives snapshots of *non-authoritative* external sources
(e.g., datacenter.fyi, news) and *authoritative* web pages (e.g., ERCOT calendar)
that are useful as *context* for the Decision Memo, but must NOT enter the
authoritative rule evidence chain.

Design contract
---------------
- Authoritative rule evidence  → docs/raw/ or docs/snapshots/ + registry/doc_registry.json + src/ingest.py
- Context (signals / leads)    → docs/snapshots/context/<source_id>/<YYYY-MM-DD>.html
                                  + context/context_snapshot.jsonl

Every context entry in context_snapshot.jsonl MUST carry:
  - source_id, url, retrieved_at, sha256 (of the snapshot file)
  - confidence (default from context_sources.json, can be overridden)
  - category: "authoritative" | "non_authoritative"
  - notes

Context snapshot entries feed into render_memo.py for display in the
"Context snapshot (non-audit evidence)" section of the memo.
They do NOT participate in citation_audit and do NOT affect path evaluation.

Usage
-----
    python src/ingest_context.py                          # fetch all enabled sources
    python src/ingest_context.py --source-id datacenter_fyi_oncor
    python src/ingest_context.py --dry-run                # print what would be fetched, no writes
    python src/ingest_context.py --list                   # list configured sources

Output
------
    context/context_snapshot.jsonl  (upserted; keyed by source_id + date)
    docs/snapshots/context/<source_id>/<YYYY-MM-DD>.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SOURCES_PATH = REPO_ROOT / "context" / "context_sources.json"
CONTEXT_SNAPSHOT_PATH = REPO_ROOT / "context" / "context_snapshot.jsonl"
SNAPSHOTS_CONTEXT_DIR = REPO_ROOT / "docs" / "snapshots" / "context"

# Timeout for HTTP fetches (seconds)
FETCH_TIMEOUT = 20

# Max body size to save (bytes). Truncate beyond this to avoid huge HTML files.
MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MB


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def today_str() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_sources() -> List[Dict[str, Any]]:
    with CONTEXT_SOURCES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshot_index() -> Dict[str, Dict[str, Any]]:
    """
    Load context_snapshot.jsonl as a dict keyed by (source_id, date).
    Key = f"{source_id}::{date}"
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not CONTEXT_SNAPSHOT_PATH.exists():
        return index
    with CONTEXT_SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                k = f"{entry.get('source_id', '')}::{entry.get('snapshot_date', '')}"
                index[k] = entry
            except Exception:
                pass
    return index


def save_snapshot_index(index: Dict[str, Dict[str, Any]]) -> None:
    CONTEXT_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Sort by source_id then date for stable output
    rows = sorted(index.values(), key=lambda e: (e.get("source_id", ""), e.get("snapshot_date", "")))
    with CONTEXT_SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def fetch_url(url: str) -> Optional[bytes]:
    """
    Fetch URL content with a simple User-Agent. Returns raw bytes or None on failure.
    Caller should handle and log errors.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "tx-large-load-v0/context-ingest "
                "(research/educational; contact: see repo README)"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            body = resp.read(MAX_BODY_BYTES)
            return body
    except urllib.error.HTTPError as e:
        print(f"  [WARN] HTTP {e.code} fetching {url}: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [WARN] Error fetching {url}: {e}", file=sys.stderr)
        return None


def ingest_source(
    source: Dict[str, Any],
    *,
    dry_run: bool,
    date: str,
    snapshot_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fetch and archive a single context source.

    Returns a context snapshot entry (whether or not dry_run).
    """
    source_id: str = source["source_id"]
    url: str = source["url"]
    category: str = source.get("category", "non_authoritative")
    confidence: float = float(source.get("default_confidence", 0.2))
    notes: str = source.get("notes", "")
    label: str = source.get("label", source_id)

    snapshot_key = f"{source_id}::{date}"
    existing = snapshot_index.get(snapshot_key)

    if dry_run:
        status = "existing" if existing else "would_fetch"
        print(f"  [dry_run] {source_id}: {status} | {url}")
        return existing or {
            "source_id": source_id,
            "label": label,
            "url": url,
            "category": category,
            "snapshot_date": date,
            "retrieved_at": now_iso(),
            "sha256": "",
            "snapshot_path": "",
            "confidence": confidence,
            "notes": notes,
            "fetch_status": "dry_run",
        }

    if existing and existing.get("sha256") and existing.get("fetch_status") == "ok":
        print(f"  [skip] {source_id}: already fetched for {date}")
        return existing

    print(f"  [fetch] {source_id}: {url}")
    body = fetch_url(url)

    if body is None:
        entry = {
            "source_id": source_id,
            "label": label,
            "url": url,
            "category": category,
            "snapshot_date": date,
            "retrieved_at": now_iso(),
            "sha256": "",
            "snapshot_path": "",
            "confidence": max(0.0, confidence - 0.1),  # slightly penalize failed fetch
            "notes": notes,
            "fetch_status": "error",
        }
        snapshot_index[snapshot_key] = entry
        return entry

    sha = sha256_bytes(body)

    # Save to docs/snapshots/context/<source_id>/<date>.html
    out_dir = SNAPSHOTS_CONTEXT_DIR / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.html"
    out_path.write_bytes(body)

    rel_path = out_path.relative_to(REPO_ROOT).as_posix()

    entry = {
        "source_id": source_id,
        "label": label,
        "url": url,
        "category": category,
        "snapshot_date": date,
        "retrieved_at": now_iso(),
        "sha256": sha,
        "snapshot_path": rel_path,
        "size_bytes": len(body),
        "confidence": confidence,
        "notes": notes,
        "fetch_status": "ok",
    }
    snapshot_index[snapshot_key] = entry
    print(f"    → saved {rel_path} (sha256={sha[:16]}...)")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch and archive context snapshots (non-authoritative external sources)."
    )
    ap.add_argument(
        "--source-id",
        dest="source_ids",
        action="append",
        metavar="ID",
        help="Fetch only this source_id (repeatable). Default: all enabled sources.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without writing any files.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List all configured context sources and exit.",
    )
    ap.add_argument(
        "--date",
        default=today_str(),
        help="Override snapshot date (YYYY-MM-DD). Default: today UTC.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Include disabled sources (enabled=false). Default: skip disabled.",
    )
    args = ap.parse_args()

    if not CONTEXT_SOURCES_PATH.exists():
        print(f"ERROR: context_sources.json not found at {CONTEXT_SOURCES_PATH}", file=sys.stderr)
        return 1

    sources = load_sources()

    if args.list:
        print(f"{'source_id':<45} {'category':<20} {'enabled':<8} url")
        print("-" * 120)
        for s in sources:
            enabled = "yes" if s.get("enabled", True) else "no"
            print(f"{s['source_id']:<45} {s.get('category',''):<20} {enabled:<8} {s.get('url','')}")
        return 0

    # Filter
    if not args.all:
        sources = [s for s in sources if s.get("enabled", True)]
    if args.source_ids:
        sid_set = set(args.source_ids)
        sources = [s for s in sources if s["source_id"] in sid_set]
        missing = sid_set - {s["source_id"] for s in sources}
        if missing:
            print(f"[WARN] source_ids not found (or disabled): {sorted(missing)}", file=sys.stderr)

    if not sources:
        print("No sources to process.")
        return 0

    snapshot_index = load_snapshot_index()
    date = args.date

    print(f"ingest_context: processing {len(sources)} source(s) for date={date}")
    for source in sources:
        ingest_source(source, dry_run=args.dry_run, date=date, snapshot_index=snapshot_index)

    if not args.dry_run:
        save_snapshot_index(snapshot_index)
        print(f"\nUpdated: {CONTEXT_SNAPSHOT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
