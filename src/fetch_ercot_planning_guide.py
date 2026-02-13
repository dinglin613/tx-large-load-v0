from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RAW_DIR = REPO_ROOT / "docs" / "raw"
REGISTRY_PATH = REPO_ROOT / "registry" / "doc_registry.json"

# ERCOT Planning Guide Library (2025) → December 15, 2025 Combined PDF
DEFAULT_URL = "https://www.ercot.com/files/docs/2025/12/14/December-15-2025-Planning-Guide.pdf"
DEFAULT_DOC_ID = "ERCOT_PLANNING_GUIDE"
DEFAULT_EFFECTIVE_DATE = "2025-12-15"


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


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tx-large-load-v0/0.1 (citation-audit)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def load_registry() -> List[Dict[str, Any]]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: List[Dict[str, Any]]) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        f.write("\n")


def today_iso() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()


def main() -> int:
    _best_effort_utf8_stdout()

    ap = argparse.ArgumentParser(description="Fetch ERCOT Planning Guide PDF and update registry artifacts")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    ap.add_argument("--effective-date", default=DEFAULT_EFFECTIVE_DATE)
    ap.add_argument("--out", default="", help="Override output path (defaults to docs/raw/DOC_ID__DATE.pdf)")
    args = ap.parse_args()

    doc_id = str(args.doc_id).strip()
    effective_date = str(args.effective_date).strip()
    if not doc_id or not effective_date:
        raise SystemExit("doc-id and effective-date required")

    out_path = Path(args.out) if args.out else (DOCS_RAW_DIR / f"{doc_id}__{effective_date}.pdf")
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    if out_path.exists():
        print(json.dumps({"skipped": True, "reason": "already_exists", "path": str(out_path)}, ensure_ascii=False))
    else:
        download(args.url, out_path)
        print(json.dumps({"downloaded": True, "url": args.url, "path": str(out_path)}, ensure_ascii=False))

    sha = sha256_file(out_path)
    size = out_path.stat().st_size
    print(json.dumps({"sha256": sha, "size_bytes": size}, ensure_ascii=False))

    # Run ingest to upsert artifacts[] for all known docs.
    try:
        from ingest import main as ingest_main  # type: ignore

        ingest_main()
    except Exception as e:
        print(json.dumps({"ingest_error": str(e)}, ensure_ascii=False))
        return 4

    # Update Planning Guide top-level metadata to match latest fetched combined PDF.
    registry = load_registry()
    updated = False
    for d in registry:
        if not isinstance(d, dict):
            continue
        if str(d.get("doc_id")) != doc_id:
            continue
        d["source_url"] = str(args.url)
        d["effective_date"] = effective_date
        d["retrieved_date"] = today_iso()
        d["hash"] = sha.upper()
        updated = True
        break

    if updated:
        save_registry(registry)
        print(json.dumps({"registry_updated": True, "doc_id": doc_id, "effective_date": effective_date}, ensure_ascii=False))
    else:
        print(json.dumps({"registry_updated": False, "reason": "doc_id_not_found", "doc_id": doc_id}, ensure_ascii=False))
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

