from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from io_jsonl import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = REPO_ROOT / "rules"


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def index_by_rule_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = r.get("rule_id")
        if rid:
            out[str(rid)] = r
    return out


def promote(
    *,
    from_path: Path,
    to_path: Path,
    rule_ids: Set[str],
    new_status: str,
    reviewer: str,
    note: str,
) -> None:
    src = read_jsonl(from_path)
    dst = read_jsonl(to_path)
    dst_by_id = index_by_rule_id(dst)

    moved = 0
    kept_src: List[Dict[str, Any]] = []
    for r in src:
        rid = str(r.get("rule_id", ""))
        if rid and rid in rule_ids:
            r = dict(r)
            r["review_status"] = new_status
            r["reviewed_at"] = now_iso()
            r["reviewed_by"] = reviewer
            if note:
                r["review_notes"] = note
            dst_by_id[rid] = r
            moved += 1
        else:
            kept_src.append(r)

    write_jsonl(from_path, kept_src)
    write_jsonl(to_path, list(dst_by_id.values()))
    print(json.dumps({"moved": moved, "from": str(from_path), "to": str(to_path)}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal rule review/publish gate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_review = sub.add_parser("review", help="promote drafts -> reviewed")
    p_review.add_argument("--rule-id", action="append", required=True)
    p_review.add_argument("--reviewer", default="unknown")
    p_review.add_argument("--note", default="")

    p_pub = sub.add_parser("publish", help="promote reviewed -> published")
    p_pub.add_argument("--rule-id", action="append", required=True)
    p_pub.add_argument("--reviewer", default="unknown")
    p_pub.add_argument("--note", default="")

    args = ap.parse_args()

    if args.cmd == "review":
        promote(
            from_path=RULES_DIR / "drafts.jsonl",
            to_path=RULES_DIR / "reviewed.jsonl",
            rule_ids=set(args.rule_id),
            new_status="reviewed",
            reviewer=args.reviewer,
            note=args.note,
        )
        return 0

    if args.cmd == "publish":
        promote(
            from_path=RULES_DIR / "reviewed.jsonl",
            to_path=RULES_DIR / "published.jsonl",
            rule_ids=set(args.rule_id),
            new_status="published",
            reviewer=args.reviewer,
            note=args.note,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
