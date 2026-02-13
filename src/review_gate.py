from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from io_jsonl import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = REPO_ROOT / "rules"
SCHEMA_DIR = REPO_ROOT / "schema"
REGISTRY_DIR = REPO_ROOT / "registry"


_CACHE_REQUEST_FIELDS: Set[str] | None = None
_CACHE_DOC_IDS: Set[str] | None = None


def load_request_fields() -> Set[str]:
    global _CACHE_REQUEST_FIELDS
    if _CACHE_REQUEST_FIELDS is not None:
        return _CACHE_REQUEST_FIELDS
    path = SCHEMA_DIR / "request.schema.json"
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    props = schema.get("properties") or {}
    fields = {str(k) for k in props.keys()} if isinstance(props, dict) else set()
    # evaluator supports legacy alias
    fields.add("voltage_options")
    _CACHE_REQUEST_FIELDS = fields
    return fields


def load_doc_ids() -> Set[str]:
    global _CACHE_DOC_IDS
    if _CACHE_DOC_IDS is not None:
        return _CACHE_DOC_IDS
    path = REGISTRY_DIR / "doc_registry.json"
    if not path.exists():
        _CACHE_DOC_IDS = set()
        return _CACHE_DOC_IDS
    with path.open("r", encoding="utf-8") as f:
        reg = json.load(f)
    doc_ids = {str(d.get("doc_id")) for d in (reg or []) if isinstance(d, dict) and d.get("doc_id")}
    _CACHE_DOC_IDS = doc_ids
    return doc_ids


ALLOWED_PRED_OPS = {"exists", "equals", "eq", "gte", "any_of", "any_true"}


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


def validate_for_publish(rule: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    loc = str(rule.get("loc", "")).strip()
    if (not loc) or ("tbd" in loc.lower()):
        errs.append("loc_missing_or_tbd")

    conf = rule.get("confidence", None)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = -1.0
    if conf_f < 0.7:
        errs.append("confidence_lt_0_7")

    tags = rule.get("trigger_tags") or []
    if not isinstance(tags, list) or len(tags) == 0:
        errs.append("trigger_tags_empty")

    doc_id = str(rule.get("doc_id", "")).strip()
    if not doc_id:
        errs.append("doc_id_missing")
    elif doc_id not in load_doc_ids():
        errs.append("doc_id_not_in_registry")

    # DSL lint: field references must exist in request schema (fail-fast on typos).
    request_fields = load_request_fields()
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    if isinstance(dsl, dict):
        rf = dsl.get("required_fields") or []
        if isinstance(rf, list):
            for f in rf:
                if f and str(f) not in request_fields:
                    errs.append("dsl_required_fields_unknown")
                    break
        preds = dsl.get("predicates") or []
        if isinstance(preds, list):
            for p in preds:
                if not isinstance(p, dict):
                    errs.append("dsl_predicate_not_object")
                    break
                op = p.get("op")
                if op not in ALLOWED_PRED_OPS:
                    errs.append("dsl_predicate_unknown_op")
                    break
                field = p.get("field")
                if field and str(field) not in request_fields:
                    errs.append("dsl_predicate_unknown_field")
                    break
        if dsl.get("kind") == "conditional_gate":
            af = dsl.get("applicability_fields") or []
            if not isinstance(af, list) or len(af) == 0:
                errs.append("dsl_missing_applicability_fields")
            else:
                for f in af:
                    if f and str(f) not in request_fields:
                        errs.append("dsl_applicability_fields_unknown")
                        break

    return errs


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
    p_pub.add_argument("--allow-unsafe", action="store_true", help="Bypass v0 publish quality checks")

    p_review_all = sub.add_parser("review-all", help="promote all drafts -> reviewed")
    p_review_all.add_argument("--reviewer", default="unknown")
    p_review_all.add_argument("--note", default="")

    p_pub_all = sub.add_parser("publish-all", help="promote all reviewed -> published")
    p_pub_all.add_argument("--reviewer", default="unknown")
    p_pub_all.add_argument("--note", default="")
    p_pub_all.add_argument("--allow-unsafe", action="store_true", help="Bypass v0 publish quality checks")

    p_check = sub.add_parser("check-published", help="validate current published rule set")

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
        if not args.allow_unsafe:
            reviewed = read_jsonl(RULES_DIR / "reviewed.jsonl")
            reviewed_by_id = index_by_rule_id(reviewed)
            bad: Dict[str, List[str]] = {}
            for rid in args.rule_id:
                r = reviewed_by_id.get(rid)
                if not r:
                    continue
                errs = validate_for_publish(r)
                if errs:
                    bad[rid] = errs
            if bad:
                print(json.dumps({"publish_blocked": bad}, ensure_ascii=False, indent=2))
                return 3

        promote(
            from_path=RULES_DIR / "reviewed.jsonl",
            to_path=RULES_DIR / "published.jsonl",
            rule_ids=set(args.rule_id),
            new_status="published",
            reviewer=args.reviewer,
            note=args.note,
        )
        return 0

    if args.cmd == "review-all":
        drafts = read_jsonl(RULES_DIR / "drafts.jsonl")
        rule_ids = {str(r.get("rule_id")) for r in drafts if r.get("rule_id")}
        promote(
            from_path=RULES_DIR / "drafts.jsonl",
            to_path=RULES_DIR / "reviewed.jsonl",
            rule_ids=rule_ids,
            new_status="reviewed",
            reviewer=args.reviewer,
            note=args.note,
        )
        return 0

    if args.cmd == "publish-all":
        reviewed = read_jsonl(RULES_DIR / "reviewed.jsonl")
        rule_ids = [str(r.get("rule_id")) for r in reviewed if r.get("rule_id")]
        if not args.allow_unsafe:
            bad: Dict[str, List[str]] = {}
            for rid in rule_ids:
                r = next((x for x in reviewed if str(x.get("rule_id")) == rid), None)
                if not r:
                    continue
                errs = validate_for_publish(r)
                if errs:
                    bad[rid] = errs
            if bad:
                print(json.dumps({"publish_blocked": bad}, ensure_ascii=False, indent=2))
                return 3

        promote(
            from_path=RULES_DIR / "reviewed.jsonl",
            to_path=RULES_DIR / "published.jsonl",
            rule_ids=set(rule_ids),
            new_status="published",
            reviewer=args.reviewer,
            note=args.note,
        )
        return 0

    if args.cmd == "check-published":
        published = read_jsonl(RULES_DIR / "published.jsonl")
        bad: Dict[str, List[str]] = {}
        for r in published:
            rid = str(r.get("rule_id", ""))
            errs = validate_for_publish(r)
            if rid and errs:
                bad[rid] = errs
        if bad:
            print(json.dumps({"published_invalid": bad}, ensure_ascii=False, indent=2))
            return 4
        print(json.dumps({"published_ok": True, "count": len(published)}, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
