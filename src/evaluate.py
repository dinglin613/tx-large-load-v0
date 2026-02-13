from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from io_jsonl import iter_jsonl, read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "graph" / "process_graph.yaml"
RULES_PUBLISHED = REPO_ROOT / "rules" / "published.jsonl"
DOC_REGISTRY = REPO_ROOT / "registry" / "doc_registry.json"


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_doc_registry() -> List[Dict[str, Any]]:
    if not DOC_REGISTRY.exists():
        return []
    with DOC_REGISTRY.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_docs(registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {d.get("doc_id"): d for d in registry if d.get("doc_id")}


def doc_provenance_entry(doc: Dict[str, Any]) -> Dict[str, Any]:
    arts = doc.get("artifacts") or []
    # keep it small: for v0 we only need path+sha for audit trail
    return {
        "doc_id": doc.get("doc_id"),
        "title": doc.get("title"),
        "effective_date": doc.get("effective_date"),
        "artifacts": [
            {"path": a.get("path"), "sha256": a.get("sha256")}
            for a in arts
            if a.get("path") and a.get("sha256")
        ],
        "source_url": doc.get("source_url"),
    }


def get_field(req: Dict[str, Any], field: str) -> Any:
    return req.get(field)


def load_published_rules() -> Dict[str, Dict[str, Any]]:
    if not RULES_PUBLISHED.exists():
        return {}
    rules = read_jsonl(RULES_PUBLISHED)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rules:
        rid = r.get("rule_id")
        if not rid:
            continue
        if r.get("review_status") != "published":
            continue
        out[str(rid)] = r
    return out


def required_fields_from_rule(rule: Dict[str, Any]) -> List[str]:
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    if dsl.get("kind") == "completeness_check":
        rf = dsl.get("required_fields") or []
        return [str(x) for x in rf]
    return []


def risk_from_matched_rules(req: Dict[str, Any], matched_rules: List[Dict[str, Any]], missing_inputs: List[str]) -> Dict[str, Any]:
    # v0: qualitative, rule-driven knobs only (no numeric "probabilities")
    upgrade_score = 0.0
    wait_score = 0.0
    ops_score = 0.0
    evidence: List[Dict[str, Any]] = []

    for r in matched_rules:
        tags = set(r.get("trigger_tags") or [])
        rid = r.get("rule_id")
        evidence.append(
            {
                "rule_id": rid,
                "doc_id": r.get("doc_id"),
                "loc": r.get("loc"),
                "trigger_tags": sorted(tags),
            }
        )
        if "data_completeness" in tags:
            # if any required fields missing, count as waiting/latency risk
            required = required_fields_from_rule(r)
            if any((req.get(f) in (None, "", [])) for f in required):
                wait_score += 2.0
        if "process_latency" in tags:
            wait_score += 0.5
        if "timeline_risk" in tags:
            wait_score += 1.0
        if "re_study" in tags:
            wait_score += 1.5
        if "queue_density" in tags or "queue_dependency" in tags or "must_study" in tags:
            wait_score += 1.0
        if "study_dependency" in tags:
            wait_score += 0.5
        if "financial_security" in tags:
            wait_score += 0.5
        if "modeling_requirement" in tags or "telemetry_requirement" in tags:
            wait_score += 0.5
        if "protection" in tags or "short_circuit" in tags:
            upgrade_score += 1.0
        if "upgrade_exposure" in tags:
            upgrade_score += 0.5
        if "new_substation" in tags or "new_line" in tags:
            upgrade_score += 3.0
        if "dynamic_stability" in tags or "dynamic_study" in tags or "sso" in tags:
            upgrade_score += 1.0
        if "curtailment_risk" in tags:
            ops_score += 2.0
        if "operational_constraint" in tags or "commissioning_limit" in tags:
            ops_score += 1.0

    # request-level heuristics (still explainable, but not tied to specific clause)
    try:
        mw = float(req.get("load_mw_total") or 0)
    except Exception:
        mw = 0.0
    if mw >= 500:
        upgrade_score += 2.0
    elif mw >= 300:
        upgrade_score += 1.0

    if req.get("energization_plan") == "phased":
        upgrade_score -= 0.5

    # Missing inputs: if evaluator already sees missing core fields, raise wait risk
    if missing_inputs:
        wait_score += 0.5

    # Buckets are qualitative “pressure” indicators
    le_12 = "unknown"
    m12_24 = "unknown"
    gt_24 = "unknown"
    exposure = "unknown"
    operational_exposure = "unknown"

    if matched_rules:
        if upgrade_score >= 3:
            gt_24 = "up"
            m12_24 = "up"
            le_12 = "down"
            exposure = "high"
        elif upgrade_score >= 0.75:
            m12_24 = "up"
            le_12 = "down"
            exposure = "medium"
        else:
            le_12 = "up"
            exposure = "low"

        status = "rule_driven_v0"
        if wait_score >= 2:
            # waiting pressure shifts mass away from <=12
            le_12 = "down" if le_12 != "unknown" else "down"
            m12_24 = "up" if m12_24 != "unknown" else "up"
    else:
        status = "insufficient_rule_coverage"

    if matched_rules:
        if ops_score >= 2:
            operational_exposure = "high"
        elif ops_score >= 1:
            operational_exposure = "medium"
        else:
            operational_exposure = "low"

    return {
        "timeline_buckets": {
            "le_12_months": le_12,
            "m12_24_months": m12_24,
            "gt_24_months": gt_24,
        },
        "upgrade_exposure_bucket": exposure,
        "operational_exposure_bucket": operational_exposure,
        "status": status,
        "evidence": evidence,
    }

def eval_pred(req: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[bool, str]:
    op = pred.get("op")
    field = pred.get("field")
    if not op:
        return False, "missing op"
    if op == "exists":
        if not field:
            return False, "missing field"
        v = get_field(req, field)
        ok = v is not None and v != "" and v != []
        return ok, f"exists({field})={ok}"
    if op == "equals":
        if not field:
            return False, "missing field"
        v = get_field(req, field)
        ok = v == pred.get("value")
        return ok, f"{field}=={pred.get('value')} ({v})"
    if op == "gte":
        if not field:
            return False, "missing field"
        v = get_field(req, field)
        try:
            ok = float(v) >= float(pred.get("value"))
        except Exception:
            return False, f"gte({field}) invalid numeric"
        return ok, f"{field}>={pred.get('value')} ({v})"
    if op == "any_of":
        if not field:
            return False, "missing field"
        v = get_field(req, field)
        options = pred.get("values") or []
        ok = v in options
        return ok, f"{field} in {options} ({v})"
    return False, f"unknown op: {op}"


def criteria_satisfied(req: Dict[str, Any], preds: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    traces: List[str] = []
    for p in preds or []:
        ok, msg = eval_pred(req, p)
        traces.append(msg)
        if not ok:
            return False, traces
    return True, traces


def rule_predicates(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    preds = dsl.get("predicates")
    if isinstance(preds, list):
        return [p for p in preds if isinstance(p, dict)]
    return []


def rule_matches_request(rule: Dict[str, Any], req: Dict[str, Any]) -> Tuple[bool, List[str]]:
    preds = rule_predicates(rule)
    if not preds:
        return False, []
    return criteria_satisfied(req, preds)


def evaluate_graph(req: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = graph.get("edges") or []
    nodes_by_id = {n.get("id"): n for n in (graph.get("nodes") or [])}
    rules_by_id = load_published_rules()
    docs_by_id = index_docs(load_doc_registry())

    current = "N0"
    path_nodes: List[str] = [current]
    traversed_edges: List[Dict[str, Any]] = []
    missing_inputs: List[str] = []
    matched_rules: List[Dict[str, Any]] = []

    # deterministic traversal: at each node, pick the first satisfied outgoing edge in file order
    for _ in range(200):
        outgoing = [e for e in edges if e.get("from") == current]
        if not outgoing:
            break
        chosen = None
        chosen_traces: List[str] = []
        for e in outgoing:
            ok, traces = criteria_satisfied(req, e.get("criteria") or [])
            if ok:
                chosen = e
                chosen_traces = traces
                break
            # collect missing fields for "exists" predicates that failed
            for p in e.get("criteria") or []:
                if p.get("op") == "exists" and p.get("field") and not get_field(req, p["field"]):
                    missing_inputs.append(p["field"])
        if chosen is None:
            break

        to = chosen.get("to")
        traversed_edges.append(
            {
                "edge_id": chosen.get("id"),
                "from": chosen.get("from"),
                "to": to,
                "criteria_traces": chosen_traces,
                "rule_ids": chosen.get("rule_ids") or [],
            }
        )
        for rid in (chosen.get("rule_ids") or []):
            r = rules_by_id.get(rid)
            if r:
                matched_rules.append(r)
        if not to:
            break
        current = to
        path_nodes.append(current)
        if current == "N6":
            break

    # Levers: purely structural (no claims)
    levers = []
    if req.get("energization_plan") == "phased":
        levers.append(
            {
                "lever_id": "phased_energization",
                "status": "present_in_request",
                "note": "Request includes a phased energization plan (can be used as an option lever).",
            }
        )
    if isinstance(req.get("voltage_options_kv"), list) and len(req["voltage_options_kv"]) >= 2:
        levers.append(
            {
                "lever_id": "voltage_level_choice",
                "status": "present_in_request",
                "note": "Request includes multiple voltage options (can be evaluated as alternative options).",
            }
        )

    # Parallel flags: rules that match request but are not necessarily on the main path
    matched_rule_ids = {r.get("rule_id") for r in matched_rules if r.get("rule_id")}
    flags: List[Dict[str, Any]] = []
    flagged_rules: List[Dict[str, Any]] = []

    for rid, r in rules_by_id.items():
        if rid in matched_rule_ids:
            continue
        ok, traces = rule_matches_request(r, req)
        if ok:
            flagged_rules.append(r)
            flags.append(
                {
                    "rule_id": rid,
                    "doc_id": r.get("doc_id"),
                    "loc": r.get("loc"),
                    "trigger_tags": r.get("trigger_tags") or [],
                    "match_traces": traces,
                }
            )

    all_signal_rules = matched_rules + flagged_rules

    # Missing inputs from completeness-check rules (published)
    for r in all_signal_rules:
        for f in required_fields_from_rule(r):
            if req.get(f) in (None, "", []):
                missing_inputs.append(f)

    risk = risk_from_matched_rules(req, all_signal_rules, missing_inputs=sorted(set(missing_inputs)))

    # Attach provenance for any docs referenced by evidence
    used_doc_ids = sorted({e.get("doc_id") for e in (risk.get("evidence") or []) if e.get("doc_id")})
    used_docs = [doc_provenance_entry(docs_by_id[d]) for d in used_doc_ids if d in docs_by_id]

    return {
        "evaluated_at": now_iso(),
        "request": req,
        "graph": {
            "path_nodes": path_nodes,
            "path_node_labels": [nodes_by_id.get(n, {}).get("label", n) for n in path_nodes],
            "traversed_edges": traversed_edges,
            "graph_version": graph.get("version"),
        },
        "missing_inputs": sorted(set(missing_inputs)),
        "levers": levers,
        "flags": flags,
        "risk": risk,
        "provenance": {
            "doc_registry_count": len(load_doc_registry()),
            "rules_source": "rules/published.jsonl",
            "docs": used_docs,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate process graph for large load requests")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL requests")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL evaluations")
    args = ap.parse_args()

    graph = load_yaml(GRAPH_PATH)

    rows_out: List[Dict[str, Any]] = []
    for req in iter_jsonl(args.in_path):
        rows_out.append(evaluate_graph(req, graph))

    write_jsonl(args.out_path, rows_out)
    print(f"Wrote {len(rows_out)} evaluations to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
