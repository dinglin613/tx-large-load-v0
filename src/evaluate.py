from __future__ import annotations

import argparse
import hashlib
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


_SHA256_CACHE: Dict[str, str] = {}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_cached(path: Path) -> str:
    k = str(path.resolve())
    v = _SHA256_CACHE.get(k)
    if v:
        return v
    if not path.exists():
        return ""
    v = sha256_file(path)
    _SHA256_CACHE[k] = v
    return v


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
    # Backward/alias support for request fields
    if field in req:
        return req.get(field)
    if field == "voltage_options_kv" and "voltage_options" in req:
        return req.get("voltage_options")
    return None


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
    # Generic required_fields (works for any rule type)
    generic = dsl.get("required_fields") or []
    out: List[str] = [str(x) for x in generic] if isinstance(generic, list) else []
    if dsl.get("kind") == "completeness_check":
        rf = dsl.get("required_fields") or []
        out.extend([str(x) for x in rf])
    # dedupe while preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


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

def eval_pred(req: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    op = pred.get("op")
    field = pred.get("field")
    if not op:
        return False, "missing op", []
    # aliases
    if op == "eq":
        op = "equals"
    if op == "exists":
        if not field:
            return False, "missing field", []
        v = get_field(req, field)
        ok = v is not None and v != "" and v != []
        missing = [field] if not ok else []
        return ok, f"exists({field})={ok}", missing
    if op == "equals":
        if not field:
            return False, "missing field", []
        v = get_field(req, field)
        if v is None:
            return False, f"{field} missing", [field]
        ok = v == pred.get("value")
        return ok, f"{field}=={pred.get('value')} ({v})", []
    if op == "gte":
        if not field:
            return False, "missing field", []
        v = get_field(req, field)
        if v is None:
            return False, f"{field} missing", [field]
        try:
            ok = float(v) >= float(pred.get("value"))
        except Exception:
            return False, f"gte({field}) invalid numeric", []
        return ok, f"{field}>={pred.get('value')} ({v})", []
    if op == "any_of":
        if not field:
            return False, "missing field", []
        v = get_field(req, field)
        if v is None:
            return False, f"{field} missing", [field]
        options = pred.get("values") or pred.get("value") or []
        ok = v in options
        return ok, f"{field} in {options} ({v})", []
    if op == "any_true":
        if not field:
            return False, "missing field", []
        v = get_field(req, field)
        if v is None:
            return False, f"{field} missing", [field]
        if not isinstance(v, list):
            return False, f"{field} not a list", []
        options = pred.get("value") or pred.get("values") or []
        ok = any(x in options for x in v)
        return ok, f"any_true({field} in {options}) ({v})", []
    return False, f"unknown op: {op}", []


def criteria_satisfied(req: Dict[str, Any], preds: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    traces: List[str] = []
    missing: List[str] = []
    for p in preds or []:
        ok, msg, missing_fields = eval_pred(req, p)
        traces.append(msg)
        missing.extend(missing_fields)
        if not ok:
            return False, traces, sorted(set(missing))
    return True, traces, sorted(set(missing))


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
    ok, traces, _missing = criteria_satisfied(req, preds)
    return ok, traces


def normalize_request(req: Dict[str, Any]) -> Dict[str, Any]:
    # Make test/demo inputs forgiving.
    r = dict(req)
    if "project_type" not in r and r.get("operator_area") == "ERCOT":
        r["project_type"] = "large_load"
    if "voltage_options_kv" not in r and isinstance(r.get("voltage_options"), list):
        r["voltage_options_kv"] = r["voltage_options"]
    return r


def has_gate_predicate(rule: Dict[str, Any], *, field: str, value: Any = True) -> bool:
    for p in rule_predicates(rule):
        op = p.get("op")
        if op == "eq":
            op = "equals"
        if op == "equals" and p.get("field") == field and p.get("value") == value:
            return True
    return False


def eval_rule_tri_state(rule: Dict[str, Any], req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a rule with predicates into a tri-state:
      - satisfied
      - missing (some required fields absent)
      - not_satisfied
      - not_applicable (explicit gate predicate disables it)
    """
    rid = rule.get("rule_id")
    preds = rule_predicates(rule)
    required = required_fields_from_rule(rule)
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    is_completeness = dsl.get("kind") == "completeness_check"

    # If rule is explicitly gated on energization, treat it as not applicable unless requesting energization.
    if has_gate_predicate(rule, field="is_requesting_energization", value=True) and req.get("is_requesting_energization") is not True:
        return {
            "rule_id": rid,
            "doc_id": rule.get("doc_id"),
            "loc": rule.get("loc"),
            "trigger_tags": rule.get("trigger_tags") or [],
            "status": "not_applicable",
            "missing_fields": [],
            "traces": [],
        }

    # Conditional gates ("if required ...") should not fail when the request explicitly says the gate is not required.
    #
    # IMPORTANT: only fields listed in dsl.applicability_fields are treated as applicability toggles.
    # Completion fields (e.g., telemetry_operational_and_accurate) must still fail if False.
    if dsl.get("kind") == "conditional_gate":
        app_fields = dsl.get("applicability_fields") or []
        if isinstance(app_fields, list):
            for field in [str(x) for x in app_fields if x]:
                v = get_field(req, field)
                if v is False:
                    return {
                        "rule_id": rid,
                        "doc_id": rule.get("doc_id"),
                        "loc": rule.get("loc"),
                        "trigger_tags": rule.get("trigger_tags") or [],
                        "status": "not_applicable",
                        "missing_fields": [],
                        "traces": [f"{field}==True (False) => not_applicable"],
                    }

    # Missing required fields.
    #
    # v0 nuance:
    # - For "completeness_check" rules, many request fields are modeled as booleans
    #   where False means "not provided" (e.g., one_line_diagram=false). Treat False as missing.
    # - For non-completeness rules, False is usually a valid factual value (e.g., agreements_executed=false),
    #   and should not be treated as "missing".
    def _missing_required_field(field: str) -> bool:
        if field not in req:
            return True
        v = req.get(field)
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        if is_completeness and (v is False):
            return True
        if is_completeness and isinstance(v, list) and len(v) == 0:
            return True
        return False

    missing_required = [f for f in required if _missing_required_field(f)]

    if not preds:
        # No predicates => cannot evaluate satisfied/not_satisfied; report missing if required fields exist.
        return {
            "rule_id": rid,
            "doc_id": rule.get("doc_id"),
            "loc": rule.get("loc"),
            "trigger_tags": rule.get("trigger_tags") or [],
            "status": "missing" if missing_required else "unknown",
            "missing_fields": missing_required,
            "traces": [],
        }

    ok, traces, missing_from_preds = criteria_satisfied(req, preds)
    missing_all = sorted(set(missing_required + missing_from_preds))

    if missing_all:
        status = "missing"
    else:
        status = "satisfied" if ok else "not_satisfied"

    return {
        "rule_id": rid,
        "doc_id": rule.get("doc_id"),
        "loc": rule.get("loc"),
        "trigger_tags": rule.get("trigger_tags") or [],
        "status": status,
        "missing_fields": missing_all,
        "traces": traces,
    }


def evaluate_graph(req: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    req = normalize_request(req)
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
            ok, traces, missing_fields = criteria_satisfied(req, e.get("criteria") or [])
            if ok:
                chosen = e
                chosen_traces = traces
                break
            missing_inputs.extend(missing_fields)
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

    # Rule checks (predicate-backed), for memo-grade checklists.
    rule_checks: List[Dict[str, Any]] = []
    energization_checklist: List[Dict[str, Any]] = []

    for rid, r in rules_by_id.items():
        if not rule_predicates(r) and not required_fields_from_rule(r):
            continue
        chk = eval_rule_tri_state(r, req)
        rule_checks.append(chk)

        tags = set(r.get("trigger_tags") or [])
        stage = str(r.get("stage") or "")
        if ("energization_gate" in tags) or (stage.lower() == "energization"):
            # Only show energization checklist when user is explicitly requesting energization,
            # OR when the rule isn't gated (rare).
            if req.get("is_requesting_energization") is True or not has_gate_predicate(r, field="is_requesting_energization", value=True):
                energization_checklist.append(chk)

    # Missing inputs from completeness-check rules (published)
    for r in all_signal_rules:
        rf = required_fields_from_rule(r)
        if not rf:
            continue

        dsl = ((r.get("criteria") or {}).get("dsl") or {})
        is_completeness = dsl.get("kind") == "completeness_check"
        if not is_completeness:
            # Only enforce non-completeness required_fields when the rule is "in play".
            preds = rule_predicates(r)
            if preds:
                ok, _tr, _miss = criteria_satisfied(req, preds)
                if not ok:
                    # Special-case energization gates: only relevant when requesting energization.
                    if has_gate_predicate(r, field="is_requesting_energization", value=True) and req.get("is_requesting_energization") is not True:
                        continue
        for f in rf:
            # Same missing semantics as tri-state (but applied to memo-level missing_inputs list)
            if f not in req:
                missing_inputs.append(f)
                continue
            v = req.get(f)
            if v is None:
                missing_inputs.append(f)
                continue
            if isinstance(v, str) and v.strip() == "":
                missing_inputs.append(f)
                continue
            if is_completeness and (v is False):
                missing_inputs.append(f)
                continue
            if is_completeness and isinstance(v, list) and len(v) == 0:
                missing_inputs.append(f)
                continue

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
        "rule_checks": rule_checks,
        "energization_checklist": energization_checklist,
        "risk": risk,
        "provenance": {
            "doc_registry_count": len(load_doc_registry()),
            "graph_source": "graph/process_graph.yaml",
            "graph_sha256": sha256_cached(GRAPH_PATH),
            "rules_source": "rules/published.jsonl",
            "rules_sha256": sha256_cached(RULES_PUBLISHED),
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
