from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "schema"

DEFAULT_REQUEST_SCHEMA = SCHEMA_DIR / "request.schema.json"
DEFAULT_RULE_SCHEMA = SCHEMA_DIR / "rule.schema.json"
DEFAULT_DOC_REGISTRY = REPO_ROOT / "registry" / "doc_registry.json"
DEFAULT_RULES_PUBLISHED = REPO_ROOT / "rules" / "published.jsonl"
DEFAULT_GRAPH = REPO_ROOT / "graph" / "process_graph.yaml"


def _best_effort_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py>=3.7
    except Exception:
        pass


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            rows.append((i, json.loads(s)))
    return rows


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def allowed_request_fields_from_schema(schema: Dict[str, Any]) -> Set[str]:
    props = (schema.get("properties") or {})
    if isinstance(props, dict):
        return {str(k) for k in props.keys()}
    return set()


def normalize_op(op: Any) -> str:
    s = "" if op is None else str(op)
    return "equals" if s == "eq" else s


ALLOWED_OPS: Set[str] = {"exists", "equals", "gte", "any_of", "any_true"}


@dataclass
class Issue:
    kind: str  # error | warn
    code: str
    message: str
    where: str


def fmt_issues(issues: Sequence[Issue]) -> str:
    return json.dumps(
        {
            "ok": not any(i.kind == "error" for i in issues),
            "error_count": sum(1 for i in issues if i.kind == "error"),
            "warn_count": sum(1 for i in issues if i.kind == "warn"),
            "issues": [
                {
                    "kind": i.kind,
                    "code": i.code,
                    "where": i.where,
                    "message": i.message,
                }
                for i in issues
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def validate_with_jsonschema(
    *, rows: Sequence[Tuple[int, Dict[str, Any]]], schema: Dict[str, Any], label: str
) -> List[Issue]:
    v = Draft202012Validator(schema)
    issues: List[Issue] = []
    for line_no, obj in rows:
        for err in sorted(v.iter_errors(obj), key=str):
            issues.append(
                Issue(
                    kind="error",
                    code="schema_validation_error",
                    where=f"{label}:L{line_no}",
                    message=err.message,
                )
            )
    return issues


def lint_predicate(
    *, pred: Dict[str, Any], allowed_fields: Set[str], where: str, allow_field_aliases: Set[str]
) -> List[Issue]:
    issues: List[Issue] = []
    op = normalize_op(pred.get("op"))
    field = pred.get("field")

    if op not in ALLOWED_OPS:
        issues.append(
            Issue(
                kind="error",
                code="predicate_unknown_op",
                where=where,
                message=f"Unknown predicate op={pred.get('op')!r}. Allowed ops: {sorted(ALLOWED_OPS)}",
            )
        )
        return issues

    if op in {"exists", "equals", "gte", "any_of", "any_true"}:
        if not isinstance(field, str) or not field.strip():
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_missing_field",
                    where=where,
                    message=f"Predicate op={op} requires a non-empty string field",
                )
            )
        else:
            if field not in allowed_fields and field not in allow_field_aliases:
                issues.append(
                    Issue(
                        kind="error",
                        code="predicate_unknown_field",
                        where=where,
                        message=f"Predicate references unknown request field: {field}",
                    )
                )

    if op in {"equals", "gte"}:
        if "value" not in pred:
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_missing_value",
                    where=where,
                    message=f"Predicate op={op} requires key 'value'",
                )
            )

    if op == "any_of":
        vals = pred.get("values", None)
        if vals is None:
            # evaluator supports "value" as an alias, but we lint hard to avoid ambiguity
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_any_of_missing_values",
                    where=where,
                    message="Predicate op='any_of' must use key 'values' (list) (avoid using 'value')",
                )
            )
        elif not isinstance(vals, list) or len(vals) == 0:
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_any_of_values_not_list",
                    where=where,
                    message="Predicate op='any_of' requires non-empty list in 'values'",
                )
            )

    if op == "any_true":
        vals2 = pred.get("value", None)
        if vals2 is None:
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_any_true_missing_value",
                    where=where,
                    message="Predicate op='any_true' must use key 'value' (list)",
                )
            )
        elif not isinstance(vals2, list) or len(vals2) == 0:
            issues.append(
                Issue(
                    kind="error",
                    code="predicate_any_true_value_not_list",
                    where=where,
                    message="Predicate op='any_true' requires non-empty list in 'value'",
                )
            )

    return issues


def lint_rule_dsl(
    *,
    rule: Dict[str, Any],
    allowed_fields: Set[str],
    known_doc_ids: Set[str],
    where: str,
    allow_field_aliases: Set[str],
) -> List[Issue]:
    issues: List[Issue] = []

    doc_id = rule.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        if doc_id not in known_doc_ids:
            issues.append(
                Issue(
                    kind="error",
                    code="rule_unknown_doc_id",
                    where=where,
                    message=f"doc_id not found in registry: {doc_id}",
                )
            )

    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    if not isinstance(dsl, dict):
        return issues

    required_fields = dsl.get("required_fields") or []
    if required_fields and not isinstance(required_fields, list):
        issues.append(
            Issue(
                kind="error",
                code="dsl_required_fields_not_list",
                where=where,
                message="criteria.dsl.required_fields must be a list of strings",
            )
        )
    else:
        for f in [str(x) for x in required_fields if x is not None]:
            if f not in allowed_fields and f not in allow_field_aliases:
                issues.append(
                    Issue(
                        kind="error",
                        code="dsl_required_field_unknown",
                        where=where,
                        message=f"criteria.dsl.required_fields references unknown request field: {f}",
                    )
                )

    if dsl.get("kind") == "conditional_gate":
        app_fields = dsl.get("applicability_fields") or []
        if not isinstance(app_fields, list) or len(app_fields) == 0:
            issues.append(
                Issue(
                    kind="error",
                    code="dsl_conditional_gate_missing_applicability_fields",
                    where=where,
                    message="conditional_gate requires non-empty criteria.dsl.applicability_fields (list)",
                )
            )
        else:
            for f in [str(x) for x in app_fields if x is not None]:
                if f not in allowed_fields and f not in allow_field_aliases:
                    issues.append(
                        Issue(
                            kind="error",
                            code="dsl_applicability_field_unknown",
                            where=where,
                            message=f"criteria.dsl.applicability_fields references unknown request field: {f}",
                        )
                    )

    preds = dsl.get("predicates") or []
    if preds and not isinstance(preds, list):
        issues.append(
            Issue(
                kind="error",
                code="dsl_predicates_not_list",
                where=where,
                message="criteria.dsl.predicates must be a list of objects",
            )
        )
    else:
        for j, p in enumerate(preds):
            if not isinstance(p, dict):
                issues.append(
                    Issue(
                        kind="error",
                        code="dsl_predicate_not_object",
                        where=f"{where}:predicates[{j}]",
                        message="Predicate must be an object",
                    )
                )
                continue
            issues.extend(
                lint_predicate(
                    pred=p,
                    allowed_fields=allowed_fields,
                    where=f"{where}:predicates[{j}]",
                    allow_field_aliases=allow_field_aliases,
                )
            )

    return issues


def lint_request_fields(
    *, req: Dict[str, Any], allowed_fields: Set[str], where: str, allow_unknown_fields: bool, allow_field_aliases: Set[str]
) -> List[Issue]:
    if allow_unknown_fields:
        return []
    issues: List[Issue] = []
    for k in req.keys():
        if k in allowed_fields:
            continue
        if k in allow_field_aliases:
            continue
        issues.append(
            Issue(
                kind="error",
                code="request_unknown_field",
                where=where,
                message=f"Unknown request field: {k}",
            )
        )
    return issues


def main() -> int:
    _best_effort_utf8_stdout()

    ap = argparse.ArgumentParser(description="Validate requests / rules / graph (schema + lint)")
    ap.add_argument("--request-schema", default=str(DEFAULT_REQUEST_SCHEMA))
    ap.add_argument("--rule-schema", default=str(DEFAULT_RULE_SCHEMA))
    ap.add_argument("--doc-registry", default=str(DEFAULT_DOC_REGISTRY))
    ap.add_argument("--rules", default=str(DEFAULT_RULES_PUBLISHED), help="Rules JSONL to validate")
    ap.add_argument("--requests", default="", help="Requests JSONL to validate (optional)")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH), help="Graph YAML to validate")
    ap.add_argument("--allow-unknown-request-fields", action="store_true", help="Do not fail on extra request keys")
    ap.add_argument("--only", choices=["all", "rules", "requests", "graph"], default="all")
    args = ap.parse_args()

    request_schema_path = Path(args.request_schema)
    rule_schema_path = Path(args.rule_schema)
    doc_registry_path = Path(args.doc_registry)
    rules_path = Path(args.rules)
    graph_path = Path(args.graph)
    requests_path = Path(args.requests) if args.requests else None

    issues: List[Issue] = []

    request_schema = read_json(request_schema_path)
    rule_schema = read_json(rule_schema_path)

    allowed_fields = allowed_request_fields_from_schema(request_schema)
    # Backward compatibility aliases supported by evaluator.
    allow_field_aliases = {"voltage_options"}

    registry = read_json(doc_registry_path)
    known_doc_ids = {str(d.get("doc_id")) for d in (registry or []) if isinstance(d, dict) and d.get("doc_id")}

    # Rules
    if args.only in {"all", "rules"}:
        rule_rows = read_jsonl(rules_path)
        issues.extend(validate_with_jsonschema(rows=rule_rows, schema=rule_schema, label=str(rules_path)))

        # Additional lint: rule_id uniqueness, doc_id existence, DSL field references.
        seen_rule_ids: Set[str] = set()
        for line_no, r in rule_rows:
            rid = r.get("rule_id")
            where = f"{rules_path}:L{line_no}:{rid or 'rule'}"
            if rid:
                rid_s = str(rid)
                if rid_s in seen_rule_ids:
                    issues.append(
                        Issue(
                            kind="error",
                            code="rule_id_duplicate",
                            where=where,
                            message=f"Duplicate rule_id: {rid_s}",
                        )
                    )
                else:
                    seen_rule_ids.add(rid_s)
            issues.extend(
                lint_rule_dsl(
                    rule=r,
                    allowed_fields=allowed_fields,
                    known_doc_ids=known_doc_ids,
                    where=where,
                    allow_field_aliases=allow_field_aliases,
                )
            )

    # Requests
    if args.only in {"all", "requests"} and requests_path is not None:
        req_rows = read_jsonl(requests_path)
        issues.extend(validate_with_jsonschema(rows=req_rows, schema=request_schema, label=str(requests_path)))
        for line_no, req in req_rows:
            where = f"{requests_path}:L{line_no}:{req.get('project_name') or 'request'}"
            issues.extend(
                lint_request_fields(
                    req=req,
                    allowed_fields=allowed_fields,
                    where=where,
                    allow_unknown_fields=bool(args.allow_unknown_request_fields),
                    allow_field_aliases=allow_field_aliases,
                )
            )

    # Graph
    if args.only in {"all", "graph"}:
        graph = load_yaml(graph_path)
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        node_ids = [n.get("id") for n in nodes if isinstance(n, dict)]
        node_id_set = {str(x) for x in node_ids if x}

        if len(node_id_set) != len([x for x in node_ids if x]):
            issues.append(
                Issue(
                    kind="error",
                    code="graph_duplicate_node_id",
                    where=str(graph_path),
                    message="Graph nodes contain duplicate or missing ids",
                )
            )

        edge_ids: Set[str] = set()
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                issues.append(
                    Issue(
                        kind="error",
                        code="graph_edge_not_object",
                        where=f"{graph_path}:edges[{i}]",
                        message="Edge must be an object",
                    )
                )
                continue

            eid = e.get("id")
            if not isinstance(eid, str) or not eid.strip():
                issues.append(
                    Issue(
                        kind="error",
                        code="graph_edge_missing_id",
                        where=f"{graph_path}:edges[{i}]",
                        message="Edge missing id",
                    )
                )
            else:
                if eid in edge_ids:
                    issues.append(
                        Issue(
                            kind="error",
                            code="graph_duplicate_edge_id",
                            where=f"{graph_path}:edges[{i}]:{eid}",
                            message=f"Duplicate edge id: {eid}",
                        )
                    )
                edge_ids.add(eid)

            fr = e.get("from")
            to = e.get("to")
            if fr not in node_id_set:
                issues.append(
                    Issue(
                        kind="error",
                        code="graph_edge_unknown_from",
                        where=f"{graph_path}:edges[{i}]:{eid}",
                        message=f"Edge.from unknown node id: {fr}",
                    )
                )
            if to not in node_id_set:
                issues.append(
                    Issue(
                        kind="error",
                        code="graph_edge_unknown_to",
                        where=f"{graph_path}:edges[{i}]:{eid}",
                        message=f"Edge.to unknown node id: {to}",
                    )
                )

            for j, p in enumerate(e.get("criteria") or []):
                if not isinstance(p, dict):
                    issues.append(
                        Issue(
                            kind="error",
                            code="graph_edge_predicate_not_object",
                            where=f"{graph_path}:edges[{i}]:{eid}:criteria[{j}]",
                            message="Edge criteria predicate must be an object",
                        )
                    )
                    continue
                issues.extend(
                    lint_predicate(
                        pred=p,
                        allowed_fields=allowed_fields,
                        where=f"{graph_path}:edges[{i}]:{eid}:criteria[{j}]",
                        allow_field_aliases=allow_field_aliases,
                    )
                )

        # edge rule_id references must exist in published rules
        published_rule_ids: Set[str] = set()
        if rules_path.exists():
            for _line_no, r in read_jsonl(rules_path):
                rid = r.get("rule_id")
                if rid:
                    published_rule_ids.add(str(rid))

        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                continue
            eid = e.get("id", f"edges[{i}]")
            for rid in (e.get("rule_ids") or []):
                if str(rid) not in published_rule_ids:
                    issues.append(
                        Issue(
                            kind="error",
                            code="graph_edge_unknown_rule_id",
                            where=f"{graph_path}:edges[{i}]:{eid}",
                            message=f"Edge references unknown rule_id (not in {rules_path.name}): {rid}",
                        )
                    )

    print(fmt_issues(issues))
    return 0 if not any(i.kind == "error" for i in issues) else 2


if __name__ == "__main__":
    raise SystemExit(main())

