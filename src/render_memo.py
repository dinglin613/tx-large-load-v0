from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from io_jsonl import iter_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "memo" / "templates" / "memo_template.html"


def load_template() -> str:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return f.read()


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def esc_join(xs: List[Any], sep: str = ", ") -> str:
    return esc(sep.join([str(x) for x in xs if x not in (None, "")]))


def code_list(xs: List[Any]) -> str:
    xs2 = [str(x) for x in (xs or []) if str(x).strip()]
    if not xs2:
        return "<span class=\"muted small\">None</span>"
    return " ".join(f"<code>{esc(x)}</code>" for x in xs2)


def chip_list(xs: List[Any]) -> str:
    xs2 = [str(x) for x in (xs or []) if str(x).strip()]
    if not xs2:
        return "<span class=\"muted small\">None</span>"
    return "\n".join(f"<span class=\"chip\"><code>{esc(x)}</code></span>" for x in xs2)


def badge_for_status(status: str) -> str:
    s = (status or "").strip().lower()
    if s == "satisfied":
        cls = "good"
    elif s == "missing":
        cls = "warn"
    elif s == "not_satisfied":
        cls = "bad"
    else:
        cls = ""
    cls_attr = f" badge {cls}".strip()
    return f"<span class=\"{cls_attr}\"><code>{esc(status)}</code></span>"


def render_eval_to_html(tpl: str, ev: Dict[str, Any]) -> str:
    req = ev.get("request") or {}
    graph = (ev.get("graph") or {})
    risk = ev.get("risk") or {}

    path_labels = graph.get("path_node_labels") or []
    traversed = graph.get("traversed_edges") or []

    flags = ev.get("flags") or []

    risk_status = risk.get("status", "unknown")
    tb = (risk.get("timeline_buckets") or {})

    evidence = risk.get("evidence") or []

    edges_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(e.get('edge_id'))}</code></td>"
        f"<td class=\"nowrap\"><code>{esc(e.get('from'))}</code> → <code>{esc(e.get('to'))}</code></td>"
        f"<td class=\"wrap\"><span class=\"small\">{esc(' | '.join(e.get('criteria_traces') or []))}</span></td>"
        f"<td class=\"wrap\">{code_list(e.get('rule_ids') or [])}</td>"
        "</tr>"
        for e in traversed
    ) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>"

    missing_chips = chip_list(ev.get("missing_inputs") or [])

    levers = ev.get("levers") or []
    levers_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(l.get('lever_id'))}</code></td>"
        f"<td class=\"wrap\">{esc(l.get('note'))}</td>"
        "</tr>"
        for l in levers
    ) or "<tr><td colspan=\"2\" class=\"muted\">None detected in request</td></tr>"

    flags_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(f.get('rule_id'))}</code></td>"
        f"<td class=\"nowrap\"><code>{esc(f.get('doc_id'))}</code></td>"
        f"<td class=\"wrap\"><span class=\"muted\">{esc(f.get('loc'))}</span></td>"
        f"<td class=\"wrap\">{code_list(f.get('trigger_tags') or [])}</td>"
        f"<td class=\"wrap\"><span class=\"small\">{esc(' | '.join(f.get('match_traces') or []))}</span></td>"
        "</tr>"
        for f in flags
    ) or "<tr><td colspan=\"5\" class=\"muted\">None</td></tr>"

    checklist_rows = render_checklist_rows(ev.get("energization_checklist") or [])

    evidence_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(e.get('rule_id'))}</code></td>"
        f"<td class=\"nowrap\"><code>{esc(e.get('doc_id'))}</code></td>"
        f"<td class=\"wrap\"><span class=\"muted\">{esc(e.get('loc'))}</span></td>"
        f"<td class=\"wrap\">{code_list(e.get('trigger_tags') or [])}</td>"
        "</tr>"
        for e in evidence
    ) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>"

    return (
        tpl.replace("{{project_name}}", esc(req.get("project_name")))
        .replace("{{rendered_at}}", esc(datetime.now(timezone.utc).isoformat()))
        .replace("{{evaluated_at}}", esc(ev.get("evaluated_at")))
        .replace("{{operator_area}}", esc(req.get("operator_area")))
        .replace("{{tdsp_area}}", esc(req.get("tdsp_area")))
        .replace("{{load_mw_total}}", esc(req.get("load_mw_total")))
        .replace("{{cod_target_window}}", esc(req.get("cod_target_window")))
        .replace("{{is_requesting_energization}}", esc(req.get("is_requesting_energization")))
        .replace("{{request_json}}", esc(json.dumps(req, ensure_ascii=False, indent=2)))
        .replace("{{path}}", esc(" → ".join(path_labels)))
        .replace("{{edges_rows}}", edges_rows)
        .replace("{{missing_chips}}", missing_chips)
        .replace("{{levers_rows}}", levers_rows)
        .replace("{{flags_rows}}", flags_rows)
        .replace("{{checklist_rows}}", checklist_rows)
        .replace("{{risk_status}}", esc(risk_status))
        .replace("{{bucket_le_12}}", esc(tb.get("le_12_months", "unknown")))
        .replace("{{bucket_12_24}}", esc(tb.get("m12_24_months", "unknown")))
        .replace("{{bucket_gt_24}}", esc(tb.get("gt_24_months", "unknown")))
        .replace("{{upgrade_exposure}}", esc(risk.get("upgrade_exposure_bucket", "unknown")))
        .replace("{{operational_exposure}}", esc(risk.get("operational_exposure_bucket", "unknown")))
        .replace("{{evidence_rows}}", evidence_rows)
        .replace("{{graph_version}}", esc(graph.get("graph_version")))
        .replace("{{rules_source}}", esc((ev.get("provenance") or {}).get("rules_source")))
        .replace("{{docs_rows}}", render_docs_rows(ev.get("provenance") or {}))
    )


def render_checklist_rows(items: List[Dict[str, Any]]) -> str:
    rows = [c for c in items if c.get("status") != "not_applicable"]
    if not rows:
        return "<tr><td colspan=\"6\" class=\"muted\">Not requested (set <code>is_requesting_energization=true</code> to evaluate)</td></tr>"

    out: List[str] = []
    for c in rows:
        missing = c.get("missing_fields") or []
        traces = c.get("traces") or []
        out.append(
            "<tr>"
            f"<td class=\"nowrap\">{badge_for_status(str(c.get('status') or 'unknown'))}</td>"
            f"<td class=\"nowrap\"><code>{esc(c.get('rule_id'))}</code></td>"
            f"<td class=\"nowrap\"><code>{esc(c.get('doc_id'))}</code></td>"
            f"<td class=\"wrap\"><span class=\"muted\">{esc(c.get('loc'))}</span></td>"
            f"<td class=\"wrap\">{code_list(missing)}</td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(' | '.join(traces))}</span></td>"
            "</tr>"
        )
    return "\n".join(out)


def render_docs_rows(prov: Dict[str, Any]) -> str:
    docs = prov.get("docs") or []
    if not docs:
        return "<tr><td colspan=\"5\" class=\"muted\">None</td></tr>"

    rows: List[str] = []
    for d in docs:
        doc_id = d.get("doc_id")
        title = d.get("title")
        eff = d.get("effective_date")
        arts = d.get("artifacts") or []
        if not arts:
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(doc_id)}</code></td>"
                f"<td class=\"wrap\">{esc(title)}</td>"
                f"<td class=\"nowrap\"><code>{esc(eff)}</code></td>"
                "<td class=\"muted\">None</td>"
                "<td class=\"muted\">None</td>"
                "</tr>"
            )
            continue

        for i, a in enumerate(arts):
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(doc_id) if i == 0 else ''}</code></td>"
                f"<td class=\"wrap\">{esc(title) if i == 0 else ''}</td>"
                f"<td class=\"nowrap\"><code>{esc(eff) if i == 0 else ''}</code></td>"
                f"<td class=\"wrap\"><code>{esc(a.get('path'))}</code></td>"
                f"<td class=\"wrap\"><code>{esc(a.get('sha256'))}</code></td>"
                "</tr>"
            )

    return "\n".join(rows)


def safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "memo"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render Decision Memo HTML from evaluations JSONL")
    ap.add_argument("--in", dest="in_path", required=True, help="Input evaluations JSONL")
    ap.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory for HTML memos")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl = load_template()

    n = 0
    for ev in iter_jsonl(args.in_path):
        req = ev.get("request") or {}
        name = safe_filename(str(req.get("project_name", f"memo_{n+1}")))
        out_path = out_dir / f"{name}.html"
        out_path.write_text(render_eval_to_html(tpl, ev), encoding="utf-8")
        n += 1

    print(f"Wrote {n} memos to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
