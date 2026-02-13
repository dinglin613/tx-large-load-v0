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


def render_eval_to_html(tpl: str, ev: Dict[str, Any]) -> str:
    req = ev.get("request") or {}
    graph = (ev.get("graph") or {})
    risk = ev.get("risk") or {}

    path_labels = graph.get("path_node_labels") or []
    traversed = graph.get("traversed_edges") or []

    edges_html = "\n".join(
        f"<li><code>{esc(e.get('edge_id'))}</code>: {esc(e.get('from'))} → {esc(e.get('to'))}"
        f"<br/><small>{esc(' | '.join(e.get('criteria_traces') or []))}</small></li>"
        for e in traversed
    )

    missing_html = "\n".join(f"<li><code>{esc(f)}</code></li>" for f in (ev.get("missing_inputs") or [])) or "<li>None</li>"
    levers_html = "\n".join(
        f"<li><code>{esc(l.get('lever_id'))}</code>: {esc(l.get('note'))}</li>" for l in (ev.get("levers") or [])
    ) or "<li>None detected in request</li>"

    risk_status = risk.get("status", "unknown")
    tb = (risk.get("timeline_buckets") or {})

    evidence = risk.get("evidence") or []
    evidence_html = "\n".join(
        "<li>"
        f"<code>{esc(e.get('rule_id'))}</code>"
        f" — <code>{esc(e.get('doc_id'))}</code>"
        f" — <span class=\"muted\">{esc(e.get('loc'))}</span>"
        f"<br/><small>{esc(', '.join(e.get('trigger_tags') or []))}</small>"
        "</li>"
        for e in evidence
    ) or "<li>None</li>"

    return (
        tpl.replace("{{project_name}}", esc(req.get("project_name")))
        .replace("{{rendered_at}}", esc(datetime.now(timezone.utc).isoformat()))
        .replace("{{request_json}}", esc(json.dumps(req, ensure_ascii=False, indent=2)))
        .replace("{{path}}", esc(" → ".join(path_labels)))
        .replace("{{edges_html}}", edges_html or "<li>None</li>")
        .replace("{{missing_html}}", missing_html)
        .replace("{{levers_html}}", levers_html)
        .replace("{{risk_status}}", esc(risk_status))
        .replace("{{bucket_le_12}}", esc(tb.get("le_12_months", "unknown")))
        .replace("{{bucket_12_24}}", esc(tb.get("m12_24_months", "unknown")))
        .replace("{{bucket_gt_24}}", esc(tb.get("gt_24_months", "unknown")))
        .replace("{{upgrade_exposure}}", esc(risk.get("upgrade_exposure_bucket", "unknown")))
        .replace("{{evidence_html}}", evidence_html)
        .replace("{{graph_version}}", esc(graph.get("graph_version")))
        .replace("{{rules_source}}", esc((ev.get("provenance") or {}).get("rules_source")))
    )


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
