from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from io_jsonl import iter_jsonl
from citation_audit import audit_citations


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "memo" / "templates" / "memo_template.html"
GRAPH_PATH = REPO_ROOT / "graph" / "process_graph.yaml"
REQUEST_SCHEMA_PATH = REPO_ROOT / "schema" / "request.schema.json"


_REQ_FIELD_DESC_CACHE: Dict[str, str] | None = None


def load_request_field_descriptions() -> Dict[str, str]:
    global _REQ_FIELD_DESC_CACHE
    if _REQ_FIELD_DESC_CACHE is not None:
        return _REQ_FIELD_DESC_CACHE
    if not REQUEST_SCHEMA_PATH.exists():
        _REQ_FIELD_DESC_CACHE = {}
        return _REQ_FIELD_DESC_CACHE
    try:
        schema = json.loads(REQUEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception:
        _REQ_FIELD_DESC_CACHE = {}
        return _REQ_FIELD_DESC_CACHE
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    out: Dict[str, str] = {}
    if isinstance(props, dict):
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("description"), str):
                out[str(k)] = v["description"].strip()
    _REQ_FIELD_DESC_CACHE = out
    return out


def field_label(field: str) -> str:
    """
    Return a short human-friendly label for a request field.
    """
    desc = load_request_field_descriptions().get(field, "").strip()
    if not desc:
        return field
    # Take first sentence-ish chunk, keep compact.
    s = desc.split(".")[0].strip()
    # Trim repetitive schema prefixes (UI-only).
    prefixes = [
        "Energization gate input:",
        "Energization readiness input (v0):",
        "Feasibility screening input (v0):",
        "Applicability toggle for",
    ]
    for p in prefixes:
        if s.lower().startswith(p.lower()):
            s = s[len(p) :].strip()
            break
    if len(s) > 92:
        s = s[:92].rstrip() + "…"
    return s


_TRAILING_CITATION_PAREN_RE = re.compile(r"\s*\([^)]*(?:Planning Guide|Section|ERCOT)[^)]*\)\s*\.?\s*$", re.IGNORECASE)


def humanize_requirement_text(text: str) -> str:
    """
    UI-only, deterministic cleanup for requirement text:
    - keep meaning, drop trailing citation parentheticals
    - keep it compact / less legalistic in the "Top next actions" list
    """
    t = " ".join(str(text or "").split()).strip()
    if not t:
        return ""
    t = _TRAILING_CITATION_PAREN_RE.sub("", t).strip()
    if t.endswith("."):
        t = t[:-1].strip()
    # Soften common boilerplate prefixes.
    prefixes = [
        "Prior to initiation of the LLIS process,",
        "As part of Section",
        "As part of",
        "Initial Energization prerequisites may include",
        "Initial energization prerequisites may include",
        "Initial Energization prerequisites include",
        "Initial energization prerequisites include",
    ]
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p) :].lstrip()
            break
    return t


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


def trace_list(xs: List[Any]) -> str:
    """
    Render predicate/criteria traces as wrapping code chips for readability.
    """
    xs2 = [str(x) for x in (xs or []) if str(x).strip()]
    if not xs2:
        return "<span class=\"muted small\">None</span>"
    return "<span class=\"trace-wrap\">" + "".join(f"<code>{esc(x)}</code>" for x in xs2) + "</span>"


def chip_list(xs: List[Any]) -> str:
    xs2 = [str(x) for x in (xs or []) if str(x).strip()]
    if not xs2:
        return "<span class=\"muted small\">None</span>"
    return "\n".join(f"<span class=\"chip\"><code>{esc(x)}</code></span>" for x in xs2)


def rule_ref_details(*, rule_id: Any, doc_id: Any, loc: Any, summary: str = "IDs & citation") -> str:
    rid = str(rule_id or "").strip()
    did = str(doc_id or "").strip()
    l = str(loc or "").strip()
    if not (rid or did or l):
        return "<span class=\"muted small\">—</span>"
    return (
        "<details class=\"mini\">"
        f"<summary>{esc(summary)}</summary>"
        "<div class=\"flip-box\">"
        "<div class=\"muted small\">rule_id</div>"
        f"<div><code>{esc(rid)}</code></div>"
        "<div class=\"muted small\" style=\"margin-top:8px\">doc_id</div>"
        f"<div><code>{esc(did)}</code></div>"
        "<div class=\"muted small\" style=\"margin-top:8px\">loc</div>"
        f"<div class=\"muted\">{esc(l)}</div>"
        "</div>"
        "</details>"
    )


def render_timeline_model_html(ev: Dict[str, Any]) -> str:
    """
    Memo-friendly explanation of v0 timeline signals.
    This is not a probability model; it is a deterministic mapping from rule/tag-driven totals.
    """
    risk = ev.get("risk") or {}

    upgrade_score = 0.0
    wait_score = 0.0
    ops_score = 0.0
    for rt in (ev.get("risk_trace") or []):
        if not isinstance(rt, dict):
            continue
        if rt.get("kind") != "summary":
            continue
        try:
            upgrade_score = float(rt.get("upgrade_score") or 0.0)
            wait_score = float(rt.get("wait_score") or 0.0)
            ops_score = float(rt.get("ops_score") or 0.0)
        except Exception:
            upgrade_score = wait_score = ops_score = 0.0
        break

    tb = risk.get("timeline_buckets") or {}
    le_12 = str(tb.get("le_12_months", "unknown"))
    m12_24 = str(tb.get("m12_24_months", "unknown"))
    gt_24 = str(tb.get("gt_24_months", "unknown"))

    parts = [
        "<div class=\"muted small\">Deterministic mapping (v0; not probabilities)</div>",
        "<ul class=\"small\" style=\"margin:6px 0 0 18px\">"
        "<li><code>upgrade_score ≥ 3</code> ⇒ <code>upgrade_exposure_bucket=high</code>, set <code>&gt;24=up</code>, <code>12–24=up</code>, <code>≤12=down</code></li>"
        "<li><code>0.75 ≤ upgrade_score &lt; 3</code> ⇒ <code>upgrade_exposure_bucket=medium</code>, set <code>12–24=up</code>, <code>≤12=down</code></li>"
        "<li><code>upgrade_score &lt; 0.75</code> ⇒ <code>upgrade_exposure_bucket=low</code> (timeline may remain <code>unknown</code> unless shifted by wait pressure)</li>"
        "<li><code>wait_score ≥ 2</code> ⇒ additional shift away from <code>≤12</code> into <code>12–24</code> (set <code>≤12=down</code>, <code>12–24=up</code>)</li>"
        "</ul>",
        "<div class=\"muted small\" style=\"margin-top:10px\">Observed totals (from triggered rules/tags)</div>",
        "<div class=\"chips\">"
        f"<span class=\"chip\"><span class=\"small\">upgrade_score</span> <code>{esc(upgrade_score)}</code></span>"
        f"<span class=\"chip\"><span class=\"small\">wait_score</span> <code>{esc(wait_score)}</code></span>"
        f"<span class=\"chip\"><span class=\"small\">ops_score</span> <code>{esc(ops_score)}</code></span>"
        "</div>",
        "<div class=\"muted small\" style=\"margin-top:10px\">Resulting timeline signals</div>",
        "<div class=\"chips\">"
        f"<span class=\"chip\"><span class=\"small\">≤12</span> <code>{esc(le_12)}</code></span>"
        f"<span class=\"chip\"><span class=\"small\">12–24</span> <code>{esc(m12_24)}</code></span>"
        f"<span class=\"chip\"><span class=\"small\">&gt;24</span> <code>{esc(gt_24)}</code></span>"
        "</div>",
        "<div class=\"muted small\" style=\"margin-top:10px\">Audit trail</div>",
        "<div class=\"muted small\">See “Risk trace (raw JSON)” for per-rule tag contributions, and “Evidence” for <code>rule_id → doc_id/loc</code>.</div>",
    ]
    return "\n".join(parts)


def render_memo_guidance_html(mg: Any) -> str:
    """
    Render rule-level memo guidance (Owner / Recipient / Evidence / What to do).

    This is intentionally NOT citation-audited; it is human process guidance.
    """
    if not isinstance(mg, dict):
        return ""
    if mg.get("verified") is not True:
        return ""
    owner = str(mg.get("owner") or "").strip()
    recipient = str(mg.get("recipient") or "").strip()
    what = str(mg.get("what_to_do") or "").strip()
    evidence = mg.get("evidence") or []
    ev2 = [str(x).strip() for x in evidence if str(x).strip()] if isinstance(evidence, list) else []
    sources = mg.get("sources") or []
    sources2 = [s for s in sources if isinstance(s, dict) and str(s.get("doc_id") or "").strip() and str(s.get("loc") or "").strip()]

    if (not owner) and (not recipient) and (not what) and (not ev2):
        return ""

    parts: List[str] = []
    if sources2:
        parts.append("<div class=\"muted small\">Guidance sources (docs/raw)</div>")
        chips_src: List[str] = []
        for s in sources2[:6]:
            doc_id = str(s.get("doc_id") or "").strip()
            loc = str(s.get("loc") or "").strip()
            chips_src.append(
                f"<span class=\"chip\"><span class=\"small\">source</span> <code>{esc(doc_id)}</code> <span class=\"muted\">{esc(loc)}</span></span>"
            )
        parts.append("<div class=\"chips\">" + "\n".join(chips_src) + "</div>")
    parts.append("<div class=\"muted small\">Owner / recipient</div>")
    chips: List[str] = []
    if owner:
        chips.append(f"<span class=\"chip\"><span class=\"small\">owner</span> <code>{esc(owner)}</code></span>")
    if recipient:
        chips.append(f"<span class=\"chip\"><span class=\"small\">recipient</span> <code>{esc(recipient)}</code></span>")
    parts.append("<div class=\"chips\">" + ("\n".join(chips) if chips else "<span class=\"muted small\">None</span>") + "</div>")

    if what:
        parts.append("<div class=\"muted small\" style=\"margin-top:8px\">What to do</div>")
        parts.append(f"<div class=\"small\"><span class=\"muted\">{esc(what)}</span></div>")

    parts.append("<div class=\"muted small\" style=\"margin-top:8px\">Evidence (completion proof)</div>")
    if ev2:
        lis = "".join(f"<li class=\"small\"><span class=\"muted\">{esc(x)}</span></li>" for x in ev2[:12])
        more = f"<div class=\"muted small\" style=\"margin-top:6px\">(+{len(ev2)-12} more)</div>" if len(ev2) > 12 else ""
        parts.append("<ul style=\"margin:6px 0 0 18px\">" + lis + "</ul>" + more)
    else:
        parts.append("<div class=\"muted small\">None</div>")

    return "\n".join(parts)


def render_uncertainty_html(ev: Dict[str, Any]) -> str:
    u = ev.get("uncertainties") or {}
    missing = [x for x in (u.get("missing") or []) if isinstance(x, dict)]
    unknown = [x for x in (u.get("unknown") or []) if isinstance(x, dict)]
    assumptions = [x for x in (u.get("assumptions") or []) if isinstance(x, dict)]
    op_disc = [x for x in (u.get("operator_discretion") or []) if isinstance(x, dict)]
    queue = [x for x in (u.get("queue_state_dependent") or []) if isinstance(x, dict)]

    reason_counts = Counter([str(m.get("reason") or "") for m in missing])
    # Stable display order (compact)
    reason_order = [
        ("absent", "absent"),
        ("null", "explicit null"),
        ("empty_string", "empty string"),
        ("empty_list", "empty list"),
        ("false_as_missing", "false-as-missing"),
        ("empty_list_as_missing", "empty-list-as-missing"),
        ("unknown", "other"),
    ]

    chips: List[str] = []
    total_missing = len(missing)
    total_unknown = len(unknown)
    chips.append(f"<span class=\"chip warn\"><span class=\"small\">missing</span> <code>{total_missing}</code></span>")
    if total_unknown:
        chips.append(f"<span class=\"chip\"><span class=\"small\">unknown (explicit)</span> <code>{total_unknown}</code></span>")
    if assumptions:
        chips.append(f"<span class=\"chip\"><span class=\"small\">assumptions</span> <code>{len(assumptions)}</code></span>")
    if op_disc:
        chips.append(f"<span class=\"chip\"><span class=\"small\">operator discretion</span> <code>{len(op_disc)}</code></span>")
    if queue:
        chips.append(f"<span class=\"chip\"><span class=\"small\">queue-state dependent</span> <code>{len(queue)}</code></span>")

    parts: List[str] = []
    parts.append("<div class=\"chips\">" + "\n".join(chips) + "</div>")

    # Missing breakdown (counts only; full list is in “Missing inputs” section).
    if total_missing:
        breakdown = []
        for key, label in reason_order:
            n = int(reason_counts.get(key, 0))
            if n:
                breakdown.append(f"<span class=\"chip\"><span class=\"small\">{esc(label)}</span> <code>{n}</code></span>")
        if breakdown:
            parts.append("<div class=\"muted small\" style=\"margin-top:10px\">Missing breakdown</div>")
            parts.append("<div class=\"chips\">" + "\n".join(breakdown) + "</div>")

    # Unknown inputs (explicit markers)
    if total_unknown:
        fields = [str(x.get("field") or "") for x in unknown if str(x.get("field") or "").strip()]
        parts.append("<details style=\"margin-top:10px\"><summary>Show explicit unknown inputs</summary>")
        parts.append("<div class=\"hint muted small\">These fields were provided but marked as unknown (e.g., <code>null</code> or <code>\"unknown\"</code>).</div>")
        parts.append("<div class=\"chips\" style=\"margin-top:8px\">" + chip_list(sorted(set(fields))) + "</div>")
        parts.append("</details>")

    # Assumptions (non-cited heuristics)
    if assumptions:
        parts.append("<details style=\"margin-top:10px\"><summary>Show non-cited assumptions (heuristics)</summary>")
        parts.append("<div class=\"hint muted small\">Heuristics are explicitly labeled and contribute to v0 risk buckets; they are not backed by PDF citations.</div>")
        rows = []
        items = sorted(assumptions, key=lambda x: str(x.get("rule_id") or ""))
        for a in items:
            rid = a.get("rule_id")
            loc = a.get("loc")
            notes = a.get("notes") or []
            note1 = notes[0] if isinstance(notes, list) and notes else ""
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(rid)}</code></td>"
                f"<td class=\"wrap\"><span class=\"muted\">{esc(loc)}</span></td>"
                f"<td class=\"wrap\"><span class=\"small\">{esc(note1)}</span></td>"
                "</tr>"
            )
        parts.append("<div style=\"overflow-x:auto;margin-top:8px\"><table>")
        parts.append("<thead><tr><th class=\"nowrap\">rule_id</th><th>loc</th><th>note</th></tr></thead>")
        parts.append("<tbody>" + ("\n".join(rows) or "<tr><td colspan=\"3\" class=\"muted\">None</td></tr>") + "</tbody>")
        parts.append("</table></div></details>")

    # Operator discretion signals
    if op_disc:
        parts.append("<details style=\"margin-top:10px\"><summary>Show operator discretion signals</summary>")
        parts.append("<div class=\"hint muted small\">These signals indicate engineering / operator judgement may materially affect outcomes.</div>")
        rows = []
        items = sorted(op_disc, key=lambda x: (str(x.get("doc_id") or ""), str(x.get("rule_id") or "")))
        for it in items:
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(it.get('rule_id'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(it.get('doc_id'))}</code></td>"
                f"<td class=\"wrap\"><span class=\"muted\">{esc(it.get('loc'))}</span></td>"
                f"<td class=\"wrap\">{code_list(it.get('trigger_tags') or [])}</td>"
                "</tr>"
            )
        parts.append("<div style=\"overflow-x:auto;margin-top:8px\"><table>")
        parts.append("<thead><tr><th class=\"nowrap\">rule_id</th><th class=\"nowrap\">doc_id</th><th>loc</th><th>tags</th></tr></thead>")
        parts.append("<tbody>" + ("\n".join(rows) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>") + "</tbody>")
        parts.append("</table></div></details>")

    # Queue-state dependent signals
    if queue:
        parts.append("<details style=\"margin-top:10px\"><summary>Show queue-state dependent signals</summary>")
        parts.append("<div class=\"hint muted small\">These signals depend on queue density / dependencies; timelines can change as other projects progress.</div>")
        rows = []
        items = sorted(queue, key=lambda x: (str(x.get("doc_id") or ""), str(x.get("rule_id") or "")))
        for it in items:
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(it.get('rule_id'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(it.get('doc_id'))}</code></td>"
                f"<td class=\"wrap\"><span class=\"muted\">{esc(it.get('loc'))}</span></td>"
                f"<td class=\"wrap\">{code_list(it.get('trigger_tags') or [])}</td>"
                "</tr>"
            )
        parts.append("<div style=\"overflow-x:auto;margin-top:8px\"><table>")
        parts.append("<thead><tr><th class=\"nowrap\">rule_id</th><th class=\"nowrap\">doc_id</th><th>loc</th><th>tags</th></tr></thead>")
        parts.append("<tbody>" + ("\n".join(rows) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>") + "</tbody>")
        parts.append("</table></div></details>")

    if (not total_missing) and (not total_unknown) and (not assumptions) and (not op_disc) and (not queue):
        return "<div class=\"muted small\">None detected</div>"
    return "\n".join(parts)


def render_path_graph(ev: Dict[str, Any]) -> str:
    """
    Render a lightweight “graph view” of the selected path.

    This is intentionally dependency-free (single-file HTML) and complements
    the edge audit log table.
    """
    graph = ev.get("graph") or {}
    labels = graph.get("path_node_labels") or []
    nodes = graph.get("path_nodes") or []
    edges = graph.get("traversed_edges") or []

    if not labels:
        return "<div class=\"muted small\">(no path)</div>"

    parts: List[str] = ["<div class=\"path-graph\" aria-label=\"selected process path\">"]
    for i, label in enumerate(labels):
        node_id = nodes[i] if i < len(nodes) else ""
        parts.append(
            "<div class=\"pnode\">"
            f"<div class=\"t\">{esc(label)}</div>"
            f"<div class=\"s\"><code>{esc(node_id)}</code></div>"
            "</div>"
        )
        if i < len(labels) - 1:
            eid = ""
            if i < len(edges):
                eid = str(edges[i].get("edge_id") or "")
            parts.append(
                "<div class=\"pedge\">"
                f"<div class=\"eid\"><code>{esc(eid)}</code></div>"
                "<div class=\"arr\">→</div>"
                "</div>"
            )
    parts.append("</div>")
    return "\n".join(parts)


def _wrap_svg_text(s: str, *, max_chars: int = 22, max_lines: int = 3) -> List[str]:
    """
    Simple deterministic word-wrap for SVG <text>.
    """
    s = (s or "").strip()
    if not s:
        return [""]
    words = s.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    # If truncated, add ellipsis to the last line.
    if len(words) > 0 and len(lines) == max_lines:
        joined = " ".join(lines)
        if joined != s and not lines[-1].endswith("…"):
            lines[-1] = (lines[-1][: max(0, max_chars - 1)] + "…") if len(lines[-1]) >= max_chars else (lines[-1] + "…")
    return lines


def render_full_process_graph_svg(ev: Dict[str, Any]) -> str:
    """
    Render the entire process graph from graph/process_graph.yaml and highlight
    the selected path from the evaluation.
    """
    if not GRAPH_PATH.exists():
        return "<div class=\"muted small\">(graph/process_graph.yaml not found)</div>"

    with GRAPH_PATH.open("r", encoding="utf-8") as f:
        graph = yaml.safe_load(f) or {}

    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict) and e.get("id") and e.get("from") and e.get("to")]
    if not nodes:
        return "<div class=\"muted small\">(no nodes in graph)</div>"

    nodes_by_id = {str(n["id"]): n for n in nodes}
    node_ids = list(nodes_by_id.keys())

    # Highlight sets from evaluation output.
    path_nodes = [str(x) for x in ((ev.get("graph") or {}).get("path_nodes") or []) if x]
    path_edges = [str(x.get("edge_id")) for x in (((ev.get("graph") or {}).get("traversed_edges") or [])) if isinstance(x, dict) and x.get("edge_id")]
    path_node_set = set(path_nodes)
    path_edge_set = set(path_edges)

    # Build adjacency + in-degrees for a simple layered layout.
    out_edges: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in node_ids}
    in_deg: Dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        a = str(e.get("from"))
        b = str(e.get("to"))
        if a not in nodes_by_id or b not in nodes_by_id:
            continue
        out_edges.setdefault(a, []).append(e)
        in_deg[b] = int(in_deg.get(b, 0)) + 1

    # Deterministic topological order (Kahn), then longest-distance-from-start ranks.
    start = "N0" if "N0" in nodes_by_id else min(node_ids)
    q = sorted([nid for nid in node_ids if in_deg.get(nid, 0) == 0])
    topo: List[str] = []
    in_deg2 = dict(in_deg)
    while q:
        u = q.pop(0)
        topo.append(u)
        for e in out_edges.get(u, []):
            v = str(e.get("to"))
            if v not in in_deg2:
                continue
            in_deg2[v] -= 1
            if in_deg2[v] == 0:
                q.append(v)
                q.sort()
    # Fallback: include any nodes not reached by Kahn (cycles/unexpected).
    for nid in sorted(node_ids):
        if nid not in topo:
            topo.append(nid)

    dist: Dict[str, int] = {nid: -10_000 for nid in node_ids}
    dist[start] = 0
    for u in topo:
        du = dist.get(u, -10_000)
        if du < -1_000:
            continue
        for e in out_edges.get(u, []):
            v = str(e.get("to"))
            if v not in dist:
                continue
            dist[v] = max(dist[v], du + 1)

    # Assign layers (rank). Unreachable nodes go to the last layer.
    max_dist = max([d for d in dist.values() if d > -1_000] or [0])
    layer_of: Dict[str, int] = {}
    for nid in node_ids:
        d = dist.get(nid, -10_000)
        layer_of[nid] = d if d > -1_000 else (max_dist + 1)

    layers: Dict[int, List[str]] = {}
    for nid, lay in layer_of.items():
        layers.setdefault(int(lay), []).append(nid)
    for lay in layers:
        layers[lay].sort()

    layer_keys = sorted(layers.keys())
    max_nodes_per_layer = max(len(layers[k]) for k in layer_keys) if layer_keys else 1

    # Geometry
    margin_x = 20
    margin_y = 16
    # Increase node box + spacing to reduce visual crowding/overlap.
    node_w = 220
    node_h = 74
    col_gap = 90
    row_gap = 22

    inner_h = max_nodes_per_layer * node_h + max(0, max_nodes_per_layer - 1) * row_gap
    inner_w = len(layer_keys) * node_w + max(0, len(layer_keys) - 1) * col_gap
    svg_w = margin_x * 2 + inner_w
    svg_h = margin_y * 2 + inner_h + 36  # + legend

    # Place nodes, vertically centered per layer.
    pos: Dict[str, Dict[str, float]] = {}
    for i, lay in enumerate(layer_keys):
        ids = layers[lay]
        layer_h = len(ids) * node_h + max(0, len(ids) - 1) * row_gap
        y0 = margin_y + (inner_h - layer_h) / 2.0
        x = margin_x + i * (node_w + col_gap)
        for j, nid in enumerate(ids):
            y = y0 + j * (node_h + row_gap)
            pos[nid] = {"x": x, "y": y}

    def _node_center(nid: str) -> tuple[float, float]:
        p = pos[nid]
        return (float(p["x"]) + node_w / 2.0, float(p["y"]) + node_h / 2.0)

    def _node_right(nid: str) -> tuple[float, float]:
        p = pos[nid]
        return (float(p["x"]) + node_w, float(p["y"]) + node_h / 2.0)

    def _node_left(nid: str) -> tuple[float, float]:
        p = pos[nid]
        return (float(p["x"]), float(p["y"]) + node_h / 2.0)

    # Colors
    edge_norm = "#9ca3af"
    edge_sel = "#2563eb"
    node_border = "#d1d5db"
    node_sel_border = "#2563eb"
    node_fill = "#ffffff"
    # IMPORTANT: keep this opaque so background edges do not show “through” selected nodes.
    node_sel_fill = "#eff6ff"
    text_main = "#111827"
    text_muted = "#6b7280"

    # Simple routing tracks (outside node boxes) for long edges, to avoid
    # “crossing through” intermediate nodes.
    track_top_y = 8.0
    track_bottom_y = margin_y + inner_h + 8.0
    center_y = margin_y + inner_h / 2.0
    elbow_pad = 12.0

    def _esc(x: Any) -> str:
        return html.escape("" if x is None else str(x))

    def _rect_for(nid: str) -> tuple[float, float, float, float]:
        p = pos[nid]
        x = float(p["x"])
        y = float(p["y"])
        return (x, y, x + node_w, y + node_h)

    def _point_in_rect(x: float, y: float, r: tuple[float, float, float, float]) -> bool:
        return (r[0] <= x <= r[2]) and (r[1] <= y <= r[3])

    def _adjust_label_xy(lx: float, ly: float, *, a: str, b: str) -> tuple[float, float]:
        """
        Keep edge labels outside the from/to node boxes when possible.
        """
        ra = _rect_for(a)
        rb = _rect_for(b)
        if _point_in_rect(lx, ly, ra) or _point_in_rect(lx, ly, rb):
            # Prefer to move upward; if that goes inside the other box, move downward.
            ly2 = ly - (node_h / 2.0 + 10.0)
            if (not _point_in_rect(lx, ly2, ra)) and (not _point_in_rect(lx, ly2, rb)) and (ly2 > 10):
                return lx, ly2
            ly3 = ly + (node_h / 2.0 + 10.0)
            return lx, ly3
        return lx, ly

    svg: List[str] = []
    svg.append(f"<svg class=\"graph\" width=\"{svg_w:g}\" height=\"{svg_h:g}\" viewBox=\"0 0 {svg_w:g} {svg_h:g}\" role=\"img\" aria-label=\"process graph\">")
    svg.append("<defs>")
    svg.append(
        "<marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#9ca3af\"></path>"
        "</marker>"
    )
    svg.append(
        "<marker id=\"arrowSel\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto-start-reverse\">"
        "<path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#2563eb\"></path>"
        "</marker>"
    )
    svg.append("</defs>")

    # Edges first (under nodes). Draw non-selected edges first, then the selected path edges,
    # so the highlighted path remains visually on top (even when curves cross).
    edges_draw: List[Dict[str, Any]] = []
    for e in edges:
        eid = str(e.get("id") or "")
        if eid and eid not in path_edge_set:
            edges_draw.append(e)
    for e in edges:
        eid = str(e.get("id") or "")
        if eid and eid in path_edge_set:
            edges_draw.append(e)

    for e in edges_draw:
        a = str(e.get("from"))
        b = str(e.get("to"))
        eid = str(e.get("id"))
        if a not in pos or b not in pos:
            continue

        x1, y1 = _node_right(a)
        x2, y2 = _node_left(b)
        a_layer = int(layer_of.get(a, 0))
        b_layer = int(layer_of.get(b, 0))
        span = abs(b_layer - a_layer)

        is_sel = eid in path_edge_set
        stroke = edge_sel if is_sel else edge_norm
        # Keep ALL edges solid to avoid implying a different “edge type”.
        # Use opacity/width to de-emphasize non-selected edges.
        sw = 3 if is_sel else 1.6
        stroke_opacity = "1" if is_sel else "0.45"
        marker = "url(#arrowSel)" if is_sel else "url(#arrow)"
        title = f"{eid}: {a} -> {b}"

        # Routing:
        # - Short edges: simple cubic curve (compact).
        # - Long edges (span >= 2 layers): route via top/bottom track to avoid
        #   visually passing through intermediate node boxes (e.g., E2X_END).
        label_xy: Optional[tuple[float, float]] = None

        if span >= 2:
            track_y = track_top_y if ((y1 + y2) / 2.0) < center_y else track_bottom_y
            x1o = x1 + elbow_pad
            x2i = x2 - elbow_pad
            d = (
                f"M {x1:g} {y1:g} "
                f"L {x1o:g} {y1:g} "
                f"L {x1o:g} {track_y:g} "
                f"L {x2i:g} {track_y:g} "
                f"L {x2i:g} {y2:g} "
                f"L {x2:g} {y2:g}"
            )
            svg.append(
                f"<path d=\"{d}\" fill=\"none\" stroke=\"{stroke}\" stroke-opacity=\"{stroke_opacity}\" "
                f"stroke-width=\"{sw:g}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"{marker}\">"
                f"<title>{_esc(title)}</title>"
                "</path>"
            )
            if is_sel:
                # Place label near the long horizontal track segment.
                lx = (x1o + x2i) / 2.0
                ly = (track_y + 14.0) if track_y == track_top_y else (track_y - 6.0)
                label_xy = _adjust_label_xy(lx, ly, a=a, b=b)
        else:
            # Adjacent-layer (or same-layer) edges: route via the inter-column “corridor”
            # using an orthogonal polyline. This avoids visually crossing node boxes.
            if abs(y1 - y2) < 0.5:
                d = f"M {x1:g} {y1:g} L {x2:g} {y2:g}"
            else:
                mx = (x1 + x2) / 2.0
                x1o = x1 + elbow_pad
                x2i = x2 - elbow_pad
                d = (
                    f"M {x1:g} {y1:g} "
                    f"L {x1o:g} {y1:g} "
                    f"L {mx:g} {y1:g} "
                    f"L {mx:g} {y2:g} "
                    f"L {x2i:g} {y2:g} "
                    f"L {x2:g} {y2:g}"
                )
            svg.append(
                f"<path d=\"{d}\" fill=\"none\" stroke=\"{stroke}\" stroke-opacity=\"{stroke_opacity}\" "
                f"stroke-width=\"{sw:g}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" marker-end=\"{marker}\">"
                f"<title>{_esc(title)}</title>"
                "</path>"
            )
            if is_sel:
                lx = (x1 + x2) / 2.0
                ly = (y1 + y2) / 2.0 - (10.0 if abs(y1 - y2) < 0.5 else 8.0)
                label_xy = _adjust_label_xy(lx, ly, a=a, b=b)

        # Labels are appended AFTER nodes so they are never covered by node boxes.
        # Collect them here.
        if is_sel and label_xy is not None:
            if "edge_labels" not in locals():
                edge_labels: List[str] = []
            lx, ly = label_xy
            edge_labels.append(
                f"<text x=\"{lx:g}\" y=\"{ly:g}\" text-anchor=\"middle\" font-size=\"11\" fill=\"{text_muted}\" "
                f"style=\"paint-order:stroke;stroke:#ffffff;stroke-width:6;stroke-linejoin:round\">"
                f"<tspan>{_esc(eid)}</tspan></text>"
            )

    # Nodes
    for nid in node_ids:
        if nid not in pos:
            continue
        p = pos[nid]
        x = float(p["x"])
        y = float(p["y"])
        label = str(nodes_by_id[nid].get("label") or nid)
        is_sel = nid in path_node_set
        stroke = node_sel_border if is_sel else node_border
        fill = node_sel_fill if is_sel else node_fill
        sw = 2.5 if is_sel else 1.2

        svg.append(f"<rect x=\"{x:g}\" y=\"{y:g}\" width=\"{node_w:g}\" height=\"{node_h:g}\" rx=\"12\" ry=\"12\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"{sw:g}\"></rect>")
        svg.append(f"<text x=\"{(x + 12):g}\" y=\"{(y + 22):g}\" font-size=\"12\" fill=\"{text_main}\">")
        for i, line in enumerate(_wrap_svg_text(label, max_chars=24, max_lines=3)):
            dy = 0 if i == 0 else 14
            svg.append(f"<tspan x=\"{(x + 12):g}\" dy=\"{dy:g}\">{_esc(line)}</tspan>")
        svg.append("</text>")
        svg.append(f"<text x=\"{(x + 12):g}\" y=\"{(y + node_h - 12):g}\" font-size=\"11\" fill=\"{text_muted}\"><tspan>{_esc(nid)}</tspan></text>")

    # Edge labels on top (avoid being covered by node boxes)
    if "edge_labels" in locals():
        svg.extend(edge_labels)

    # Legend
    leg_y = margin_y + inner_h + 22
    svg.append(f"<g transform=\"translate({margin_x:g},{leg_y:g})\">")
    svg.append(f"<rect x=\"0\" y=\"-12\" width=\"12\" height=\"12\" rx=\"3\" fill=\"{node_sel_fill}\" stroke=\"{node_sel_border}\" stroke-width=\"2\"></rect>")
    svg.append(f"<text x=\"18\" y=\"-2\" font-size=\"12\" fill=\"{text_muted}\">selected path node</text>")
    svg.append(f"<line x1=\"170\" y1=\"-6\" x2=\"210\" y2=\"-6\" stroke=\"{edge_sel}\" stroke-width=\"3\" marker-end=\"url(#arrowSel)\"></line>")
    svg.append(f"<text x=\"218\" y=\"-2\" font-size=\"12\" fill=\"{text_muted}\">selected path edge</text>")
    svg.append("</g>")

    svg.append("</svg>")
    return "\n".join(svg)


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
    label = status
    if s == "unknown":
        label = "Unknown (confirm applicability)"
    cls_attr = f" badge {cls}".strip()
    return f"<span class=\"{cls_attr}\"><code>{esc(label)}</code></span>"


def render_decision_status_badge(ev: Dict[str, Any]) -> str:
    def _chip(label: str, status: str) -> str:
        s = (status or "").strip().lower()
        if s in {"go", "ready"}:
            cls = "good"
        elif s in {"conditional"}:
            cls = "warn"
        elif s in {"no_go", "not_ready"}:
            cls = "bad"
        else:
            cls = ""
        cls_attr = f"chip {cls}".strip()
        return f"<span class=\"{cls_attr}\"><span class=\"small\">{esc(label)}</span> <code>{esc(status)}</code></span>"

    ds = ev.get("decision_screening")
    de = ev.get("decision_energization")
    if isinstance(ds, dict) or isinstance(de, dict):
        ds2 = ds if isinstance(ds, dict) else (ev.get("decision") or {})
        de2 = de if isinstance(de, dict) else {}
        scr = _chip("screening", str(ds2.get("status") or "unknown"))
        en = _chip("energization", str(de2.get("status") or "unknown")) if isinstance(de2, dict) and de2 else _chip("energization", "unknown")
        return "\n".join([scr, en])

    # Fallback: legacy single decision
    d = ev.get("decision") or {}
    return _chip(str(d.get("mode") or "decision"), str(d.get("status") or "unknown"))


def render_decision_reason_chips(ev: Dict[str, Any]) -> str:
    def _chips(label: str, d: Any) -> List[str]:
        if not isinstance(d, dict):
            return []
        rs = d.get("reasons") or []
        if not isinstance(rs, list) or not rs:
            return []
        rs2 = [str(x) for x in rs if str(x).strip()]
        if not rs2:
            return []
        rs2 = rs2[:4]
        return [f"<span class=\"chip\"><span class=\"small\">{esc(label)}</span> <code>{esc(r)}</code></span>" for r in rs2]

    ds = ev.get("decision_screening")
    de = ev.get("decision_energization")
    if isinstance(ds, dict) or isinstance(de, dict):
        out = _chips("screening", ds) + _chips("energization", de)
        return "\n".join(out)

    # Fallback: legacy single decision
    d = ev.get("decision") or {}
    out = _chips("reason", d)
    return "\n".join(out)


def render_next_actions_html(ev: Dict[str, Any]) -> str:
    items = ev.get("next_actions") or []
    if not isinstance(items, list) or not items:
        return "<div class=\"muted small\">Next actions: None</div>"

    # Index missing field -> sources (edge criteria / rule required_fields)
    miss_meta = {}
    u = ev.get("uncertainties") or {}
    for m in (u.get("missing") or []):
        if not isinstance(m, dict):
            continue
        f = str(m.get("field") or "")
        if not f:
            continue
        miss_meta[f] = m

    def _render_sources_for_field(field: str) -> str:
        m = miss_meta.get(field) or {}
        src = m.get("sources") or []
        if not isinstance(src, list) or not src:
            return "<div class=\"muted small\">No sources</div>"
        edge_ids = sorted({str(s.get("edge_id")) for s in src if isinstance(s, dict) and s.get("kind") == "edge_criteria" and s.get("edge_id")})
        rule_ids = sorted({str(s.get("rule_id")) for s in src if isinstance(s, dict) and s.get("kind") == "rule_required_fields" and s.get("rule_id")})
        parts = []
        if edge_ids:
            parts.append("<div class=\"muted small\">From edge criteria</div>")
            parts.append("<div class=\"chips\">" + chip_list(edge_ids) + "</div>")
        if rule_ids:
            parts.append("<div class=\"muted small\" style=\"margin-top:8px\">From rule required_fields</div>")
            parts.append("<div class=\"chips\">" + chip_list(rule_ids[:16]) + ("<span class=\"muted small\">…</span>" if len(rule_ids) > 16 else "") + "</div>")
        return "\n".join(parts) if parts else "<div class=\"muted small\">No sources</div>"

    def _action_chip(label: str, value: Any) -> str:
        v = "" if value is None else str(value)
        if not v.strip():
            return ""
        return f"<span class=\"chip\"><span class=\"small\">{esc(label)}</span> <code>{esc(v.strip())}</code></span>"

    def _owner_recipient_chips(mg: Any) -> str:
        if not isinstance(mg, dict):
            return "<span class=\"muted small\">None</span>"
        owner = str(mg.get("owner") or "").strip()
        recipient = str(mg.get("recipient") or "").strip()
        chips = []
        if owner:
            chips.append(_action_chip("owner", owner))
        if recipient:
            chips.append(_action_chip("recipient", recipient))
        if not chips:
            return "<span class=\"muted small\">None</span>"
        return "<div class=\"chips\">" + "\n".join([c for c in chips if c]) + "</div>"

    def _done_evidence_preview(mg: Any, *, max_items: int = 3) -> str:
        if not isinstance(mg, dict):
            return "<span class=\"muted small\">None</span>"
        evs = mg.get("evidence") or []
        if not isinstance(evs, list):
            return "<span class=\"muted small\">None</span>"
        xs = [str(x).strip() for x in evs if str(x).strip()]
        if not xs:
            return "<span class=\"muted small\">None</span>"
        xs = xs[: max(0, int(max_items))]
        lis = "".join(f"<li class=\"small\"><span class=\"muted\">{esc(x)}</span></li>" for x in xs)
        return "<ul class=\"done-list\" style=\"margin:6px 0 0 18px\">" + lis + "</ul>"

    def _blocked_by_chips(fields: Any) -> str:
        fs = [str(x) for x in (fields or []) if str(x).strip() and str(x) != "is_requesting_energization"]
        if not fs:
            return "<span class=\"muted small\">None</span>"
        chips = []
        for f in fs[:8]:
            chips.append(
                f"<span class=\"chip\"><span class=\"small\">field</span> <span class=\"muted\">{esc(field_label(f))}</span> <code>{esc(f)}</code></span>"
            )
        return "<div class=\"chips\">" + "\n".join(chips) + "</div>"

    def _guidance_sources_chips(mg: Any) -> str:
        if not isinstance(mg, dict):
            return "<div class=\"muted small\">None</div>"
        src = mg.get("sources") or []
        if not isinstance(src, list):
            return "<div class=\"muted small\">None</div>"
        src2 = []
        for s in src:
            if not isinstance(s, dict):
                continue
            doc = str(s.get("doc_id") or "").strip()
            loc = str(s.get("loc") or "").strip()
            if not doc or not loc:
                continue
            src2.append((doc, loc))
        if not src2:
            return "<div class=\"muted small\">None</div>"
        chips = [
            f"<span class=\"chip\"><span class=\"small\">source</span> <code>{esc(doc)}</code> <span class=\"muted\">{esc(loc)}</span></span>"
            for doc, loc in src2[:8]
        ]
        return "<div class=\"chips\">" + "\n".join(chips) + "</div>"

    out: List[str] = []
    out.append("<div class=\"muted small\">Top next actions</div>")
    out.append("<div class=\"action-cards\" style=\"margin-top:8px\">")

    for i, a in enumerate(items[:10], start=1):
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind") or "")

        # Card header chips
        head_chips: List[str] = []
        head_chips.append(_action_chip("rank", i))
        if kind:
            head_chips.append(_action_chip("type", kind))

        if kind == "provide_field":
            f = str(a.get("field") or "").strip()
            if not f:
                continue
            lbl = field_label(f)
            title = f"Provide: {lbl}"

            drawer = (
                "<details class=\"action-details\">"
                "<summary>Evidence & audit details</summary>"
                f"<div class=\"flip-box\">{_render_sources_for_field(f)}</div>"
                "</details>"
            )

            out.append("<div class=\"action-card\">")
            out.append("<div class=\"action-head\">")
            out.append(f"<div class=\"action-title\">{esc(title)} <code>{esc(f)}</code></div>")
            out.append("<div class=\"chips\">" + "\n".join([c for c in head_chips if c]) + "</div>")
            out.append("</div>")
            out.append("<div class=\"action-grid\">")
            out.append("<div class=\"action-k\">Done = evidence</div>")
            out.append("<div class=\"action-v\"><span class=\"muted small\">Field is provided (non-empty / non-null) in the request snapshot.</span></div>")
            out.append("</div>")
            out.append(drawer)
            out.append("</div>")
            continue

        if kind == "energization_gate":
            rid = str(a.get("rule_id") or "").strip()
            doc_id = str(a.get("doc_id") or "").strip()
            loc = str(a.get("loc") or "").strip()
            status = str(a.get("status") or "").strip().lower()
            crit = str(a.get("criteria_text") or "").strip()
            cond = a.get("conditions") or []
            ask = a.get("ask_fields") or []
            mg = a.get("memo_guidance") if isinstance(a.get("memo_guidance"), dict) else None

            # Title: prefer plain-language memo guidance
            what = str((mg or {}).get("what_to_do") or "").strip() if isinstance(mg, dict) else ""
            if what:
                title = what
            else:
                # fallback
                fields_for_title = [str(x) for x in (ask or (a.get("required_fields") or [])) if str(x).strip() and str(x) != "is_requesting_energization"]
                if fields_for_title:
                    lbls = [field_label(f) for f in fields_for_title[:4]]
                    title = "Provide/confirm: " + "; ".join(lbls)
                else:
                    title = humanize_requirement_text(crit) or (crit or f"Gate requirement ({rid})")

            if status:
                head_chips.append(_action_chip("status", status))

            # Evidence drawer (audit/engineering details)
            drawer_parts: List[str] = []
            drawer_parts.append("<div class=\"muted small\">Guidance sources</div>")
            drawer_parts.append(_guidance_sources_chips(mg))
            drawer_parts.append("<div class=\"muted small\" style=\"margin-top:10px\">rule</div>")
            chips_rule = [
                _action_chip("rule_id", rid),
                (_action_chip("doc_id", doc_id) if doc_id else ""),
                (_action_chip("status", status) if status else ""),
            ]
            drawer_parts.append("<div class=\"chips\">" + "\n".join([c for c in chips_rule if c]) + "</div>")
            if loc:
                drawer_parts.append("<div class=\"muted small\" style=\"margin-top:10px\">loc</div>")
                drawer_parts.append(f"<div class=\"small\"><span class=\"muted\">{esc(loc)}</span></div>")
            if crit:
                drawer_parts.append("<div class=\"muted small\" style=\"margin-top:10px\">source text</div>")
                drawer_parts.append(f"<div class=\"small\"><span class=\"muted\">{esc(crit)}</span></div>")
            if cond:
                drawer_parts.append("<div class=\"muted small\" style=\"margin-top:10px\">predicate traces</div>")
                drawer_parts.append(f"<div class=\"small\">{trace_list(cond)}</div>")

            why = "energization gate (missing / not_satisfied)"
            if status == "unknown":
                why = "energization gate (unknown applicability)"

            drawer_parts.append("<div class=\"muted small\" style=\"margin-top:10px\">why in top actions</div>")
            drawer_parts.append(f"<div class=\"small\"><span class=\"muted\">{esc(why)}</span></div>")

            drawer = (
                "<details class=\"action-details\">"
                "<summary>Evidence & audit details</summary>"
                "<div class=\"flip-box\">"
                + "\n".join(drawer_parts)
                + "</div>"
                "</details>"
            )

            out.append("<div class=\"action-card\">")
            out.append("<div class=\"action-head\">")
            out.append(f"<div class=\"action-title\">{esc(title)}</div>")
            out.append("<div class=\"chips\">" + "\n".join([c for c in head_chips if c]) + "</div>")
            out.append("</div>")

            out.append("<div class=\"action-grid\">")
            out.append("<div class=\"action-k\">Owner → recipient</div>")
            out.append(f"<div class=\"action-v\">{_owner_recipient_chips(mg)}</div>")
            out.append("<div class=\"action-k\">Done = evidence</div>")
            out.append(f"<div class=\"action-v\">{_done_evidence_preview(mg, max_items=3)}</div>")
            out.append("<div class=\"action-k\">Blocked by</div>")
            out.append(f"<div class=\"action-v\">{_blocked_by_chips(ask)}</div>")
            out.append("</div>")

            out.append(drawer)
            out.append("</div>")
            continue

        # Fallback card
        out.append("<div class=\"action-card\">")
        out.append("<div class=\"action-head\">")
        out.append(f"<div class=\"action-title\"><code>{esc(kind or 'action')}</code></div>")
        out.append("<div class=\"chips\">" + "\n".join([c for c in head_chips if c]) + "</div>")
        out.append("</div></div>")

    out.append("</div>")
    return "\n".join(out)


def render_recommendation_html(ev: Dict[str, Any]) -> str:
    rec = ev.get("recommendation") or {}
    if not isinstance(rec, dict) or not rec:
        return "<div class=\"muted small\">None</div>"
    rid = rec.get("recommended_option_id")
    is_base = bool(rec.get("recommended_is_baseline"))
    rationale = rec.get("rationale") or []
    ranked = rec.get("candidates_ranked") or []
    basis = rec.get("basis") or {}

    parts: List[str] = []
    parts.append("<div class=\"chips\">")
    parts.append(f"<span class=\"chip good\"><span class=\"small\">recommended</span> <code>{esc(rid)}</code></span>")
    parts.append(f"<span class=\"chip\"><span class=\"small\">baseline?</span> <code>{esc(str(is_base))}</code></span>")
    parts.append("</div>")
    # Make the scoring rule explicit (v0, bounded, no probabilities).
    parts.append("<div class=\"muted small\" style=\"margin-top:10px\">Scoring rule (v0, explainable)</div>")
    parts.append(
        "<div class=\"small\"><code>score = (missing_inputs_count, energization_penalty, upgrade_bucket_rank, ops_bucket_rank, risk_pressure, timeline_score)</code></div>"
    )
    parts.append(
        "<div class=\"muted small\" style=\"margin-top:6px\">Where <code>energization_penalty = 2*not_satisfied + missing</code> (only when energization is requested), "
        "bucket ranks are <code>low&lt;medium&lt;high&lt;unknown</code>, "
        "<code>risk_pressure = upgrade_score + wait_score + ops_score</code> (tie-breaker from v0 tag scoring), "
        "and <code>timeline_score</code> is derived from the qualitative timeline buckets.</div>"
    )
    ctx_note = basis.get("context_note")
    if isinstance(ctx_note, str) and ctx_note.strip():
        parts.append(f"<div class=\"muted small\" style=\"margin-top:6px\">Context: {esc(ctx_note.strip())}</div>")
    if isinstance(rationale, list) and rationale:
        parts.append("<div class=\"muted small\" style=\"margin-top:10px\">Rationale</div>")
        parts.append("<ul style=\"margin:6px 0 0 18px\">")
        for r in rationale[:6]:
            parts.append(f"<li class=\"small\"><code>{esc(r)}</code></li>")
        parts.append("</ul>")
    if isinstance(ranked, list) and ranked:
        parts.append("<div class=\"muted small\" style=\"margin-top:10px\">Top ranked candidates (v0)</div>")
        rows = []
        for c in ranked[:6]:
            if not isinstance(c, dict):
                continue
            s = c.get("summary") or {}
            tb = s.get("timeline_buckets") or {}
            rs = s.get("risk_scores") or {}
            en = s.get("energization") or {}
            cc = (en.get("checklist_counts") or {}) if isinstance(en, dict) else {}
            rows.append(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(c.get('option_id'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(c.get('lever_id'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(c.get('source') or '')}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(s.get('missing_inputs_count'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(s.get('upgrade_exposure_bucket'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(s.get('operational_exposure_bucket'))}</code></td>"
                f"<td class=\"nowrap\"><span class=\"small\"><code>risk:{esc((rs.get('upgrade_score',0) or 0) + (rs.get('wait_score',0) or 0) + (rs.get('ops_score',0) or 0))}</code></span></td>"
                f"<td class=\"nowrap\"><span class=\"small\">≤12:{esc(tb.get('le_12_months','?'))} 12–24:{esc(tb.get('m12_24_months','?'))} >24:{esc(tb.get('gt_24_months','?'))}</span></td>"
                f"<td class=\"nowrap\"><span class=\"small\">gate not_sat:{esc(cc.get('not_satisfied',0))} missing:{esc(cc.get('missing',0))}</span></td>"
                "</tr>"
            )
        parts.append("<div style=\"overflow-x:auto;margin-top:8px\"><table>")
        parts.append("<thead><tr><th class=\"nowrap\">option</th><th class=\"nowrap\">lever</th><th class=\"nowrap\">source</th><th class=\"nowrap\">missing</th><th class=\"nowrap\">upgrade</th><th class=\"nowrap\">ops</th><th class=\"nowrap\">risk</th><th class=\"nowrap\">timeline</th><th class=\"nowrap\">gates</th></tr></thead>")
        parts.append("<tbody>" + ("\n".join(rows) or "<tr><td colspan=\"9\" class=\"muted\">None</td></tr>") + "</tbody>")
        parts.append("</table></div>")
    parts.append("<div class=\"muted small\" style=\"margin-top:10px\">Note: v0 recommendation is bounded to baseline + generated options; it is not a global optimizer.</div>")
    return "\n".join(parts)


def _humanize_field_id(fid: Any) -> str:
    s = str(fid or "").strip()
    if not s:
        return ""
    acr = {"ercot", "tdsp", "tsp", "llis", "ille", "nom", "qsa", "sso", "dme", "poi", "cod"}
    words = []
    for w in s.replace("_", " ").split():
        wl = w.lower()
        if wl in acr:
            words.append(wl.upper())
        else:
            words.append(w[:1].upper() + w[1:])
    return " ".join(words)


def _deliverable_hint(field: str) -> str:
    m = {
        "one_line_diagram": "One-line diagram (PDF)",
        "load_projection_5y": "5-year load projection (spreadsheet/PDF)",
        "llis_formal_request_submitted": "Formal LLIS initiation request + confirmation",
        "llis_data_package_submitted": "Study data package submission receipt (files + transmittal)",
        "ack_change_notification_obligation": "Signed acknowledgement letter/email",
        "phases": "Phasing schedule table (MW increments + dates) in Load Commissioning Plan",
        "telemetry_operational_and_accurate": "Telemetry commissioning evidence (test results + signoff)",
        "nom_included": "NOM inclusion confirmation (ERCOT notice/screenshot)",
        "agreements_executed": "Executed agreements + financial security receipt",
    }
    return m.get(field, "Supporting evidence artifact (as applicable)")


def render_executive_brief_html(ev: Dict[str, Any]) -> str:
    """
    Non-technical brief: forwardable, minimal jargon/IDs. Uses existing eval fields only.
    """
    req = ev.get("request") or {}
    dec_s = ev.get("decision_screening") or {}
    dec_e = ev.get("decision_energization") or {}

    screening_status = str(dec_s.get("status") or "unknown")
    e_reasons = " ".join(str(x) for x in (dec_e.get("reasons") or []) if str(x).strip()).lower()
    energ_note = "not requested" if ("energization_not_requested" in e_reasons or "not_requested" in e_reasons) else str(dec_e.get("status") or "unknown")

    miss = [str(x) for x in (ev.get("missing_inputs") or []) if str(x).strip()]
    miss_set = set(miss)

    missing_reqs: List[Dict[str, Any]] = []
    for rc in (ev.get("rule_checks") or []):
        if not isinstance(rc, dict):
            continue
        if str(rc.get("status") or "") != "missing":
            continue
        mf = [str(x) for x in (rc.get("missing_fields") or []) if str(x).strip()]
        hit = [f for f in mf if f in miss_set]
        if not hit:
            continue
        crit = str(rc.get("criteria_text") or "").strip()
        for f in hit:
            missing_reqs.append({"field": f, "requirement": crit})
        if len(missing_reqs) >= 12:
            break

    seen = set()
    missing_reqs2: List[Dict[str, Any]] = []
    for it in missing_reqs:
        f = str(it.get("field") or "")
        if not f or f in seen:
            continue
        seen.add(f)
        missing_reqs2.append(it)
    if not missing_reqs2:
        missing_reqs2 = [{"field": f, "requirement": ""} for f in miss[:8]]

    # Recommendation (bounded)
    rec = ev.get("recommendation") or {}
    rid = str(rec.get("recommended_option_id") or "baseline")
    is_base = bool(rec.get("recommended_is_baseline"))
    rationale = rec.get("rationale") or []
    r1 = str(rationale[0]) if isinstance(rationale, list) and rationale else ""
    reco_line = f"Keep baseline ({rid})" if is_base else f"Prefer option {rid}"

    # Risk line (signals, not probabilities)
    risk = ev.get("risk") or {}
    tb = risk.get("timeline_buckets") or {}
    risk_line = (
        f"Timeline signals: <=12={tb.get('le_12_months','unknown')}, "
        f"12-24={tb.get('m12_24_months','unknown')}, "
        f">24={tb.get('gt_24_months','unknown')}; "
        f"upgrade={risk.get('upgrade_exposure_bucket','unknown')}, "
        f"ops={risk.get('operational_exposure_bucket','unknown')}."
    )

    drivers = []
    for d in (ev.get("top_drivers") or []):
        if not isinstance(d, dict):
            continue
        s = str(d.get("summary") or "").strip() or str(d.get("rule_id") or "").strip()
        if s:
            drivers.append(s)
        if len(drivers) >= 3:
            break

    snap = (
        f"Operator={req.get('operator_area','?')}, TDSP={req.get('tdsp_area','?')}, "
        f"Load={req.get('load_mw_total','?')}MW, COD target={req.get('cod_target_window','n/a')}."
    )

    blocking_items = []
    for it in missing_reqs2[:3]:
        f = str(it.get("field") or "").strip()
        reqtxt = str(it.get("requirement") or "").strip()
        headline = _humanize_field_id(f) or f
        note = (f"<div class=\"muted small\">{esc(reqtxt)}</div>" if reqtxt else "")
        blocking_items.append("<li><strong>" + esc(headline) + "</strong>" + note + "</li>")
    blocking_html = "<ul>" + ("".join(blocking_items) if blocking_items else "<li class=\"muted small\">None</li>") + "</ul>"

    action_rows = []
    for it in missing_reqs2[:8]:
        f = str(it.get("field") or "").strip()
        reqtxt = str(it.get("requirement") or "").strip()
        action_rows.append(
            "<tr>"
            f"<td class=\"wrap\"><strong>{esc(_humanize_field_id(f) or f)}</strong><div class=\"muted small\">{esc(reqtxt) if reqtxt else ''}</div></td>"
            f"<td class=\"nowrap\"><code>Customer</code></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(_deliverable_hint(f))}</span></td>"
            "</tr>"
        )
    actions_table = (
        "<div style=\"overflow-x:auto;margin-top:8px\">"
        "<table>"
        "<thead><tr><th>Blocking now - deliverable</th><th class=\"nowrap\">owner</th><th>what to produce</th></tr></thead>"
        "<tbody>"
        + ("".join(action_rows) if action_rows else "<tr><td colspan=\"3\" class=\"muted\">None</td></tr>")
        + "</tbody></table></div>"
    )

    parts = []
    parts.append(f"<div class=\"muted small\">Project snapshot: <code>{esc(req.get('project_name'))}</code> — {esc(snap)}</div>")
    parts.append("<div style=\"margin-top:10px\" class=\"kv\">")
    parts.append("<div class=\"k\">1) Current conclusion</div>")
    parts.append(f"<div class=\"v\"><strong>Screening:</strong> {esc(screening_status)}; <strong>Energization:</strong> {esc(energ_note)}</div>")
    parts.append("<div class=\"k\">2) Blocking now</div>")
    parts.append(f"<div class=\"v\">{blocking_html}</div>")
    parts.append("<div class=\"k\">3) Next actions</div>")
    parts.append("<div class=\"v\">Complete the blocking deliverables below (owners + concrete outputs).</div>")
    parts.append("<div class=\"k\">4) Config options & recommendation (bounded)</div>")
    parts.append(f"<div class=\"v\"><strong>{esc(reco_line)}</strong><div class=\"muted small\">{esc(r1) if r1 else ''}</div></div>")
    parts.append("<div class=\"k\">5) Major risks & external uncertainty</div>")
    parts.append(
        "<div class=\"v\">"
        + esc(risk_line)
        + (("<div class=\"muted small\" style=\"margin-top:6px\">Top drivers: " + esc("; ".join(drivers)) + "</div>") if drivers else "")
        + "<div class=\"muted small\" style=\"margin-top:6px\">Final requirements may depend on ERCOT/TSP/TDSP discretion, queue/system state, and site verification. Full evidence is available below.</div>"
        + "</div>"
    )
    parts.append("</div>")
    parts.append("<div style=\"margin-top:10px\"><strong>Blocking-now checklist (owner + deliverable)</strong></div>")
    parts.append(actions_table)
    return "\n".join(parts)


def render_top_drivers_html(ev: Dict[str, Any]) -> str:
    items = ev.get("top_drivers") or []
    if not isinstance(items, list) or not items:
        return "<div class=\"muted small\">None</div>"
    rows: List[str] = []
    for it in items[:8]:
        if not isinstance(it, dict):
            continue
        c = it.get("contributions") or {}
        rows.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(it.get('rule_id'))}</code></td>"
            f"<td class=\"nowrap\"><code>{esc(it.get('doc_id'))}</code></td>"
            f"<td class=\"wrap\"><span class=\"muted\">{esc(it.get('loc'))}</span></td>"
            f"<td class=\"wrap\">{code_list(it.get('trigger_tags') or [])}</td>"
            f"<td class=\"nowrap\"><span class=\"small\"><code>up:{esc(c.get('upgrade',0))}</code> <code>wait:{esc(c.get('wait',0))}</code> <code>ops:{esc(c.get('ops',0))}</code></span></td>"
            "</tr>"
        )
    return (
        "<div style=\"overflow-x:auto;margin-top:8px\"><table>"
        "<thead><tr><th class=\"nowrap\">rule_id</th><th class=\"nowrap\">doc_id</th><th>loc</th><th>tags</th><th class=\"nowrap\">contrib</th></tr></thead>"
        "<tbody>"
        + ("\n".join(rows) or "<tr><td colspan=\"5\" class=\"muted\">None</td></tr>")
        + "</tbody></table></div>"
    )

def _count_status(items: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"satisfied": 0, "missing": 0, "not_satisfied": 0, "unknown": 0, "not_applicable": 0}
    for it in items or []:
        s = str(it.get("status") or "").strip().lower()
        if not s:
            out["unknown"] += 1
        elif s in out:
            out[s] += 1
        else:
            out["unknown"] += 1
    return out


def _chips_for_status_counts(counts: Dict[str, int], *, include_total: bool = True) -> str:
    sat = int(counts.get("satisfied", 0))
    miss = int(counts.get("missing", 0))
    ns = int(counts.get("not_satisfied", 0))
    unk = int(counts.get("unknown", 0))
    na = int(counts.get("not_applicable", 0))

    parts: List[str] = []
    parts.append(f"<span class=\"chip good\"><span class=\"small\">satisfied</span> <code>{sat}</code></span>")
    parts.append(f"<span class=\"chip warn\"><span class=\"small\">missing</span> <code>{miss}</code></span>")
    parts.append(f"<span class=\"chip bad\"><span class=\"small\">not_satisfied</span> <code>{ns}</code></span>")
    if unk:
        parts.append(f"<span class=\"chip\"><span class=\"small\">unknown</span> <code>{unk}</code></span>")
    if na:
        parts.append(f"<span class=\"chip\"><span class=\"small\">not_applicable</span> <code>{na}</code></span>")
    if include_total:
        total = sat + miss + ns + unk + na
        parts.append(f"<span class=\"chip\"><span class=\"small\">total</span> <code>{total}</code></span>")
    return "\n".join(parts)


def render_checklist_stats(items: List[Dict[str, Any]]) -> str:
    # Hide not_applicable to align with displayed checklist rows.
    rows = [c for c in (items or []) if c.get("status") != "not_applicable"]
    if not rows:
        return "<span class=\"muted small\">Not requested (no checklist rows)</span>"
    counts = _count_status(rows)
    return _chips_for_status_counts(counts, include_total=True)


def render_evidence_stats(evidence: List[Dict[str, Any]], rule_checks: List[Dict[str, Any]]) -> tuple[str, int]:
    """
    Evidence list is rule_ids + heuristics. We compute tri-state stats where possible
    by joining evidence rule_id -> rule_checks.status.
    """
    ev_items = evidence or []
    total = len(ev_items)
    checks_by_id: Dict[str, Dict[str, Any]] = {}
    for c in rule_checks or []:
        rid = str(c.get("rule_id") or "")
        if rid:
            checks_by_id[rid] = c

    tri: List[Dict[str, Any]] = []
    heur = 0
    for e in ev_items:
        rid = str(e.get("rule_id") or "")
        doc = str(e.get("doc_id") or "")
        if doc == "NON_CITED_HEURISTIC":
            heur += 1
            continue
        c = checks_by_id.get(rid)
        if c:
            tri.append(c)

    if not ev_items:
        return "<span class=\"muted small\">None</span>", 0

    counts = _count_status(tri)
    chips = _chips_for_status_counts(counts, include_total=False)
    extra = f"<span class=\"chip\"><span class=\"small\">heuristics</span> <code>{heur}</code></span>" if heur else ""
    extra2 = f"<span class=\"chip\"><span class=\"small\">evidence items</span> <code>{total}</code></span>"
    html_out = "\n".join([chips, extra, extra2]).strip()
    return html_out, total


def render_evidence_grouped_by_doc(evidence: List[Dict[str, Any]]) -> str:
    ev_items = evidence or []
    if not ev_items:
        return "<div class=\"muted small\">None</div>"
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in ev_items:
        doc = str(e.get("doc_id") or "UNKNOWN_DOC")
        groups.setdefault(doc, []).append(e)
    out: List[str] = []
    for doc_id in sorted(groups.keys()):
        items = groups[doc_id]
        # deterministic sort: rule_id then loc
        items2 = sorted(items, key=lambda x: (str(x.get("rule_id") or ""), str(x.get("loc") or "")))
        out.append(
            "<details style=\"margin-top:8px\">"
            f"<summary><code>{esc(doc_id)}</code> <span class=\"muted small\">({len(items2)})</span></summary>"
            "<div style=\"overflow-x:auto;margin-top:8px\">"
            "<table>"
            "<thead><tr><th class=\"nowrap\">rule_id</th><th>loc</th><th>tags</th></tr></thead>"
            "<tbody>"
            + "\n".join(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(it.get('rule_id'))}</code></td>"
                f"<td class=\"wrap\"><span class=\"muted\">{esc(it.get('loc'))}</span></td>"
                f"<td class=\"wrap\">{code_list(it.get('trigger_tags') or [])}</td>"
                "</tr>"
                for it in items2
            )
            + "</tbody></table></div></details>"
        )
    return "\n".join(out)


def render_context_snapshot_html(ev: Dict[str, Any]) -> str:
    ctx = ev.get("context_snapshot")
    if not isinstance(ctx, dict):
        # Fallback for older eval outputs.
        req = ev.get("request") or {}
        ctx = req.get("context")
    if not isinstance(ctx, dict):
        return "<div class=\"muted small\">None provided</div>"
    sigs = ctx.get("signals") or []
    if not isinstance(sigs, list) or not sigs:
        return "<div class=\"muted small\">None provided</div>"

    as_of = str(ctx.get("as_of") or "").strip()

    # Index context contributions from risk_trace (kind=context) by CTX_{signal_id}
    contrib_by_id: Dict[str, Dict[str, Any]] = {}
    for rt in (ev.get("risk_trace") or []):
        if not isinstance(rt, dict):
            continue
        if rt.get("kind") != "context":
            continue
        rid = str(rt.get("rule_id") or "")
        if rid:
            contrib_by_id[rid] = rt

    def _fmt_contrib(c: Any) -> str:
        if not isinstance(c, dict):
            return "<span class=\"muted small\">n/a</span>"
        try:
            up = float(c.get("upgrade") or 0.0)
            wt = float(c.get("wait") or 0.0)
            op = float(c.get("ops") or 0.0)
        except Exception:
            up = wt = op = 0.0
        parts = [f"up:{up:+g}", f"wait:{wt:+g}", f"ops:{op:+g}"]
        return "<span class=\"small\"><code>" + "</code> <code>".join(esc(p) for p in parts) + "</code></span>"

    rows: List[str] = []
    for s in sigs:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("signal_id") or "").strip()
        typ = str(s.get("type") or "").strip()
        val = s.get("value")
        conf = s.get("confidence")
        src = str(s.get("source") or "").strip()
        notes = str(s.get("notes") or "").strip()
        rid = f"CTX_{sid}" if sid else ""
        rt = contrib_by_id.get(rid) or {}
        contrib = rt.get("contributions")

        src_cell = esc(src) if src else '<span class="muted small">n/a</span>'
        notes_cell = esc(notes) if notes else '<span class="muted small">n/a</span>'

        rows.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(sid)}</code></td>"
            f"<td class=\"nowrap\"><code>{esc(typ)}</code></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val)}</span></td>"
            f"<td class=\"nowrap\"><code>{esc(conf)}</code></td>"
            f"<td class=\"wrap\"><span class=\"small\">{src_cell}</span></td>"
            f"<td class=\"wrap\"><span class=\"small\">{notes_cell}</span></td>"
            f"<td class=\"nowrap\">{_fmt_contrib(contrib)}</td>"
            "</tr>"
        )

    head = ""
    if as_of:
        head = f"<div class=\"chips\"><span class=\"chip\"><span class=\"small\">as_of</span> <code>{esc(as_of)}</code></span></div>"
    table = (
        "<div style=\"overflow-x:auto;margin-top:10px\">"
        "<table>"
        "<thead><tr>"
        "<th class=\"nowrap\">signal_id</th>"
        "<th class=\"nowrap\">type</th>"
        "<th>value</th>"
        "<th class=\"nowrap\">confidence</th>"
        "<th>source</th>"
        "<th>notes</th>"
        "<th class=\"nowrap\">v0 effect</th>"
        "</tr></thead>"
        "<tbody>"
        + ("\n".join(rows) if rows else "<tr><td colspan=\"7\" class=\"muted\">None</td></tr>")
        + "</tbody></table></div>"
    )
    return head + table


def render_evidence_grouped_by_tag(evidence: List[Dict[str, Any]]) -> str:
    ev_items = evidence or []
    if not ev_items:
        return "<div class=\"muted small\">None</div>"
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in ev_items:
        tags = e.get("trigger_tags") or []
        if not tags:
            groups.setdefault("(no_tag)", []).append(e)
            continue
        for t in sorted({str(x) for x in tags if str(x).strip()}):
            groups.setdefault(t, []).append(e)

    out: List[str] = []
    for tag in sorted(groups.keys()):
        items = groups[tag]
        items2 = sorted(items, key=lambda x: (str(x.get("doc_id") or ""), str(x.get("rule_id") or "")))
        out.append(
            "<details style=\"margin-top:8px\">"
            f"<summary><code>{esc(tag)}</code> <span class=\"muted small\">({len(items2)})</span></summary>"
            "<div style=\"overflow-x:auto;margin-top:8px\">"
            "<table>"
            "<thead><tr><th class=\"nowrap\">doc_id</th><th class=\"nowrap\">rule_id</th><th>loc</th></tr></thead>"
            "<tbody>"
            + "\n".join(
                "<tr>"
                f"<td class=\"nowrap\"><code>{esc(it.get('doc_id'))}</code></td>"
                f"<td class=\"nowrap\"><code>{esc(it.get('rule_id'))}</code></td>"
                f"<td class=\"wrap\"><span class=\"muted\">{esc(it.get('loc'))}</span></td>"
                "</tr>"
                for it in items2
            )
            + "</tbody></table></div></details>"
        )
    return "\n".join(out)


def render_levers_catalog_html(ev: Dict[str, Any]) -> str:
    items = ev.get("levers_catalog_analysis") or []
    if not isinstance(items, list) or not items:
        return "<div class=\"muted small\">None</div>"

    def _lever_class_badge(cls: Any) -> str:
        c = str(cls or "unknown").strip().lower()
        if c not in {"hard", "design", "external", "unknown"}:
            c = "unknown"
        # Keep neutral styling; avoid implying good/bad.
        if c == "design":
            return f"<span class=\"chip info\"><span class=\"small\">class</span> <code>{esc(c)}</code></span>"
        if c == "external":
            return f"<span class=\"chip warn\"><span class=\"small\">class</span> <code>{esc(c)}</code></span>"
        if c == "hard":
            return f"<span class=\"chip\"><span class=\"small\">class</span> <code>{esc(c)}</code></span>"
        return f"<span class=\"chip\"><span class=\"small\">class</span> <code>{esc(c)}</code></span>"

    rows: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        lever_id = it.get("lever_id")
        lever_class = it.get("lever_class")
        label = it.get("label") or ""
        present = it.get("present_fields") or []
        missing = it.get("missing_fields") or []
        n = it.get("referenced_rule_count") or 0
        rids = it.get("referenced_rule_ids") or []
        rids_preview = [str(x) for x in rids[:8]]
        more_html = ' <span class="muted small">…</span>' if len(rids) > len(rids_preview) else ""
        rows.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(lever_id)}</code></td>"
            f"<td class=\"nowrap\">{_lever_class_badge(lever_class)}</td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(label)}</span></td>"
            f"<td class=\"wrap\">{code_list(present)}</td>"
            f"<td class=\"wrap\">{code_list(missing)}</td>"
            f"<td class=\"nowrap\"><code>{esc(n)}</code></td>"
            f"<td class=\"wrap\">{code_list(rids_preview)}{more_html}</td>"
            "</tr>"
        )

    if not rows:
        return "<div class=\"muted small\">None</div>"

    return (
        "<div style=\"overflow-x:auto;margin-top:8px\">"
        "<table>"
        "<thead><tr>"
        "<th class=\"nowrap\">lever_id</th>"
        "<th class=\"nowrap\">class</th>"
        "<th>label</th>"
        "<th>present fields</th>"
        "<th>missing fields</th>"
        "<th class=\"nowrap\">rule refs</th>"
        "<th>example rule_ids</th>"
        "</tr></thead>"
        "<tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def render_lever_next_actions_html(ev: Dict[str, Any], *, top_n: int = 5) -> str:
    """
    Produce a bounded, memo-friendly guide for "which levers to decide next":
    - What to ask (missing/unknown lever fields)
    - What we observed (option delta signals)
    - Which rule checks are currently blocked (missing) by those fields
    """
    lever_items = ev.get("levers_catalog_analysis") or []
    if not isinstance(lever_items, list) or not lever_items:
        return "<div class=\"muted small\">None</div>"

    options = ev.get("options") or []
    if not isinstance(options, list):
        options = []

    u = ev.get("uncertainties") or {}
    missing = [x for x in (u.get("missing") or []) if isinstance(x, dict)]
    unknown = [x for x in (u.get("unknown") or []) if isinstance(x, dict)]
    unknown_fields = {str(x.get("field") or "") for x in unknown if str(x.get("field") or "").strip()}

    rule_checks = ev.get("rule_checks") or []
    if not isinstance(rule_checks, list):
        rule_checks = []

    # Index: lever_id -> option deltas
    opt_by_lever: Dict[str, List[Dict[str, Any]]] = {}
    for o in options:
        if not isinstance(o, dict):
            continue
        lid = str(o.get("lever_id") or "")
        if not lid:
            continue
        opt_by_lever.setdefault(lid, []).append(o)

    # Index missing sources: field -> rule_ids that require it
    missing_rule_sources: Dict[str, set[str]] = {}
    for m in missing:
        f = str(m.get("field") or "")
        if not f:
            continue
        for s in (m.get("sources") or []):
            if not isinstance(s, dict):
                continue
            if s.get("kind") == "rule_required_fields" and s.get("rule_id"):
                missing_rule_sources.setdefault(f, set()).add(str(s.get("rule_id")))

    def _fmt_value(v: Any) -> str:
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        return esc(v)

    def _fmt_predicate(p: Dict[str, Any]) -> str:
        op = str(p.get("op") or "")
        if op == "eq":
            op = "equals"
        field = str(p.get("field") or "")
        if op == "exists":
            return f"provide {field}"
        if op == "equals":
            return f"{field} == {_fmt_value(p.get('value'))}"
        if op == "gte":
            return f"{field} >= {_fmt_value(p.get('value'))}"
        if op == "any_of":
            vals = p.get("values", None)
            if vals is None:
                vals = p.get("value", None)
            return f"{field} in {_fmt_value(vals)}"
        if op == "any_true":
            return f"{field} contains any of {_fmt_value(p.get('value'))}"
        return f"{op}({field})"

    def _lever_related_conditions(chk: Dict[str, Any], lever_fields: List[str]) -> List[str]:
        lf = set(lever_fields or [])
        preds = chk.get("predicates") or []
        out: List[str] = []
        if isinstance(preds, list):
            for p in preds:
                if not isinstance(p, dict):
                    continue
                f = str(p.get("field") or "")
                if f and f in lf:
                    out.append(_fmt_predicate(p))
        # Also include required_fields that are lever fields (if no predicate mentions them).
        rf = [str(x) for x in (chk.get("required_fields") or []) if str(x).strip()]
        for f in rf:
            if f in lf and not any(s.startswith(f + " ") or s.endswith(" " + f) or (f + " ==") in s for s in out):
                out.append(f"provide {f}")
        # Stable + dedupe
        seen = set()
        out2 = []
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            out2.append(s)
        return out2

    def _score(it: Dict[str, Any]) -> float:
        lever_id = str(it.get("lever_id") or "")
        missing_fields = [str(x) for x in (it.get("missing_fields") or []) if str(x).strip()]
        present_fields = [str(x) for x in (it.get("present_fields") or []) if str(x).strip()]
        opt = opt_by_lever.get(lever_id, [])

        s = 0.0
        # If options show a path change, it's a very high-signal decision lever.
        any_path_change = False
        any_bucket_change = False
        for o in opt:
            d = o.get("delta") or {}
            if d.get("path_changed"):
                any_path_change = True
            up = d.get("upgrade_exposure_bucket_changed") or {}
            ops = d.get("operational_exposure_bucket_changed") or {}
            if str(up.get("from")) != str(up.get("to")) or str(ops.get("from")) != str(ops.get("to")):
                any_bucket_change = True
        if any_path_change:
            s += 6.0
        if any_bucket_change:
            s += 2.5
        # Missing/unknown lever fields => actionable next questions.
        s += 1.0 * len(missing_fields)
        s += 0.75 * len([f for f in present_fields if f in unknown_fields])
        # If many rules reference the lever, deciding it will unlock more explanation.
        try:
            s += min(2.0, float(it.get("referenced_rule_count") or 0) / 5.0)
        except Exception:
            pass
        return s

    # Build per-lever rows
    enriched = []
    for it in lever_items:
        if not isinstance(it, dict):
            continue
        lever_id = str(it.get("lever_id") or "")
        if not lever_id:
            continue
        req_fields = [str(x) for x in (it.get("request_fields") or []) if str(x).strip()]
        missing_fields = [str(x) for x in (it.get("missing_fields") or []) if str(x).strip()]
        unknown_lever_fields = sorted({f for f in req_fields if f in unknown_fields})

        lf_set = set(req_fields)

        # For memo: show which rule-check rows are likely to become "satisfied"
        # once lever-related inputs are provided / changed.
        missing_to_satisfied: List[Dict[str, Any]] = []
        not_satisfied_to_satisfied: List[Dict[str, Any]] = []

        for c in rule_checks:
            if not isinstance(c, dict):
                continue
            status = str(c.get("status") or "").strip().lower()
            rid = str(c.get("rule_id") or "")
            if not rid:
                continue

            missing_f = {str(x) for x in (c.get("missing_fields") or []) if str(x).strip()}
            fields_ref = {str(x) for x in (c.get("fields_referenced") or []) if str(x).strip()}

            # missing -> satisfied (if the missing lever fields are provided and predicates are met)
            if status == "missing" and (missing_f & lf_set):
                cond = _lever_related_conditions(c, req_fields)
                missing_to_satisfied.append(
                    {
                        "rule_id": rid,
                        "doc_id": c.get("doc_id"),
                        "loc": c.get("loc"),
                        "conditions": cond,
                        "missing_fields": sorted(missing_f & lf_set),
                    }
                )

            # not_satisfied -> satisfied (if lever-related predicate conditions are met)
            if status == "not_satisfied" and (fields_ref & lf_set):
                cond = _lever_related_conditions(c, req_fields)
                if cond:
                    not_satisfied_to_satisfied.append(
                        {
                            "rule_id": rid,
                            "doc_id": c.get("doc_id"),
                            "loc": c.get("loc"),
                            "conditions": cond,
                        }
                    )

        # Option delta summary for this lever.
        opt = opt_by_lever.get(lever_id, [])
        any_path_changed = False
        miss_deltas: List[int] = []
        any_upgrade_change = False
        any_ops_change = False
        for o in opt:
            d = o.get("delta") or {}
            any_path_changed = any_path_changed or bool(d.get("path_changed"))
            try:
                miss_deltas.append(int(d.get("missing_inputs_count_delta") or 0))
            except Exception:
                pass
            up = d.get("upgrade_exposure_bucket_changed") or {}
            ops = d.get("operational_exposure_bucket_changed") or {}
            any_upgrade_change = any_upgrade_change or (str(up.get("from")) != str(up.get("to")))
            any_ops_change = any_ops_change or (str(ops.get("from")) != str(ops.get("to")))

        delta_hint = []
        if opt:
            delta_hint.append(f"options={len(opt)}")
            delta_hint.append("path_changed" if any_path_changed else "path_same")
            if miss_deltas:
                delta_hint.append(f"missingΔ[{min(miss_deltas):+d},{max(miss_deltas):+d}]")
            if any_upgrade_change:
                delta_hint.append("upgrade_bucket_changed")
            if any_ops_change:
                delta_hint.append("ops_bucket_changed")
        else:
            delta_hint.append("no_options_generated")

        enriched.append(
            {
                "lever_id": lever_id,
                "lever_class": str(it.get("lever_class") or "unknown"),
                "label": str(it.get("label") or ""),
                "ask_missing_fields": missing_fields,
                "ask_unknown_fields": unknown_lever_fields,
                "missing_to_satisfied": missing_to_satisfied,
                "not_satisfied_to_satisfied": not_satisfied_to_satisfied,
                "referenced_rule_count": int(it.get("referenced_rule_count") or 0),
                "delta_hint": " | ".join(delta_hint),
                "_score": _score(it),
            }
        )

    if not enriched:
        return "<div class=\"muted small\">None</div>"

    # Pick top levers (deterministic sort)
    enriched.sort(key=lambda x: (-float(x.get("_score") or 0.0), str(x.get("lever_id") or "")))
    top = enriched[: max(1, int(top_n))]

    rows: List[str] = []
    for it in top:
        mflip = it.get("missing_to_satisfied") or []
        nflip = it.get("not_satisfied_to_satisfied") or []

        def _render_flip_items(items: List[Dict[str, Any]]) -> str:
            if not items:
                return "<div class=\"muted small\">None</div>"
            items2 = sorted(items, key=lambda x: str(x.get("rule_id") or ""))
            lis: List[str] = []
            for x in items2[:12]:
                rid = x.get("rule_id")
                cond = x.get("conditions") or []
                cond2 = [str(c) for c in cond[:2]] if isinstance(cond, list) else []
                cond_s = " ; ".join(cond2)
                extra = " …" if (isinstance(cond, list) and len(cond) > len(cond2)) else ""
                details = f"<span class=\"muted small\"> — {esc(cond_s)}{esc(extra)}</span>" if cond_s else ""
                lis.append(f"<li class=\"small\"><code>{esc(rid)}</code>{details}</li>")
            more = f"<div class=\"muted small\" style=\"margin-top:6px\">(+{len(items2)-12} more)</div>" if len(items2) > 12 else ""
            return "<ul>" + "".join(lis) + "</ul>" + more

        m_count = len(mflip) if isinstance(mflip, list) else 0
        n_count = len(nflip) if isinstance(nflip, list) else 0

        chips = (
            "<div class=\"chips\">"
            f"<span class=\"chip warn\"><span class=\"small\">missing→sat</span> <code>{m_count}</code></span>"
            f"<span class=\"chip info\"><span class=\"small\">not_sat→sat</span> <code>{n_count}</code></span>"
            "</div>"
        )

        details = ""
        if m_count or n_count:
            details = (
                "<details class=\"mini\">"
                "<summary>Show rules</summary>"
                "<div class=\"flip-box\">"
                f"<div class=\"muted small\">missing → satisfied</div>{_render_flip_items(mflip)}"
                f"<div class=\"muted small\" style=\"margin-top:10px\">not_satisfied → satisfied</div>{_render_flip_items(nflip)}"
                "</div>"
                "</details>"
            )

        flips_html = f"<div class=\"lever-flips\">{chips}{details}</div>"

        cls = str(it.get("lever_class") or "unknown").strip().lower()
        if cls not in {"hard", "design", "external", "unknown"}:
            cls = "unknown"
        cls_badge = (
            f"<span class=\"chip info\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "design" else
            (f"<span class=\"chip warn\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "external" else
             f"<span class=\"chip\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>")
        )

        ask_missing = it.get("ask_missing_fields") or []
        ask_unknown = it.get("ask_unknown_fields") or []
        unk_html = ""
        if isinstance(ask_unknown, list) and ask_unknown:
            unk_html = '<div class="muted small" style="margin-top:6px">unknown: ' + code_list(ask_unknown) + "</div>"
        ask_cell = code_list(ask_missing) + unk_html

        rows.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(it.get('lever_id'))}</code></td>"
            f"<td class=\"nowrap\">{cls_badge}</td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(it.get('label'))}</span></td>"
            f"<td class=\"wrap\">{ask_cell}</td>"
            f"<td class=\"wrap\"><span class=\"small\"><code>{esc(it.get('delta_hint'))}</code></span></td>"
            f"<td class=\"wrap\"><code>{esc(it.get('referenced_rule_count'))}</code></td>"
            f"<td class=\"wrap\">{flips_html}</td>"
            "</tr>"
        )

    return (
        "<div style=\"overflow-x:auto;margin-top:8px\">"
        "<table class=\"lever-next\">"
        "<thead><tr>"
        "<th class=\"nowrap\">lever</th>"
        "<th class=\"nowrap\">class</th>"
        "<th>label</th>"
        "<th>ask next (fields)</th>"
        "<th class=\"nowrap\">observed delta</th>"
        "<th class=\"nowrap\">rule refs</th>"
        "<th>rule check flips</th>"
        "</tr></thead>"
        "<tbody>"
        + ("\n".join(rows) or "<tr><td colspan=\"7\" class=\"muted\">None</td></tr>")
        + "</tbody></table></div>"
    )


def render_citation_audit_rows(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "<tr><td colspan=\"6\" class=\"muted\">Not run</td></tr>"
    findings = report.get("findings") or []
    if not findings:
        return "<tr><td colspan=\"6\" class=\"muted\">None</td></tr>"

    rows: List[str] = []
    for f in findings:
        status = str(f.get("status") or "")
        if status == "ok":
            continue
        rows.append(
            "<tr>"
            f"<td class=\"nowrap\">{badge_for_status(status)}</td>"
            f"<td class=\"nowrap\"><code>{esc(f.get('rule_id'))}</code></td>"
            f"<td class=\"nowrap\"><code>{esc(f.get('doc_id'))}</code></td>"
            f"<td class=\"wrap\"><span class=\"muted\">{esc(f.get('loc'))}</span></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(f.get('anchor_used') or '')}</span></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(f.get('message'))}</span></td>"
            "</tr>"
        )
    if not rows:
        return "<tr><td colspan=\"6\" class=\"muted\">No issues (all ok)</td></tr>"
    return "\n".join(rows)


def render_eval_to_html(tpl: str, ev: Dict[str, Any], *, citation_audit: Optional[Dict[str, Any]]) -> str:
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
        f"<td class=\"wrap\">{trace_list(e.get('criteria_traces') or [])}</td>"
        f"<td class=\"wrap\">{code_list(e.get('rule_ids') or [])}</td>"
        "</tr>"
        for e in traversed
    ) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>"

    missing_chips = chip_list(ev.get("missing_inputs") or [])

    levers = ev.get("levers") or []
    levers_rows_parts: List[str] = []
    for l in levers:
        cls = str(l.get("lever_class") or "unknown").strip().lower()
        if cls not in {"hard", "design", "external", "unknown"}:
            cls = "unknown"
        cls_chip = (
            f"<span class=\"chip info\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "design" else
            (f"<span class=\"chip warn\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "external" else
             f"<span class=\"chip\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>")
        )
        levers_rows_parts.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(l.get('lever_id'))}</code></td>"
            f"<td class=\"nowrap\">{cls_chip}</td>"
            f"<td class=\"wrap\">{esc(l.get('note'))}</td>"
            "</tr>"
        )
    levers_rows = "\n".join(levers_rows_parts) or "<tr><td colspan=\"3\" class=\"muted\">None detected in request</td></tr>"

    flags_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(f.get('rule_id'))}</code></td>"
        f"<td class=\"nowrap\"><code>{esc(f.get('doc_id'))}</code></td>"
        f"<td class=\"wrap\"><span class=\"muted\">{esc(f.get('loc'))}</span></td>"
        f"<td class=\"wrap\">{code_list(f.get('trigger_tags') or [])}</td>"
        f"<td class=\"wrap\">{trace_list(f.get('match_traces') or [])}</td>"
        "</tr>"
        for f in flags
    ) or "<tr><td colspan=\"5\" class=\"muted\">None</td></tr>"

    checklist_rows = render_checklist_rows(ev.get("energization_checklist") or [])

    lever_class_by_id: Dict[str, str] = {}
    for it in (ev.get("levers_catalog_analysis") or []):
        if not isinstance(it, dict):
            continue
        lid = str(it.get("lever_id") or "").strip()
        if not lid:
            continue
        cls = str(it.get("lever_class") or "unknown").strip().lower()
        if cls not in {"hard", "design", "external", "unknown"}:
            cls = "unknown"
        lever_class_by_id[lid] = cls

    options_rows = render_options_rows(ev.get("options") or [], lever_class_by_id=lever_class_by_id)

    evidence_rows = "\n".join(
        "<tr>"
        f"<td class=\"nowrap\"><code>{esc(e.get('rule_id'))}</code></td>"
        f"<td class=\"nowrap\"><code>{esc(e.get('doc_id'))}</code></td>"
        f"<td class=\"wrap\"><span class=\"muted\">{esc(e.get('loc'))}</span></td>"
        f"<td class=\"wrap\">{code_list(e.get('trigger_tags') or [])}</td>"
        "</tr>"
        for e in evidence
    ) or "<tr><td colspan=\"4\" class=\"muted\">None</td></tr>"

    checklist_stats_chips = render_checklist_stats(ev.get("energization_checklist") or [])
    evidence_stats_chips, evidence_total_count = render_evidence_stats(evidence, ev.get("rule_checks") or [])
    evidence_grouped_by_doc = render_evidence_grouped_by_doc(evidence)
    evidence_grouped_by_tag = render_evidence_grouped_by_tag(evidence)

    audit_mode = "off" if not citation_audit else ("strict" if citation_audit.get("_strict") else "warn")
    audit_ok = "not_run"
    audit_err = ""
    audit_warn = ""
    audit_rows = render_citation_audit_rows(citation_audit)
    if citation_audit:
        audit_ok = str(bool(citation_audit.get("ok")))
        audit_err = str(citation_audit.get("error_count", 0))
        audit_warn = str(citation_audit.get("warn_count", 0))

    risk_trace_json = json.dumps(ev.get("risk_trace") or [], ensure_ascii=False, indent=2)
    path_graph = render_path_graph(ev)
    full_graph_svg = render_full_process_graph_svg(ev)

    # PDF embed: assume PDFs are rendered to memo/outputs/pdfs/ with the same safe filename as HTML.
    pdf_filename = f"{safe_filename(str(req.get('project_name', 'memo')))}.pdf"

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
        .replace("{{executive_brief_html}}", render_executive_brief_html(ev))
        .replace("{{pdf_filename}}", esc(pdf_filename))
        .replace("{{context_snapshot_html}}", render_context_snapshot_html(ev))
        .replace("{{path}}", esc(" → ".join(path_labels)))
        .replace("{{decision_status_badge}}", render_decision_status_badge(ev))
        .replace("{{decision_reason_chips}}", render_decision_reason_chips(ev))
        .replace("{{next_actions_html}}", render_next_actions_html(ev))
        .replace("{{recommendation_html}}", render_recommendation_html(ev))
        .replace("{{top_drivers_html}}", render_top_drivers_html(ev))
        .replace("{{path_graph}}", path_graph)
        .replace("{{full_graph_svg}}", full_graph_svg)
        .replace("{{edges_rows}}", edges_rows)
        .replace("{{missing_chips}}", missing_chips)
        .replace("{{uncertainty_html}}", render_uncertainty_html(ev))
        .replace("{{levers_rows}}", levers_rows)
        .replace("{{lever_next_actions_html}}", render_lever_next_actions_html(ev, top_n=5))
        .replace("{{levers_catalog_html}}", render_levers_catalog_html(ev))
        .replace("{{options_rows}}", options_rows)
        .replace("{{flags_rows}}", flags_rows)
        .replace("{{checklist_rows}}", checklist_rows)
        .replace("{{checklist_stats_chips}}", checklist_stats_chips)
        .replace("{{risk_status}}", esc(risk_status))
        .replace("{{bucket_le_12}}", esc(tb.get("le_12_months", "unknown")))
        .replace("{{bucket_12_24}}", esc(tb.get("m12_24_months", "unknown")))
        .replace("{{bucket_gt_24}}", esc(tb.get("gt_24_months", "unknown")))
        .replace("{{upgrade_exposure}}", esc(risk.get("upgrade_exposure_bucket", "unknown")))
        .replace("{{operational_exposure}}", esc(risk.get("operational_exposure_bucket", "unknown")))
        .replace("{{timeline_model_html}}", render_timeline_model_html(ev))
        .replace("{{evidence_rows}}", evidence_rows)
        .replace("{{evidence_stats_chips}}", evidence_stats_chips)
        .replace("{{evidence_total_count}}", esc(evidence_total_count))
        .replace("{{evidence_grouped_by_doc}}", evidence_grouped_by_doc)
        .replace("{{evidence_grouped_by_tag}}", evidence_grouped_by_tag)
        .replace("{{risk_trace_json}}", esc(risk_trace_json))
        .replace("{{graph_version}}", esc(graph.get("graph_version")))
        .replace("{{graph_sha256}}", esc((ev.get("provenance") or {}).get("graph_sha256")))
        .replace("{{rules_source}}", esc((ev.get("provenance") or {}).get("rules_source")))
        .replace("{{rules_sha256}}", esc((ev.get("provenance") or {}).get("rules_sha256")))
        .replace("{{citation_audit_mode}}", esc(audit_mode))
        .replace("{{citation_audit_ok}}", esc(audit_ok))
        .replace("{{citation_audit_error_count}}", esc(audit_err))
        .replace("{{citation_audit_warn_count}}", esc(audit_warn))
        .replace("{{citation_audit_rows}}", audit_rows)
        .replace("{{docs_rows}}", render_docs_rows(ev.get("provenance") or {}))
    )


def render_checklist_rows(items: List[Dict[str, Any]]) -> str:
    rows = [c for c in items if c.get("status") != "not_applicable"]
    if not rows:
        return "<tr><td colspan=\"5\" class=\"muted\">Not requested (set <code>is_requesting_energization=true</code> to evaluate)</td></tr>"

    out: List[str] = []
    for c in rows:
        crit = str(c.get("criteria_text") or "").strip()
        crit_html = esc(crit) if crit else '<span class="muted small">(no criteria text)</span>'
        missing = c.get("missing_fields") or []
        traces = c.get("traces") or []
        mg_html = render_memo_guidance_html(c.get("memo_guidance"))
        guidance_cell = "<span class=\"muted small\">—</span>"
        if mg_html:
            guidance_cell = (
                "<details class=\"mini\">"
                "<summary>Guidance</summary>"
                f"<div class=\"flip-box\">{mg_html}</div>"
                "</details>"
            )
        src = rule_ref_details(rule_id=c.get("rule_id"), doc_id=c.get("doc_id"), loc=c.get("loc"))
        out.append(
            "<tr>"
            f"<td class=\"nowrap\">{badge_for_status(str(c.get('status') or 'unknown'))}</td>"
            f"<td class=\"wrap\"><div class=\"stmt\">{crit_html}</div>{src}</td>"
            f"<td class=\"wrap\">{code_list(missing)}</td>"
            f"<td class=\"wrap\">{trace_list(traces)}</td>"
            f"<td class=\"wrap\">{guidance_cell}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_options_rows(items: List[Dict[str, Any]], *, lever_class_by_id: Optional[Dict[str, str]] = None) -> str:
    if not items:
        return (
            "<tr><td colspan=\"9\" class=\"muted\">"
            "None (provide multiple <code>voltage_options_kv</code> or set <code>energization_plan</code> to compare)"
            "</td></tr>"
        )

    out: List[str] = []

    def _edge_diff_cell(delta: Dict[str, Any]) -> str:
        if not bool(delta.get("path_changed")):
            return '<span class="muted small">path_same</span>'

        ed = delta.get("edge_diff")
        if not isinstance(ed, dict):
            return '<span class="muted small">(no edge_diff)</span>'

        dw = ed.get("diff_window") or {}
        try:
            start_i = int(dw.get("start_index"))
        except Exception:
            start_i = None

        bseg = ed.get("baseline_segment") or []
        oseg = ed.get("option_segment") or []
        if not isinstance(bseg, list):
            bseg = []
        if not isinstance(oseg, list):
            oseg = []

        def _render_edge_seg(seg: List[Dict[str, Any]], title: str) -> str:
            parts: List[str] = [f'<div class="muted small" style="margin-top:6px"><strong>{esc(title)}</strong></div>']
            if not seg:
                parts.append('<div class="muted small">(none)</div>')
                return "".join(parts)

            for e in seg[:8]:
                if not isinstance(e, dict):
                    continue
                eid = e.get("edge_id")
                fr = e.get("from")
                to = e.get("to")
                traces = [str(x) for x in (e.get("criteria_traces") or []) if str(x).strip()]
                trace_html = trace_list(traces[:6]) if traces else '<span class="muted small">(no predicate traces)</span>'

                deltas = e.get("criteria_field_deltas") or []
                if not isinstance(deltas, list):
                    deltas = []
                delta_rows = []
                for d in deltas[:10]:
                    if not isinstance(d, dict):
                        continue
                    delta_rows.append(
                        f'<li><code>{esc(d.get("field"))}</code>: '
                        f'<span class="muted">{esc(d.get("from"))}</span> → '
                        f'<span class="muted">{esc(d.get("to"))}</span></li>'
                    )
                delta_html = ""
                if delta_rows:
                    delta_html = (
                        '<div class="muted small" style="margin-top:6px">trigger field deltas</div>'
                        + "<ul>"
                        + "".join(delta_rows)
                        + "</ul>"
                    )

                cites = e.get("rule_citations") or []
                if not isinstance(cites, list):
                    cites = []
                cite_rows = []
                for c in cites[:12]:
                    if not isinstance(c, dict):
                        continue
                    cite_rows.append(
                        f'<li><code>{esc(c.get("rule_id"))}</code> — '
                        f'<code>{esc(c.get("doc_id") or "")}</code> — '
                        f'<span class="muted">{esc(c.get("loc") or "")}</span></li>'
                    )
                cite_html = (
                    '<div class="muted small" style="margin-top:6px">edge citations (rule → doc loc)</div>'
                    + "<ul>"
                    + ("".join(cite_rows) if cite_rows else '<li class="muted small">(none)</li>')
                    + "</ul>"
                )

                parts.append(
                    "<div style=\"margin-top:10px\">"
                    + f'<div><code>{esc(eid)}</code> <span class="muted">{esc(fr)} → {esc(to)}</span></div>'
                    + f'<div class="trace-wrap" style="margin-top:6px">{trace_html}</div>'
                    + delta_html
                    + cite_html
                    + "</div>"
                )

            if len(seg) > 8:
                parts.append(f'<div class="muted small" style="margin-top:6px">(+{len(seg)-8} more edges)</div>')

            return "".join(parts)

        hint = f"diff_start_index={start_i}" if start_i is not None else "diff"
        body = (
            f'<div class="muted small">{esc(hint)}</div>'
            + _render_edge_seg(bseg, "baseline edges (changed segment)")
            + _render_edge_seg(oseg, "option edges (changed segment)")
        )

        return (
            '<details class="mini">'
            "<summary>Edge diff + why</summary>"
            f'<div class="flip-box">{body}</div>'
            "</details>"
        )

    for it in items:
        opt_id = it.get("option_id")
        lever = it.get("lever_id")
        lever_id_s = str(lever or "").strip()
        cls = "unknown"
        if isinstance(lever_class_by_id, dict) and lever_id_s:
            cls = str(lever_class_by_id.get(lever_id_s) or "unknown").strip().lower()
        if cls not in {"hard", "design", "external", "unknown"}:
            cls = "unknown"
        cls_chip = (
            f"<span class=\"chip info\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "design" else
            (f"<span class=\"chip warn\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>" if cls == "external" else
             f"<span class=\"chip\"><span class=\"small\">class</span> <code>{esc(cls)}</code></span>")
        )
        source = str(it.get("source") or "").strip() or "unknown"
        patch = it.get("patch") or {}
        summ = it.get("summary") or {}
        delta = it.get("delta") or {}

        path = summ.get("path") or ""
        miss = summ.get("missing_inputs") or []
        miss_count = summ.get("missing_inputs_count", len(miss))
        tb = summ.get("timeline_buckets") or {}
        risk_text = (
            f"≤12:{tb.get('le_12_months','unknown')} | "
            f"12–24:{tb.get('m12_24_months','unknown')} | "
            f">24:{tb.get('gt_24_months','unknown')} | "
            f"upgrade:{summ.get('upgrade_exposure_bucket','unknown')} | "
            f"ops:{summ.get('operational_exposure_bucket','unknown')}"
        )

        patch_s = json.dumps(patch, ensure_ascii=False)

        # Compact delta rendering (bounded recommendation-friendly)
        miss_delta = delta.get("missing_inputs_count_delta", 0)
        try:
            miss_delta_i = int(miss_delta)
        except Exception:
            miss_delta_i = 0
        miss_delta_s = f"{miss_delta_i:+d}" if miss_delta_i else "0"

        path_changed = bool(delta.get("path_changed"))
        path_s = "path_changed" if path_changed else "path_same"

        up = delta.get("upgrade_exposure_bucket_changed") or {}
        ops = delta.get("operational_exposure_bucket_changed") or {}
        up_s = f"upgrade:{up.get('from','?')}→{up.get('to','?')}"
        ops_s = f"ops:{ops.get('from','?')}→{ops.get('to','?')}"
        delta_text = f"{path_s} | missing:{miss_delta_s} | {up_s} | {ops_s}"

        out.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(opt_id)}</code></td>"
            f"<td class=\"wrap\"><div class=\"chips\" style=\"margin-top:0\"><span class=\"chip\"><span class=\"small\">lever</span> <code>{esc(lever)}</code></span>{cls_chip}</div></td>"
            f"<td class=\"nowrap\"><code>{esc(source)}</code></td>"
            f"<td class=\"wrap\"><span class=\"small\"><code title=\"{esc(patch_s)}\">{esc(patch_s)}</code></span></td>"
            f"<td class=\"wrap\"><span class=\"small nowrap\" title=\"{esc(path)}\">{esc(path)}</span></td>"
            f"<td class=\"wrap\"><code>{esc(miss_count)}</code> {code_list(miss[:8])}</td>"
            f"<td class=\"wrap\"><span class=\"small\"><code>{esc(delta_text)}</code></span></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(risk_text)}</span></td>"
            f"<td class=\"wrap\">{_edge_diff_cell(delta)}</td>"
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
    ap.add_argument(
        "--citation-audit",
        choices=["off", "warn", "strict"],
        default="strict",
        help="Whether to run PDF citation audit and gate memo generation",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl = load_template()

    citation_report: Optional[Dict[str, Any]] = None
    if args.citation_audit != "off":
        strict = args.citation_audit == "strict"
        citation_report = audit_citations(strict=strict)
        # carry mode hint for rendering
        citation_report["_strict"] = strict
        if not citation_report.get("ok"):
            print(json.dumps(citation_report, ensure_ascii=False, indent=2))
            return 3

    n = 0
    for ev in iter_jsonl(args.in_path):
        req = ev.get("request") or {}
        name = safe_filename(str(req.get("project_name", f"memo_{n+1}")))
        out_path = out_dir / f"{name}.html"
        out_path.write_text(render_eval_to_html(tpl, ev, citation_audit=citation_report), encoding="utf-8")
        n += 1

    print(f"Wrote {n} memos to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
