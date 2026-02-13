from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from io_jsonl import iter_jsonl
from citation_audit import audit_citations


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "memo" / "templates" / "memo_template.html"
GRAPH_PATH = REPO_ROOT / "graph" / "process_graph.yaml"


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
    cls_attr = f" badge {cls}".strip()
    return f"<span class=\"{cls_attr}\"><code>{esc(status)}</code></span>"


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
        f"<td class=\"wrap\">{trace_list(f.get('match_traces') or [])}</td>"
        "</tr>"
        for f in flags
    ) or "<tr><td colspan=\"5\" class=\"muted\">None</td></tr>"

    checklist_rows = render_checklist_rows(ev.get("energization_checklist") or [])

    options_rows = render_options_rows(ev.get("options") or [])

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
        .replace("{{path_graph}}", path_graph)
        .replace("{{full_graph_svg}}", full_graph_svg)
        .replace("{{edges_rows}}", edges_rows)
        .replace("{{missing_chips}}", missing_chips)
        .replace("{{levers_rows}}", levers_rows)
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
            f"<td class=\"wrap\">{trace_list(traces)}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_options_rows(items: List[Dict[str, Any]]) -> str:
    if not items:
        return (
            "<tr><td colspan=\"6\" class=\"muted\">"
            "None (provide multiple <code>voltage_options_kv</code> or set <code>energization_plan</code> to compare)"
            "</td></tr>"
        )

    out: List[str] = []
    for it in items:
        opt_id = it.get("option_id")
        lever = it.get("lever_id")
        patch = it.get("patch") or {}
        summ = it.get("summary") or {}

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
        out.append(
            "<tr>"
            f"<td class=\"nowrap\"><code>{esc(opt_id)}</code></td>"
            f"<td class=\"nowrap\"><code>{esc(lever)}</code></td>"
            f"<td class=\"wrap\"><span class=\"small\"><code>{esc(patch_s)}</code></span></td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(path)}</span></td>"
            f"<td class=\"wrap\"><code>{esc(miss_count)}</code> {code_list(miss[:8])}</td>"
            f"<td class=\"wrap\"><span class=\"small\">{esc(risk_text)}</span></td>"
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
