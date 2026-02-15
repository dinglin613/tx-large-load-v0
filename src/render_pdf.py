import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF

from io_jsonl import iter_jsonl


def _s(x: Any) -> str:
    return "" if x is None else str(x)


def _pdf_text(t: Any) -> str:
    """
    fpdf2 core fonts (Helvetica) are not Unicode; sanitize to Latin-1-ish ASCII.
    We keep meaning by replacing common symbols (≤, ≥, →, …) with ASCII equivalents.
    """
    s = _s(t)
    repl = {
        "—": "-",
        "–": "-",
        "−": "-",
        "→": "->",
        "⇒": "=>",
        "≤": "<=",
        "≥": ">=",
        "…": "...",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "•": "-",
        "\u00a0": " ",  # NBSP
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    def _soft_break_long_runs(txt: str, *, max_run: int = 26) -> str:
        out: List[str] = []
        run = 0
        for ch in txt:
            if ch.isspace():
                run = 0
                out.append(ch)
                continue
            run += 1
            out.append(ch)
            if run >= max_run:
                out.append(" ")
                run = 0
        return "".join(out)

    s = _soft_break_long_runs(s, max_run=26)
    # Replace any remaining non-latin-1 chars with '?'
    return s.encode("latin-1", "replace").decode("latin-1")


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _human_id(x: Any) -> str:
    """
    Turn snake_case / machine ids into readable phrases.
    """
    s = _s(x).strip()
    if not s:
        return ""
    # keep project_name underscores (often used as separators) readable
    if s.isupper() and "_" in s:
        return s.replace("_", " ")
    s2 = s.replace("_", " ").strip()
    # light title-casing, keep known acronyms
    acr = {"ERCOT", "TDSP", "TSP", "LLIS", "ILLE", "NOM", "QSA", "SSO", "DME", "POI", "COD"}
    words = []
    for w in s2.split():
        up = w.upper()
        if up in acr:
            words.append(up)
        elif w.isupper() and len(w) <= 5:
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:])
    return " ".join(words)


def _shorten(t: Any, *, max_len: int = 160) -> str:
    s = _s(t).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _energization_note(dec_e: Dict[str, Any]) -> str:
    reasons = [str(x) for x in _as_list(dec_e.get("reasons")) if str(x).strip()]
    r = " ".join(reasons).lower()
    if "energization_not_requested" in r or "not_requested" in r:
        return "not requested"
    # map v0 statuses to exec-friendly wording
    s = _s(dec_e.get("status") or "unknown").strip().lower()
    if s == "ready":
        return "ready (based on provided inputs)"
    if s == "not_ready":
        return "not ready (based on provided inputs)"
    return s or "unknown"


def _missing_requirement_entries(ev: Dict[str, Any], *, max_items: int = 8) -> List[Dict[str, Any]]:
    """
    Translate missing_inputs into human-readable requirements using rule_checks rows.
    Each entry has: field, requirement, doc_id, loc, rule_id
    """
    miss = {str(x) for x in _as_list(ev.get("missing_inputs")) if str(x).strip()}
    out: List[Dict[str, Any]] = []
    for rc in _as_list(ev.get("rule_checks")):
        if not isinstance(rc, dict):
            continue
        if str(rc.get("status")) != "missing":
            continue
        mf = {str(x) for x in _as_list(rc.get("missing_fields")) if str(x).strip()}
        hit = sorted(mf & miss)
        if not hit:
            continue
        crit = _s(rc.get("criteria_text")).strip()
        for f in hit:
            out.append(
                {
                    "field": f,
                    "requirement": crit or f"Provide {f}",
                    "rule_id": rc.get("rule_id"),
                    "doc_id": rc.get("doc_id"),
                    "loc": rc.get("loc"),
                }
            )
        if len(out) >= max_items:
            break
    # de-dup by field preserve order
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in out:
        f = str(it.get("field") or "")
        if f in seen:
            continue
        seen.add(f)
        uniq.append(it)
    return uniq[:max_items]


def _deliverable_hint(field: str) -> str:
    f = (field or "").strip()
    m = {
        "one_line_diagram": "One-line diagram (PDF)",
        "load_projection_5y": "5-year load projection (spreadsheet/PDF)",
        "llis_formal_request_submitted": "Formal LLIS initiation request (email/letter + confirmation)",
        "llis_data_package_submitted": "Study data package submission receipt (files + transmittal)",
        "ack_change_notification_obligation": "Signed acknowledgement of change-notification obligation (letter/email)",
        "phases": "Phasing schedule table (MW increments + dates) in Load Commissioning Plan",
        "telemetry_operational_and_accurate": "Telemetry commissioning evidence (test results + signoff)",
        "nom_included": "NOM inclusion confirmation (ERCOT notice/screenshot)",
        "agreements_executed": "Executed agreements + financial security receipt",
    }
    return m.get(f, f"Evidence artifact for {_human_id(f)}")


def _blocking_actions(ev: Dict[str, Any], *, max_items: int = 8) -> List[str]:
    entries = _missing_requirement_entries(ev, max_items=max_items)
    if not entries:
        # fallback
        miss = [str(x) for x in _as_list(ev.get("missing_inputs")) if str(x).strip()]
        return [f"Provide {_human_id(x)} (owner: Customer; deliverable: {_deliverable_hint(x)})" for x in miss[:max_items]]
    out: List[str] = []
    for e in entries:
        f = str(e.get("field") or "")
        req = _shorten(e.get("requirement"), max_len=140)
        out.append(
            f"{req} Action: provide {_human_id(f)}. Owner: Customer. Deliverable: {_deliverable_hint(f)}."
        )
    return out[:max_items]

def _pick_blocking_missing(ev: Dict[str, Any], *, n: int = 3) -> List[str]:
    # Prefer human-readable requirement labels, not raw field ids.
    entries = _missing_requirement_entries(ev, max_items=max(1, n))
    if entries:
        return [_human_id(e.get("field")) or _shorten(e.get("requirement"), max_len=90) for e in entries[:n]]
    miss = [str(x) for x in _as_list(ev.get("missing_inputs")) if str(x).strip()]
    return [_human_id(x) for x in miss[: max(0, n)]]


def _next_actions_human(ev: Dict[str, Any], *, n: int = 6) -> List[str]:
    # Drive next actions primarily from blocking missing requirements (humanized).
    out = _blocking_actions(ev, max_items=max(3, n))
    # If we still have room, append any additional system-generated notes.
    for a in _as_list(ev.get("next_actions")):
        if len(out) >= n:
            break
        if not isinstance(a, dict):
            continue
        note = _s(a.get("note")).strip()
        if note and note not in out:
            out.append(note)
    return out[: max(0, n)]


def _bounded_reco_one_liner(ev: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (headline, reason).
    """
    rec = ev.get("recommendation") or {}
    rid = _s(rec.get("recommended_option_id")).strip() or "baseline"
    is_base = bool(rec.get("recommended_is_baseline"))
    rationale = rec.get("rationale") or []
    r1 = ""
    if isinstance(rationale, list) and rationale:
        r1 = _s(rationale[0]).strip()
    if not r1:
        r1 = "Bounded recommendation within baseline + generated options (v0)."
    if is_base:
        return (f"Recommendation (bounded): keep baseline", r1)
    return (f"Recommendation (bounded): prefer option {rid}", r1)


def _top_drivers_human(ev: Dict[str, Any], *, n: int = 3) -> List[str]:
    out: List[str] = []
    for d in _as_list(ev.get("top_drivers")):
        if not isinstance(d, dict):
            continue
        summary = _s(d.get("summary")).strip()
        if not summary:
            # fall back to rule_id if no summary exists
            summary = _s(d.get("rule_id")).strip()
        if summary:
            out.append(summary)
    return out[: max(0, n)]


def _risk_signal_line(ev: Dict[str, Any]) -> str:
    risk = ev.get("risk") or {}
    tb = risk.get("timeline_buckets") or {}
    le_12 = _s(tb.get("le_12_months") or "unknown")
    m12_24 = _s(tb.get("m12_24_months") or "unknown")
    gt_24 = _s(tb.get("gt_24_months") or "unknown")
    up = _s(risk.get("upgrade_exposure_bucket") or "unknown")
    ops = _s(risk.get("operational_exposure_bucket") or "unknown")
    return f"Timeline signals (qualitative): <=12={le_12}, 12-24={m12_24}, >24={gt_24}. Exposure: upgrade={up}, ops={ops}."


def _select_options_for_pdf(ev: Dict[str, Any], *, max_rows: int = 5) -> List[Dict[str, Any]]:
    """
    Baseline + a few best-ranked candidates from bounded recommendation.
    """
    rec = ev.get("recommendation") or {}
    ranked = rec.get("candidates_ranked") or []
    rows: List[Dict[str, Any]] = []
    if isinstance(ranked, list) and ranked:
        for c in ranked:
            if not isinstance(c, dict):
                continue
            # candidates_ranked entries are compact; keep as-is
            rows.append(c)
            if len(rows) >= max_rows:
                break
        return rows

    # fallback: baseline + raw options
    opts = [o for o in _as_list(ev.get("options")) if isinstance(o, dict)]
    base = {"option_id": "baseline", "patch": {}, "summary": ev.get("risk") or {}, "source": "baseline"}
    rows = [base] + opts[: max(0, max_rows - 1)]
    return rows[: max_rows]


def _option_config_summary(opt: Dict[str, Any], ev: Dict[str, Any]) -> str:
    patch = opt.get("patch") or {}
    if not isinstance(patch, dict):
        patch = {}
    req = ev.get("request") or {}
    plan = _s(patch.get("energization_plan") or req.get("energization_plan") or "").strip() or "unknown"
    vopts = patch.get("voltage_options_kv") or req.get("voltage_options_kv")
    vtxt = "unknown"
    if isinstance(vopts, list) and vopts:
        vtxt = ",".join(_s(x) for x in vopts[:4])
    mw = patch.get("load_mw_total") or req.get("load_mw_total")
    mw_txt = _s(mw).strip() or "?"
    poi = patch.get("poi_count") or req.get("poi_count")
    poi_txt = _s(poi).strip() if poi is not None else "?"
    return f"plan={plan}; voltage={vtxt} kV; load={mw_txt} MW; POIs={poi_txt}"


def _option_risk_signal(opt: Dict[str, Any]) -> str:
    summ = opt.get("summary") or {}
    if not isinstance(summ, dict):
        summ = {}
    tb = summ.get("timeline_buckets") or {}
    le_12 = _s(tb.get("le_12_months") or "unknown")
    m12_24 = _s(tb.get("m12_24_months") or "unknown")
    gt_24 = _s(tb.get("gt_24_months") or "unknown")
    up = _s(summ.get("upgrade_exposure_bucket") or "unknown")
    ops = _s(summ.get("operational_exposure_bucket") or "unknown")
    return f"≤12={le_12}, 12–24={m12_24}, >24={gt_24}; upgrade={up}, ops={ops}"


def _option_path_changed(opt: Dict[str, Any]) -> str:
    d = opt.get("delta") or {}
    if not isinstance(d, dict):
        return "unknown"
    return "changed" if bool(d.get("path_changed")) else "same"


def _option_missing_count(opt: Dict[str, Any]) -> str:
    summ = opt.get("summary") or {}
    if not isinstance(summ, dict):
        return "?"
    try:
        return str(int(summ.get("missing_inputs_count") or 0))
    except Exception:
        return "?"


class ExecutivePDF(FPDF):
    def header(self) -> None:
        # Top rule
        self.set_draw_color(229, 231, 235)
        self.set_line_width(0.4)
        self.line(self.l_margin, 12, self.w - self.r_margin, 12)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_text_color(107, 114, 128)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")


def _h1(pdf: ExecutivePDF, t: str) -> None:
    pdf.set_text_color(17, 24, 39)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, _pdf_text(t))
    pdf.ln(1)


def _h2(pdf: ExecutivePDF, t: str) -> None:
    pdf.set_text_color(17, 24, 39)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 6, _pdf_text(t))
    pdf.ln(1)


def _p(pdf: ExecutivePDF, t: str) -> None:
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _pdf_text(t))
    pdf.ln(1)


def _bullets(pdf: ExecutivePDF, items: List[str]) -> None:
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "", 10)
    for it in items:
        # Ensure we always start at the left margin (fpdf can keep x after other calls in edge cases).
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, _pdf_text(f"- {it}"))
    pdf.ln(1)


def _line(pdf: ExecutivePDF, t: str, *, bold: bool = False, size: int = 10) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.multi_cell(0, 5, _pdf_text(t))


def render_one_pdf(ev: Dict[str, Any], *, out_path: Path, max_options: int = 5, include_audit_page: bool = True) -> None:
    req = ev.get("request") or {}
    project = _s(req.get("project_name")).strip() or "decision_memo"

    dec_s = ev.get("decision_screening") or {}
    dec_e = ev.get("decision_energization") or {}
    screening_status = _s(dec_s.get("status") or "unknown")
    energ_note = _energization_note(dec_e)

    blocking = _pick_blocking_missing(ev, n=3)
    next_actions = _next_actions_human(ev, n=6)
    reco_head, reco_reason = _bounded_reco_one_liner(ev)
    drivers = _top_drivers_human(ev, n=3)

    pdf = ExecutivePDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(14, 16, 14)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    # Page 1: Executive summary
    _h1(pdf, f"Decision Brief - {project}")
    # Project snapshot line
    snap = (
        f"Operator: {_s(req.get('operator_area') or '')}; TDSP: {_s(req.get('tdsp_area') or '')}; "
        f"Load: {_s(req.get('load_mw_total') or '?')} MW; COD target: {_s(req.get('cod_target_window') or 'n/a')}."
    )
    _p(pdf, snap)

    # 5 fixed human sentences (no ids)
    _p(pdf, f"1) Current conclusion: Screening status is {screening_status}. Energization is {energ_note}.")
    _p(pdf, "2) Blocking now: the following items must be provided to proceed:")
    _bullets(pdf, blocking if blocking else ["No blocking inputs detected."])
    _p(pdf, "3) Next actions (this week):")
    _bullets(pdf, next_actions if next_actions else ["No actions generated."])
    _p(pdf, f"4) {reco_head}.")
    _p(pdf, f"Reason (bounded, v0): {_shorten(reco_reason, max_len=180)}")
    _p(pdf, "5) Main risks & external dependencies:")
    lines = [_risk_signal_line(ev)]
    if drivers:
        lines.append("Top drivers: " + "; ".join(drivers))
    lines.append("Note: final requirements may depend on ERCOT/TSP/TDSP discretion, studies, and site verification.")
    _bullets(pdf, lines)

    # Page 2: Options comparison (bounded)
    pdf.add_page()
    _h1(pdf, "Options comparison (bounded)")
    _p(pdf, "Curated options for decision-makers. Timeline uses qualitative signals (not probabilities).")

    options = _select_options_for_pdf(ev, max_rows=max_options)
    for opt in options:
        oid = _s(opt.get("option_id")).strip() or "baseline"
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(17, 24, 39)
        pdf.multi_cell(0, 6, _pdf_text(f"{oid}"))
        _line(pdf, f"Config summary: {_option_config_summary(opt, ev)}")
        _line(pdf, f"Path changed vs baseline: {_option_path_changed(opt)} | Blocking missing count: {_option_missing_count(opt)}")
        _line(pdf, f"Risk signals: {_option_risk_signal(opt)}")
        pdf.ln(2)

    # Page 3: Action checklist (blocking vs later gates)
    pdf.add_page()
    _h1(pdf, "Action checklist (to drive progress)")
    _h2(pdf, "Blocking now (focus here)")
    _p(pdf, "These are the concrete deliverables required to move the request forward.")
    _bullets(pdf, _blocking_actions(ev, max_items=10) or ["(none)"])

    _h2(pdf, "Later gates (prepare, not blocking today)")
    later: List[str] = []
    for c in _as_list(ev.get("energization_checklist")):
        if not isinstance(c, dict):
            continue
        s = _s(c.get("status")).strip().lower()
        if s in {"missing", "not_satisfied", "unknown"}:
            crit = _s(c.get("criteria_text")).strip()
            if crit:
                later.append(_shorten(crit, max_len=160))
        if len(later) >= 10:
            break
    if energ_note == "not requested":
        _p(pdf, "Shown for planning only (energization not requested in this evaluation).")
    _bullets(pdf, later or ["(no later gates listed)"])

    # Page 4: audit/provenance
    if include_audit_page:
        pdf.add_page()
        _h1(pdf, "Audit & provenance (short)")
        prov = ev.get("provenance") or {}
        _p(pdf, f"Evaluated at: {_s(ev.get('evaluated_at'))}")
        _p(pdf, f"Graph: {_s(prov.get('graph_source'))} | sha256={_s(prov.get('graph_sha256'))}")
        _p(pdf, f"Rules: {_s(prov.get('rules_source'))} | sha256={_s(prov.get('rules_sha256'))}")
        docs = _as_list(prov.get("docs"))
        if docs:
            _h2(pdf, "Referenced docs (subset)")
            for d in docs[:8]:
                if not isinstance(d, dict):
                    continue
                did = _s(d.get("doc_id"))
                title = _s(d.get("title"))
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, _pdf_text(f"{did} - {title}"))
                pdf.set_font("Helvetica", "", 9)
                arts = _as_list(d.get("artifacts"))
                if arts:
                    a0 = arts[0] if isinstance(arts[0], dict) else {}
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 4.5, _pdf_text(f"artifact: {_s(a0.get('path'))} | sha256={_s(a0.get('sha256'))}"))
                pdf.ln(1)

        _p(pdf, "Full evidence chain (rule_id → doc_id/loc → artifact sha256) is available in the HTML memo.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description="Render executive PDF memos from evaluations JSONL (bounded, human-readable)")
    ap.add_argument("--in", dest="in_path", required=True, help="Input evaluations JSONL (e.g., memo/outputs/evals.jsonl)")
    ap.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory for PDF memos")
    ap.add_argument("--max-options", type=int, default=5, help="Max options to include in the options page (default: 5)")
    ap.add_argument("--no-audit-page", action="store_true", help="Omit the audit/provenance page")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for ev in iter_jsonl(args.in_path):
        req = ev.get("request") or {}
        name = _s(req.get("project_name") or f"memo_{n+1}").strip() or f"memo_{n+1}"
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name).strip("_") or f"memo_{n+1}"
        out_path = out_dir / f"{safe}.pdf"
        render_one_pdf(
            ev,
            out_path=out_path,
            max_options=max(3, int(args.max_options)),
            include_audit_page=(not args.no_audit_page),
        )
        n += 1

    print(f"Wrote {n} PDFs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

