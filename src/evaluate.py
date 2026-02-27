from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml

from io_jsonl import iter_jsonl, read_jsonl, write_jsonl
from tag_taxonomy import TAG_RISK_CONTRIB, BASELINE_TAGS


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
    #
    # If multiple artifacts exist for the same doc_id (e.g., multiple Planning Guide versions),
    # prefer the artifact that matches the doc-level hash (canonical version).
    canonical_sha = str(doc.get("hash") or "").strip().lower()
    if canonical_sha:
        matched = [
            a
            for a in arts
            if isinstance(a, dict)
            and a.get("path")
            and a.get("sha256")
            and str(a.get("sha256")).strip().lower() == canonical_sha
        ]
    else:
        matched = []

    keep = matched or [
        a
        for a in arts
        if isinstance(a, dict) and a.get("path") and a.get("sha256")
    ]

    # Stable order: most recent retrieved_date first (if present), then path
    def _key(a: Dict[str, Any]) -> tuple[str, str]:
        return (str(a.get("retrieved_date") or ""), str(a.get("path") or ""))

    keep_sorted = sorted(keep, key=_key, reverse=True)
    return {
        "doc_id": doc.get("doc_id"),
        "title": doc.get("title"),
        "effective_date": doc.get("effective_date"),
        "artifacts": [
            {
                "path": a.get("path"),
                "sha256": a.get("sha256"),
                "retrieved_date": a.get("retrieved_date"),
            }
            for a in keep_sorted
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


def _classify_missing_reason(*, req: Dict[str, Any], field: str, is_completeness: bool) -> str:
    """
    Classify why a field is treated as "missing" in v0 evaluation.

    This is used for memo-grade uncertainty reporting; it does not change
    the evaluation semantics.
    """
    if field not in req:
        return "absent"
    v = req.get(field)
    if v is None:
        return "null"
    if isinstance(v, str) and v.strip() == "":
        return "empty_string"
    if is_completeness and v is False:
        return "false_as_missing"
    if is_completeness and isinstance(v, list) and len(v) == 0:
        return "empty_list_as_missing"
    if isinstance(v, list) and len(v) == 0:
        return "empty_list"
    return "unknown"


def _is_explicit_unknown_value(v: Any) -> bool:
    """
    Detect explicit 'unknown' style markers in the request.

    We keep this intentionally conservative to avoid mislabeling valid values.
    """
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() == "unknown":
        return True
    return False


def _missing_required_fields_for_rule(rule: Dict[str, Any], req: Dict[str, Any]) -> List[str]:
    """
    Determine which required_fields are missing for a given rule.

    Align semantics with tri-state evaluation:
    - For completeness_check rules: False (and empty lists) are treated as missing.
    - For other rules: False is a valid factual value (not "missing").
    """
    required = required_fields_from_rule(rule)
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    is_completeness = dsl.get("kind") == "completeness_check"

    missing: List[str] = []
    for f in required:
        if f not in req:
            missing.append(f)
            continue
        v = req.get(f)
        if v is None:
            missing.append(f)
            continue
        if isinstance(v, str) and v.strip() == "":
            missing.append(f)
            continue
        if is_completeness and v is False:
            missing.append(f)
            continue
        if is_completeness and isinstance(v, list) and len(v) == 0:
            missing.append(f)
            continue
    return missing


def risk_from_signals(
    req: Dict[str, Any],
    signal_rules: List[Dict[str, Any]],
    *,
    missing_inputs: List[str],
    context_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute qualitative risk buckets with a reproducible trace.

    Returns:
      (risk_dict, risk_trace_list)
    """
    # v0: qualitative, explainable scores (no numeric "probabilities")
    # INCREMENTAL scores — only project-specific risk signals count toward exposure buckets
    upgrade_score = 0.0
    wait_score = 0.0
    ops_score = 0.0
    # BASELINE scores — inherent to every large-load process; tracked for transparency
    baseline_upgrade = 0.0
    baseline_wait = 0.0
    baseline_ops = 0.0

    evidence: List[Dict[str, Any]] = []
    risk_trace: List[Dict[str, Any]] = []

    def _add_trace(
        *,
        kind: str,
        rule_id: str,
        doc_id: str,
        loc: str,
        trigger_tags: List[str],
        contributions: Dict[str, float],
        notes: List[str],
        source: str,
    ) -> None:
        risk_trace.append(
            {
                "kind": kind,
                "source": source,  # published_rule | non_cited_heuristic
                "rule_id": rule_id,
                "doc_id": doc_id,
                "loc": loc,
                "trigger_tags": trigger_tags,
                "contributions": contributions,
                "notes": notes,
            }
        )

    # 1) Published rules (graph path + parallel flags) -> contributions via trigger_tags (+ completeness penalty)
    for r in signal_rules:
        rid = str(r.get("rule_id") or "")
        doc_id = str(r.get("doc_id") or "")
        loc = str(r.get("loc") or "")
        tags = sorted(set(r.get("trigger_tags") or []))

        # Only treat a rule as a risk "signal" when it is actually triggered by its predicates.
        #
        # v0 nuance:
        # - If predicates are explicitly false (and not missing), skip the rule entirely (no risk contribution).
        # - If predicates cannot be evaluated due to missing predicate fields, only keep the rule as a risk
        #   signal when it is a data completeness concept (so missing inputs remain visible in wait pressure).
        preds = rule_predicates(r)
        if preds:
            ok, _tr, missing = criteria_satisfied(req, preds)
            if (not ok) and (not missing):
                continue
            if (not ok) and missing and ("data_completeness" not in set(tags)):
                continue

        evidence.append(
            {
                "rule_id": rid,
                "doc_id": doc_id,
                "loc": loc,
                "trigger_tags": tags,
            }
        )

        contrib = {"upgrade": 0.0, "wait": 0.0, "ops": 0.0}
        baseline_contrib = {"upgrade": 0.0, "wait": 0.0, "ops": 0.0}
        notes: List[str] = []

        for t in tags:
            # Check BASELINE first; if a tag is in both, baseline wins (prevents double-count)
            bm = BASELINE_TAGS.get(t)
            if bm:
                for k, dv in bm.items():
                    baseline_contrib[k] = float(baseline_contrib.get(k, 0.0)) + float(dv)
                notes.append(f"tag:{t} => baseline (not counted toward exposure buckets)")
                continue
            m = TAG_RISK_CONTRIB.get(t)
            if not m:
                continue
            for k, dv in m.items():
                contrib[k] = float(contrib.get(k, 0.0)) + float(dv)
                notes.append(f"tag:{t} => {k} {dv:+g}")

        if "data_completeness" in set(tags):
            missing_for_rule = _missing_required_fields_for_rule(r, req)
            if missing_for_rule:
                # Data completeness penalty stays in baseline (process-inherent)
                baseline_contrib["wait"] += 2.0
                notes.append(f"data_completeness missing={missing_for_rule} => baseline wait +2.0")

        upgrade_score += contrib["upgrade"]
        wait_score += contrib["wait"]
        ops_score += contrib["ops"]
        baseline_upgrade += baseline_contrib["upgrade"]
        baseline_wait += baseline_contrib["wait"]
        baseline_ops += baseline_contrib["ops"]

        _add_trace(
            kind="rule",
            rule_id=rid,
            doc_id=doc_id,
            loc=loc,
            trigger_tags=tags,
            contributions=contrib,
            notes=notes,
            source="published_rule",
        )

    # 2) Non-cited heuristics "rule-ified" (explicitly marked, still auditable)
    NON_CITED_DOC = "NON_CITED_HEURISTIC"

    def _add_heuristic(
        *,
        rule_id: str,
        loc: str,
        tags: List[str],
        contrib: Dict[str, float],
        notes: List[str],
    ) -> None:
        nonlocal upgrade_score, wait_score, ops_score
        tags2 = sorted(set((tags or []) + ["non_cited_heuristic"]))

        evidence.append(
            {
                "rule_id": rule_id,
                "doc_id": NON_CITED_DOC,
                "loc": loc,
                "trigger_tags": tags2,
            }
        )
        upgrade_score += float(contrib.get("upgrade") or 0.0)
        wait_score += float(contrib.get("wait") or 0.0)
        ops_score += float(contrib.get("ops") or 0.0)

        _add_trace(
            kind="heuristic",
            rule_id=rule_id,
            doc_id=NON_CITED_DOC,
            loc=loc,
            trigger_tags=tags2,
            contributions={
                "upgrade": float(contrib.get("upgrade") or 0.0),
                "wait": float(contrib.get("wait") or 0.0),
                "ops": float(contrib.get("ops") or 0.0),
            },
            notes=notes,
            source="non_cited_heuristic",
        )

    # MW threshold heuristics (mutually exclusive, deterministic)
    try:
        mw = float(req.get("load_mw_total") or 0)
    except Exception:
        mw = 0.0

    if mw >= 500:
        _add_heuristic(
            rule_id="HEUR_LOAD_MW_GTE_500",
            loc="non-cited heuristic: very large load (MW total ≥ 500) increases upgrade exposure pressure",
            tags=["mw_threshold", "upgrade_exposure"],
            contrib={"upgrade": 2.0},
            notes=[f"load_mw_total={mw} => upgrade +2.0"],
        )
    elif mw >= 300:
        _add_heuristic(
            rule_id="HEUR_LOAD_MW_GTE_300",
            loc="non-cited heuristic: large load (MW total ≥ 300) increases upgrade exposure pressure",
            tags=["mw_threshold", "upgrade_exposure"],
            contrib={"upgrade": 1.0},
            notes=[f"load_mw_total={mw} => upgrade +1.0"],
        )

    if req.get("energization_plan") == "phased":
        _add_heuristic(
            rule_id="HEUR_PHASED_PLAN_REDUCES_UPGRADE_PRESSURE",
            loc="non-cited heuristic: phased energization plan can reduce upgrade exposure pressure in screening",
            tags=["phased_plan", "upgrade_exposure"],
            contrib={"upgrade": -0.5},
            notes=["energization_plan=phased => upgrade -0.5"],
        )

    # ── Batch timeline floor heuristic ──────────────────────────────────────────
    # Source: BATCH_007 criteria.text TIME ANCHORS block (Workshop Slides 22, 24).
    # This heuristic fires when batch_zero_eligible is explicitly provided, translating
    # the batch calendar dates into a study_results_floor_date that appears in risk_trace.
    # It does NOT add to wait_score (the published BATCH_001/007 rules already do that);
    # its sole purpose is to put a calendar anchor in the trace so memo outputs are not
    # just directional labels.  All dates are preliminary per Workshop Slide 2 disclaimer.
    _batch_eligible = req.get("batch_zero_eligible")
    if _batch_eligible is not None:
        from datetime import date as _date
        _today = _date.today()
        if _batch_eligible is True:
            # Batch Zero A: results by Jun 15 2026 (Workshop Slide 22)
            _study_floor = _date(2026, 6, 15)
            _study_floor_label = "Jun 15 2026 (Batch Zero A results; Workshop Slide 22)"
            _enrg_floor_label = "late 2026–early 2027 (study floor + 6–12 months for agreements/NTP/NOM per PG9_029/PG9_034/LLI_EN_011)"
            _scenario = "batch_zero_eligible=true → Batch Zero A"
        else:
            # batch_zero_eligible=False: earliest is Batch Zero B, results Jan 15 2027 (Workshop Slide 24)
            _study_floor = _date(2027, 1, 15)
            _study_floor_label = "Jan 15 2027 (Batch Zero B results; Workshop Slide 24)"
            _enrg_floor_label = "mid–late 2027 (study floor + 6–12 months for agreements/NTP/NOM per PG9_029/PG9_034/LLI_EN_011); later if IA not submitted before Aug 1 2026 deadline → Batch 1 (results Jul 31 2027 or Jan 31 2028)"
            _scenario = "batch_zero_eligible=false → Batch Zero B (earliest)"
        _months_to_floor = max(0, (_study_floor.year - _today.year) * 12 + (_study_floor.month - _today.month))
        _add_heuristic(
            rule_id="HEUR_BATCH_TIMELINE_FLOOR",
            loc=(
                "BATCH_007_BATCH_ZERO_B_COMMITMENT_DEADLINE criteria.text TIME ANCHORS block; "
                "source: ERCOT-Large-Load-Batch-Study-Workshop-02032026.pptx Slides 22, 24 (preliminary per Slide 2 disclaimer)"
            ),
            tags=["batch_study", "timeline_risk"],
            contrib={"wait": 0.0},  # no additional score; BATCH_001/007 already contribute wait; this is trace-only
            notes=[
                f"scenario={_scenario}",
                f"study_results_floor_date={_study_floor.isoformat()}",
                f"study_results_floor_label={_study_floor_label}",
                f"months_to_study_floor_from_{_today.isoformat()}={_months_to_floor}",
                f"energization_floor_estimate={_enrg_floor_label}",
                "NOTE: floor is study-process only; transmission upgrade construction (if triggered) extends COD further",
                "NOTE: all dates preliminary — Workshop Slide 2 disclaimer; update when ERCOT publishes final batch rules",
            ],
        )

    if missing_inputs:
        _add_heuristic(
            rule_id="HEUR_MISSING_INPUTS_PRESENT_INCREASES_WAIT_PRESSURE",
            loc="non-cited heuristic: missing inputs increase process waiting/latency pressure",
            tags=["data_completeness", "process_latency"],
            contrib={"wait": 0.5},
            notes=[f"missing_inputs_count={len(missing_inputs)} => wait +0.5"],
        )

    # Configuration heuristics (non-cited, explicitly labeled).
    try:
        poi_count = int(req.get("poi_count") or 0) if req.get("poi_count") is not None else 0
    except Exception:
        poi_count = 0
    if (req.get("multiple_poi") is True) or (poi_count > 1):
        _add_heuristic(
            rule_id="HEUR_MULTI_POI_INCREASES_COMPLEXITY",
            loc="non-cited heuristic: multiple POIs increase process/study coordination complexity in screening",
            tags=["process_latency", "study_dependency"],
            contrib={"wait": 1.0, "upgrade": 0.5},
            notes=[f"multiple_poi={req.get('multiple_poi')!r}, poi_count={poi_count} => wait +1.0, upgrade +0.5"],
        )
    if req.get("is_co_located") is True:
        _add_heuristic(
            rule_id="HEUR_CO_LOCATED_INCREASES_COORDINATION_RISK",
            loc="non-cited heuristic: co-located/shared infrastructure can increase coordination and operational exposure in screening",
            tags=["engineering_review", "process_latency"],
            contrib={"wait": 0.5, "ops": 0.5},
            notes=["is_co_located=true => wait +0.5, ops +0.5"],
        )
    exp = str(req.get("export_capability") or "").strip().lower()
    if exp in {"possible", "planned"}:
        _add_heuristic(
            rule_id="HEUR_EXPORT_CAPABILITY_INCREASES_OPS_EXPOSURE",
            loc="non-cited heuristic: export capability posture can increase operational exposure and study requirements in screening",
            tags=["operational_constraint", "study_dependency"],
            contrib={"ops": 1.0, "upgrade": 0.5, "wait": 0.5},
            notes=[f"export_capability={exp} => ops +1.0, upgrade +0.5, wait +0.5"],
        )
    if str(req.get("poi_voltage_class") or "").strip().lower() == "distribution":
        _add_heuristic(
            rule_id="HEUR_DISTRIBUTION_VOLTAGE_LIMITS_OPTIONS",
            loc="non-cited heuristic: distribution-voltage POI can constrain upgrade and process options vs transmission-voltage POI in screening",
            tags=["voltage_selection_risk", "process_latency"],
            contrib={"upgrade": 0.5, "wait": 0.5},
            notes=["poi_voltage_class=distribution => upgrade +0.5, wait +0.5"],
        )

    # Context snapshot signals (system state / discretion), if provided.
    def _context_multiplier(v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return 1.0 if float(v) != 0.0 else 0.0
        s = str(v).strip().lower()
        if s in {"", "unknown", "n/a", "na", "null"}:
            return 0.0
        if s == "low":
            return 0.5
        if s in {"medium", "med"}:
            return 1.0
        if s in {"high", "severe"}:
            return 2.0
        if s in {"true", "yes"}:
            return 1.0
        if s in {"false", "no"}:
            return 0.0
        return 1.0

    if isinstance(context_snapshot, dict) and isinstance(context_snapshot.get("signals"), list):
        as_of = str(context_snapshot.get("as_of") or "").strip()
        for sig in context_snapshot.get("signals") or []:
            if not isinstance(sig, dict):
                continue
            sid = str(sig.get("signal_id") or "").strip()
            typ = str(sig.get("type") or "").strip()
            if not sid or not typ:
                continue
            val = sig.get("value")
            conf = sig.get("confidence")
            src = str(sig.get("source") or "").strip()
            note_txt = str(sig.get("notes") or "").strip()

            base = TAG_RISK_CONTRIB.get(typ)
            mult = _context_multiplier(val)
            contrib = {"upgrade": 0.0, "wait": 0.0, "ops": 0.0}
            notes2: List[str] = [f"value={val!r}", f"multiplier={mult:g}"]
            if as_of:
                notes2.append(f"as_of={as_of}")
            if src:
                notes2.append(f"source={src}")
            if conf is not None:
                try:
                    notes2.append(f"confidence={float(conf):g}")
                except Exception:
                    notes2.append(f"confidence={conf}")
            if note_txt:
                notes2.append(f"notes={note_txt}")

            if base and mult != 0.0:
                for k, dv in base.items():
                    contrib[k] = float(contrib.get(k, 0.0)) + float(dv) * float(mult)
                notes2.append(f"mapped via TAG_RISK_CONTRIB[{typ!r}]")
            else:
                if not base:
                    notes2.append("type not mapped to v0 risk buckets (recorded for auditability only)")
                else:
                    notes2.append("multiplier=0 => no v0 risk effect")

            ctx_rule_id = f"CTX_{sid}"
            ctx_loc_parts = []
            if as_of:
                ctx_loc_parts.append(f"as_of={as_of}")
            if src:
                ctx_loc_parts.append(f"source={src}")
            if note_txt:
                ctx_loc_parts.append(note_txt)
            ctx_loc = "; ".join(ctx_loc_parts) if ctx_loc_parts else f"signal_id={sid}"

            evidence.append(
                {
                    "rule_id": ctx_rule_id,
                    "doc_id": "CONTEXT_SNAPSHOT",
                    "loc": ctx_loc,
                    "trigger_tags": sorted(set([typ])),
                }
            )
            upgrade_score += float(contrib.get("upgrade") or 0.0)
            wait_score += float(contrib.get("wait") or 0.0)
            ops_score += float(contrib.get("ops") or 0.0)
            _add_trace(
                kind="context",
                rule_id=ctx_rule_id,
                doc_id="CONTEXT_SNAPSHOT",
                loc=ctx_loc,
                trigger_tags=sorted(set([typ])),
                contributions=contrib,
                notes=notes2,
                source="context_snapshot",
            )

    # 3) Upgrade/ops exposure bucket (deterministic, based on INCREMENTAL score totals only)
    #
    # Thresholds are calibrated against incremental-only scores (baseline tags excluded).
    # With baseline tags removed, typical incremental ranges are:
    #   - upgrade: 0–8  (simple project ~1; complex/Far West ~6+)
    #   - ops: 0–5      (no curtailment risk ~0; export+energization ~4+)
    #   - wait: 0–15    (simple ~2; restudy+queue ~10+)
    exposure = "unknown"
    operational_exposure = "unknown"

    reasoning: List[str] = []
    reasoning.append(f"baseline_scores: upgrade={baseline_upgrade:g}, wait={baseline_wait:g}, ops={baseline_ops:g} (tracked but excluded from bucket thresholds)")
    if signal_rules:
        if upgrade_score >= 5:
            exposure = "high"
            reasoning.append(f"upgrade_score={upgrade_score:g} >= 5 => upgrade_exposure_bucket=high")
        elif upgrade_score >= 2:
            exposure = "medium"
            reasoning.append(f"upgrade_score={upgrade_score:g} >= 2 => upgrade_exposure_bucket=medium")
        else:
            exposure = "low"
            reasoning.append(f"upgrade_score={upgrade_score:g} < 2 => upgrade_exposure_bucket=low")

        status = "rule_driven_v0"
    else:
        status = "insufficient_rule_coverage"
        reasoning.append("no published rules matched/flagged => insufficient_rule_coverage (heuristics may still apply)")

    if signal_rules:
        if ops_score >= 6:
            operational_exposure = "high"
        elif ops_score >= 3:
            operational_exposure = "medium"
        else:
            operational_exposure = "low"
        reasoning.append(f"ops_score={ops_score:g} => operational_exposure_bucket={operational_exposure}")

    # 4) timeline_estimate — source-anchored calendar estimate replacing the
    #    old directional timeline_buckets (le_12/m12_24/gt_24 up/down/unknown).
    #
    #    Structure: every field carries a `source` string citing a rule_id or doc+loc.
    #    Reads HEUR_BATCH_TIMELINE_FLOOR notes that were injected above (when
    #    batch_zero_eligible is present in the request).  When batch_zero_eligible
    #    is absent the estimate is marked "unanchored" — the old directional labels
    #    become caveats so nothing is silently dropped.
    #
    #    Post-study delay sources:
    #      • PG9_029  — 180-day agreements window after LLIS (Planning Guide §9.4(9))
    #      • PG9_034  — Notice to Proceed (NTP) required before construction (§9.5.1(iii))
    #      • LLI_EN_011_NETWORK_MODEL — NOM inclusion required before energization
    #
    _te_study_floor_date: Optional[str] = None
    _te_study_floor_source: str = ""
    _te_study_floor_scenario: str = ""
    _te_months_to_floor: Optional[int] = None
    _te_earliest_cod_quarter: Optional[str] = None
    _te_caveats: List[str] = []

    # Pull calendar anchor from HEUR_BATCH_TIMELINE_FLOOR notes (if it fired)
    for _rt in risk_trace:
        if not isinstance(_rt, dict):
            continue
        if _rt.get("rule_id") != "HEUR_BATCH_TIMELINE_FLOOR":
            continue
        for _note in (_rt.get("notes") or []):
            _n = str(_note)
            if _n.startswith("study_results_floor_date="):
                _te_study_floor_date = _n.split("=", 1)[1].strip()
            elif _n.startswith("scenario="):
                _te_study_floor_scenario = _n.split("=", 1)[1].strip()
            elif _n.startswith("months_to_study_floor_from_"):
                try:
                    _te_months_to_floor = int(_n.split("=", 1)[1].strip())
                except Exception:
                    pass
        break

    # Derive earliest_cod_quarter from floor date + post-study window (6–12 months)
    # post-study floor sources: PG9_029 (180-day agreements), PG9_034 (NTP), LLI_EN_011 (NOM)
    _POST_STUDY_MIN_MONTHS = 6    # 180 days per PG9_029/PG9_034/LLI_EN_011
    _POST_STUDY_MAX_MONTHS = 12   # upper-end heuristic (construction + NOM review time)
    _POST_STUDY_SOURCE = (
        "PG9_029_CANCELLATION_RISK_IF_SECTION_9_5_NOT_SATISFIED_180_DAYS_AGREEMENTS "
        "(PG §9.4(9), 180-day window) + "
        "PG9_034_NOTICE_TO_PROCEED_INTERCONNECTION_FACILITIES_REQUIRED "
        "(PG §9.5.1(iii)) + "
        "LLI_EN_011_NETWORK_MODEL (NOM inclusion)"
    )

    if _te_study_floor_date and _te_months_to_floor is not None:
        _te_status = "anchored"
        _te_study_floor_source = (
            "HEUR_BATCH_TIMELINE_FLOOR → BATCH_007_BATCH_ZERO_B_COMMITMENT_DEADLINE "
            "TIME ANCHORS; source: ERCOT-Large-Load-Batch-Study-Workshop-02032026.pptx "
            "Slides 22, 24 (preliminary per Slide 2 disclaimer)"
        )
        # Confidence is "medium" when batch-anchored (dates are published but preliminary);
        # upgrades to "medium-high" when missing inputs are zero and no material-change
        # restudy risk; stays at "low" if critical inputs are missing.
        _missing_count = len(missing_inputs) if isinstance(missing_inputs, list) else 0
        _has_material_change = bool(req.get("material_change_flags"))
        if _missing_count == 0 and not _has_material_change:
            _te_confidence = "medium-high"
        elif _missing_count <= 2:
            _te_confidence = "medium"
        else:
            _te_confidence = "low"
        _te_caveats = [
            "All batch dates are preliminary (Workshop Slide 2 disclaimer); update when ERCOT publishes final batch rules",
            "Post-study estimate (+6–12 months) excludes transmission upgrade construction time if upgrades are triggered",
            "PUCT 58481 IA deadline (Aug 1 2026) still in draft — Batch Zero B inclusion may shift",
        ]
        if _has_material_change:
            _te_caveats.append("material_change_flags present — restudy may delay timeline beyond floor estimate")
        if _missing_count > 0:
            _te_caveats.append(f"missing_inputs_count={_missing_count} — incomplete data reduces confidence")
        # Compute earliest_cod_quarter from study floor + min post-study delay
        try:
            from datetime import date as _d
            _sf = _d.fromisoformat(_te_study_floor_date)
            _cod_month_earliest = _sf.month + _POST_STUDY_MIN_MONTHS
            _cod_year = _sf.year + (_cod_month_earliest - 1) // 12
            _cod_month = ((_cod_month_earliest - 1) % 12) + 1
            _cod_q = (_cod_month - 1) // 3 + 1
            _te_earliest_cod_quarter = f"{_cod_year}Q{_cod_q}"
        except Exception:
            _te_earliest_cod_quarter = None
    else:
        _te_status = "unanchored"
        _te_study_floor_source = ""
        _te_study_floor_scenario = "batch_zero_eligible not provided — batch track unknown"
        _te_confidence = "low"
        _te_caveats = [
            "batch_zero_eligible not provided; study floor not calculable — provide batch qualification status to anchor timeline",
            "Without batch anchor, upgrade_score and wait_score directional signals are the only available indicators",
            f"upgrade_score={upgrade_score:g} ({exposure} upgrade exposure); wait_score={wait_score:g}",
        ]
        # Derive a loose narrative from scores so the unanchored case is not empty
        if upgrade_score >= 5:
            _te_earliest_cod_quarter = None  # can't estimate without batch date
            _te_caveats.insert(0, "High upgrade exposure detected (upgrade_score >= 5); expect >24 months if batch placement is Batch 1 or later")
        elif upgrade_score >= 2 or wait_score >= 2:
            _te_caveats.insert(0, "Medium upgrade or elevated wait pressure; timeline likely 12–24 months minimum once batch placement is known")

    timeline_estimate: Dict[str, Any] = {
        "study_results_floor_date": _te_study_floor_date,
        "study_results_floor_source": _te_study_floor_source,
        "study_results_floor_scenario": _te_study_floor_scenario,
        "months_to_study_floor": _te_months_to_floor,
        "post_study_min_months": _POST_STUDY_MIN_MONTHS,
        "post_study_max_months": _POST_STUDY_MAX_MONTHS,
        "post_study_source": _POST_STUDY_SOURCE,
        "earliest_cod_quarter": _te_earliest_cod_quarter,
        "confidence": _te_confidence,
        "status": _te_status,
        "caveats": _te_caveats,
    }
    reasoning.append(
        f"timeline_estimate.status={_te_status}; "
        f"study_floor={_te_study_floor_date or 'n/a'}; "
        f"earliest_cod_quarter={_te_earliest_cod_quarter or 'n/a'}; "
        f"confidence={_te_confidence}"
    )

    risk_trace.append(
        {
            "kind": "summary",
            "upgrade_score": upgrade_score,
            "wait_score": wait_score,
            "ops_score": ops_score,
            "baseline_upgrade": baseline_upgrade,
            "baseline_wait": baseline_wait,
            "baseline_ops": baseline_ops,
            "status": status,
            "timeline_estimate": timeline_estimate,
            "upgrade_exposure_bucket": exposure,
            "operational_exposure_bucket": operational_exposure,
            "reasoning": reasoning,
        }
    )

    return (
        {
            "timeline_estimate": timeline_estimate,
            "upgrade_exposure_bucket": exposure,
            "operational_exposure_bucket": operational_exposure,
            "status": status,
            "evidence": evidence,
            "baseline_upgrade": baseline_upgrade,
            "baseline_wait": baseline_wait,
            "baseline_ops": baseline_ops,
            "upgrade_score": upgrade_score,
            "wait_score": wait_score,
            "ops_score": ops_score,
        },
        risk_trace,
    )

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

    # Configuration levers (derive when safe; explicit values win).
    if ("multiple_poi" not in r) and isinstance(r.get("poi_count"), int):
        try:
            r["multiple_poi"] = int(r.get("poi_count") or 0) > 1
        except Exception:
            pass

    # Treat a singleton voltage_options_kv as an implied current choice (for voltage-class branching),
    # unless the request already specified an explicit voltage_choice_kv.
    if "voltage_choice_kv" not in r:
        vopts = r.get("voltage_options_kv")
        if isinstance(vopts, list) and len(vopts) == 1:
            try:
                r["voltage_choice_kv"] = float(vopts[0])
            except Exception:
                r["voltage_choice_kv"] = None

    # Voltage class: explicit poi_voltage_class wins; otherwise derive from voltage_choice_kv when numeric.
    # If voltage_choice_kv is not available, fall back to voltage_options_kv when it is unambiguous
    # (e.g., all candidate options are transmission-voltage, or all are distribution-voltage).
    if "poi_voltage_class" not in r:
        v = r.get("voltage_choice_kv")
        cls = "unknown"
        try:
            v_num = float(v) if v is not None else None
        except Exception:
            v_num = None
        if v_num is not None:
            # v0 threshold aligned with certain TDSP transmission-scope standards (e.g., 69kV+).
            cls = "transmission" if v_num >= 69.0 else "distribution"
        else:
            vopts = r.get("voltage_options_kv")
            if isinstance(vopts, list) and vopts:
                nums = []
                for vv in vopts:
                    try:
                        nums.append(float(vv))
                    except Exception:
                        nums = []
                        break
                if nums:
                    if min(nums) >= 69.0:
                        cls = "transmission"
                    elif max(nums) < 69.0:
                        cls = "distribution"
        r["poi_voltage_class"] = cls

    # Convenience applicability fields (derived, deterministic) for conditional_gate modeling.
    # These let us express "not_applicable vs unknown" without forcing users to provide extra toggles.
    pvc = str(r.get("poi_voltage_class") or "").strip().lower()
    if pvc in {"transmission", "distribution"}:
        r.setdefault("poi_voltage_class_is_transmission", pvc == "transmission")
        r.setdefault("poi_voltage_class_is_distribution", pvc == "distribution")
    else:
        # unknown / not provided
        r.setdefault("poi_voltage_class_is_transmission", None)
        r.setdefault("poi_voltage_class_is_distribution", None)
    return r


def normalize_context_snapshot(ctx: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize optional request.context into a stable shape for evaluation + memo rendering.
    """
    if not isinstance(ctx, dict):
        return None
    as_of = str(ctx.get("as_of") or "").strip()
    sigs_in = ctx.get("signals")
    if not isinstance(sigs_in, list):
        sigs_in = []
    sigs_out: List[Dict[str, Any]] = []
    for s in sigs_in:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("signal_id") or "").strip()
        typ = str(s.get("type") or "").strip()
        if not sid or not typ:
            continue
        sigs_out.append(
            {
                "signal_id": sid,
                "type": typ,
                "value": s.get("value"),
                "confidence": s.get("confidence"),
                "source": s.get("source"),
                "notes": s.get("notes"),
            }
        )
    if not sigs_out and not as_of:
        return None
    return {"as_of": as_of, "signals": sigs_out}


def has_gate_predicate(rule: Dict[str, Any], *, field: str, value: Any = True) -> bool:
    for p in rule_predicates(rule):
        op = p.get("op")
        if op == "eq":
            op = "equals"
        if op == "equals" and p.get("field") == field and p.get("value") == value:
            return True
    return False


def is_energization_mode(req: Dict[str, Any]) -> bool:
    """
    Determine whether to evaluate energization readiness in v0.

    v0 triggers energization-mode evaluation when either:
    - is_requesting_energization == true, OR
    - requested_energization_window is provided (readiness vs target window)
    """
    if req.get("is_requesting_energization") is True:
        return True
    w = req.get("requested_energization_window")
    return isinstance(w, str) and bool(w.strip())


def normalize_memo_guidance(rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize optional rule.memo_guidance into a stable, memo-friendly shape.

    This is NOT citation-audited; it is purely human process guidance.
    """
    mg = rule.get("memo_guidance")
    if not isinstance(mg, dict):
        return None

    # Policy: only render guidance that is explicitly marked verified.
    verified = bool(mg.get("verified") is True)
    if not verified:
        return None

    sources_in = mg.get("sources") or []
    sources: List[Dict[str, Any]] = []
    if isinstance(sources_in, list):
        for s in sources_in:
            if not isinstance(s, dict):
                continue
            doc_id = str(s.get("doc_id") or "").strip()
            loc = str(s.get("loc") or "").strip()
            notes = str(s.get("notes") or "").strip()
            if not doc_id or not loc:
                continue
            sources.append({"doc_id": doc_id, "loc": loc, "notes": notes})

    owner = str(mg.get("owner") or "").strip()
    recipient = str(mg.get("recipient") or "").strip()
    what_to_do = str(mg.get("what_to_do") or "").strip()

    ev = mg.get("evidence") or []
    ev2: List[str] = []
    if isinstance(ev, list):
        for x in ev:
            s = str(x).strip()
            if s:
                ev2.append(s)

    # Dedupe evidence while preserving order (deterministic).
    seen = set()
    ev3: List[str] = []
    for s in ev2:
        if s in seen:
            continue
        seen.add(s)
        ev3.append(s)

    out = {"verified": True, "sources": sources, "owner": owner, "recipient": recipient, "evidence": ev3, "what_to_do": what_to_do}
    if (not owner) and (not recipient) and (not ev3) and (not what_to_do):
        return None
    return out


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
    criteria_text = str(((rule.get("criteria") or {}).get("text") or "")).strip()
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    is_completeness = dsl.get("kind") == "completeness_check"
    dsl_kind = str(dsl.get("kind") or "")
    applicability_fields = dsl.get("applicability_fields") or []
    if not isinstance(applicability_fields, list):
        applicability_fields = []

    # If rule is explicitly gated on energization, treat it as not applicable unless requesting energization.
    if has_gate_predicate(rule, field="is_requesting_energization", value=True) and (not is_energization_mode(req)):
        return {
            "rule_id": rid,
            "doc_id": rule.get("doc_id"),
            "loc": rule.get("loc"),
            "criteria_text": criteria_text,
            "trigger_tags": rule.get("trigger_tags") or [],
            "confidence": rule.get("confidence"),
            "dsl_kind": dsl_kind,
            "required_fields": required,
            "applicability_fields": applicability_fields,
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
                # Applicability toggles are tri-state in v0:
                # - False => not_applicable
                # - missing / null / "unknown" => unknown (do NOT count as missing_fields)
                # - True => gate is applicable; proceed with normal predicate + required_fields evaluation
                if field not in req:
                    v_raw = None
                    why = "absent"
                else:
                    v_raw = req.get(field)
                    why = "null" if v_raw is None else "provided"
                v = get_field(req, field)
                if v is False:
                    return {
                        "rule_id": rid,
                        "doc_id": rule.get("doc_id"),
                        "loc": rule.get("loc"),
                        "criteria_text": criteria_text,
                        "trigger_tags": rule.get("trigger_tags") or [],
                        "confidence": rule.get("confidence"),
                        "dsl_kind": dsl_kind,
                        "required_fields": required,
                        "applicability_fields": applicability_fields,
                        "status": "not_applicable",
                        "missing_fields": [],
                        "traces": [f"{field}==True (False) => not_applicable"],
                    }
                if _is_explicit_unknown_value(v_raw) or _is_explicit_unknown_value(v):
                    return {
                        "rule_id": rid,
                        "doc_id": rule.get("doc_id"),
                        "loc": rule.get("loc"),
                        "criteria_text": criteria_text,
                        "trigger_tags": rule.get("trigger_tags") or [],
                        "confidence": rule.get("confidence"),
                        "dsl_kind": dsl_kind,
                        "required_fields": required,
                        "applicability_fields": applicability_fields,
                        "status": "unknown",
                        "missing_fields": [],
                        "traces": [f"{field} applicability unknown ({why}) => unknown (confirm applicability)"],
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
            "criteria_text": criteria_text,
            "trigger_tags": rule.get("trigger_tags") or [],
            "confidence": rule.get("confidence"),
            "dsl_kind": dsl_kind,
            "required_fields": required,
            "applicability_fields": applicability_fields,
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
        "criteria_text": criteria_text,
        "trigger_tags": rule.get("trigger_tags") or [],
        "confidence": rule.get("confidence"),
        "dsl_kind": dsl_kind,
        "required_fields": required,
        "applicability_fields": applicability_fields,
        "status": status,
        "missing_fields": missing_all,
        "traces": traces,
    }


def dedupe_checklist(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep the checklist compact and credible:
    - Drop rows that don't add actionable requirements (e.g., only gated on is_requesting_energization)
    - Deduplicate equivalent requirements across docs by preferring Planning Guide citations
    """
    doc_pref = {
        "ERCOT_PLANNING_GUIDE": 0,
        "ERCOT_LLI_ENERGIZATION_REQUEST": 1,
        "ERCOT_LARGE_LOAD_QA": 2,
        "ONCOR_STD_520_106": 3,
    }

    def _key(it: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        kind = str(it.get("dsl_kind") or "")
        req_fields = tuple(sorted([str(x) for x in (it.get("required_fields") or []) if x]))
        app_fields = tuple(sorted([str(x) for x in (it.get("applicability_fields") or []) if x]))
        return kind, app_fields, req_fields

    def _score(it: Dict[str, Any]) -> Tuple[int, float]:
        doc = str(it.get("doc_id") or "")
        pref = doc_pref.get(doc, 99)
        try:
            conf = float(it.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        # lower pref is better; higher confidence is better
        return pref, -conf

    toggles = {"is_requesting_energization", "sso_required", "dme_required", "is_large_electronic_load"}

    def _is_actionable(it: Dict[str, Any]) -> bool:
        kind = str(it.get("dsl_kind") or "")
        req_fields = [str(x) for x in (it.get("required_fields") or []) if x]
        app_fields = [str(x) for x in (it.get("applicability_fields") or []) if x]
        if it.get("status") == "not_applicable":
            return False
        # Input confirmations are better represented as missing inputs, not checklist rows.
        if kind == "input_confirmation":
            return False
        # Drop composite "roll-up" requirements to keep checklist concise.
        if len(req_fields) > 3:
            return False
        # If it only checks toggles (no real completion/condition), it's not actionable.
        non_toggle = [f for f in req_fields if f not in toggles and f not in app_fields]
        return len(non_toggle) > 0

    grouped: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Dict[str, Any]] = {}
    for it in items:
        if not _is_actionable(it):
            continue
        k = _key(it)
        cur = grouped.get(k)
        if cur is None or _score(it) < _score(cur):
            grouped[k] = it
    # Stable output: by doc preference then rule_id.
    out = list(grouped.values())
    out.sort(key=lambda x: (doc_pref.get(str(x.get("doc_id") or ""), 99), str(x.get("rule_id") or "")))
    return out


def _summarize_option_eval(ev: Dict[str, Any]) -> Dict[str, Any]:
    graph = ev.get("graph") or {}
    risk = ev.get("risk") or {}
    te = risk.get("timeline_estimate") or {}
    path = " → ".join(graph.get("path_node_labels") or [])

    # Energization readiness summary (when requested)
    req = ev.get("request") or {}
    checklist = ev.get("energization_checklist") or []
    counts = {"satisfied": 0, "missing": 0, "not_satisfied": 0, "unknown": 0, "not_applicable": 0}
    if isinstance(checklist, list):
        for it in checklist:
            if not isinstance(it, dict):
                continue
            s = str(it.get("status") or "").strip().lower() or "unknown"
            if s not in counts:
                s = "unknown"
            counts[s] += 1

    # Risk score totals (v0, explainable). These are used as tie-breakers in recommendation ranking.
    upgrade_score = 0.0
    wait_score = 0.0
    ops_score = 0.0
    baseline_upgrade = 0.0
    baseline_wait = 0.0
    baseline_ops = 0.0
    for rt in (ev.get("risk_trace") or []):
        if not isinstance(rt, dict):
            continue
        if rt.get("kind") != "summary":
            continue
        try:
            upgrade_score = float(rt.get("upgrade_score") or 0.0)
            wait_score = float(rt.get("wait_score") or 0.0)
            ops_score = float(rt.get("ops_score") or 0.0)
            baseline_upgrade = float(rt.get("baseline_upgrade") or 0.0)
            baseline_wait = float(rt.get("baseline_wait") or 0.0)
            baseline_ops = float(rt.get("baseline_ops") or 0.0)
        except Exception:
            upgrade_score = wait_score = ops_score = 0.0
            baseline_upgrade = baseline_wait = baseline_ops = 0.0
        break

    def _round3(x: float) -> float:
        try:
            return float(round(float(x), 3))
        except Exception:
            return 0.0

    return {
        "path": path,
        "missing_inputs_count": len(ev.get("missing_inputs") or []),
        "missing_inputs": ev.get("missing_inputs") or [],
        "flags_count": len(ev.get("flags") or []),
        "energization": {
            "requested": bool(is_energization_mode(req)),
            "checklist_counts": counts,
        },
        "risk_scores": {
            "upgrade_score": _round3(upgrade_score),
            "wait_score": _round3(wait_score),
            "ops_score": _round3(ops_score),
            "baseline_upgrade": _round3(baseline_upgrade),
            "baseline_wait": _round3(baseline_wait),
            "baseline_ops": _round3(baseline_ops),
        },
        "timeline_estimate": {
            "study_results_floor_date": te.get("study_results_floor_date"),
            "study_results_floor_scenario": te.get("study_results_floor_scenario"),
            "months_to_study_floor": te.get("months_to_study_floor"),
            "earliest_cod_quarter": te.get("earliest_cod_quarter"),
            "status": te.get("status", "unanchored"),
            "confidence": te.get("confidence", "low"),
        },
        "upgrade_exposure_bucket": risk.get("upgrade_exposure_bucket", "unknown"),
        "operational_exposure_bucket": risk.get("operational_exposure_bucket", "unknown"),
    }


def _diff_option_summaries(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    b_path = str(base.get("path") or "")
    o_path = str(other.get("path") or "")
    b_miss = int(base.get("missing_inputs_count") or 0)
    o_miss = int(other.get("missing_inputs_count") or 0)
    b_flags = int(base.get("flags_count") or 0)
    o_flags = int(other.get("flags_count") or 0)

    b_te = base.get("timeline_estimate") or {}
    o_te = other.get("timeline_estimate") or {}

    def _te_chg(k: str) -> Dict[str, Any]:
        return {"from": b_te.get(k), "to": o_te.get(k)}

    return {
        "path_changed": (b_path != o_path),
        "missing_inputs_count_delta": o_miss - b_miss,
        "flags_count_delta": o_flags - b_flags,
        "timeline_estimate_changed": {
            "status": _te_chg("status"),
            "study_results_floor_date": _te_chg("study_results_floor_date"),
            "months_to_study_floor": _te_chg("months_to_study_floor"),
            "earliest_cod_quarter": _te_chg("earliest_cod_quarter"),
        },
        "upgrade_exposure_bucket_changed": {
            "from": str(base.get("upgrade_exposure_bucket", "unknown")),
            "to": str(other.get("upgrade_exposure_bucket", "unknown")),
        },
        "operational_exposure_bucket_changed": {
            "from": str(base.get("operational_exposure_bucket", "unknown")),
            "to": str(other.get("operational_exposure_bucket", "unknown")),
        },
    }


def _edge_id_list(ev: Dict[str, Any]) -> List[str]:
    graph = ev.get("graph") or {}
    edges = graph.get("traversed_edges") or []
    out: List[str] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("edge_id") or "").strip()
        if eid:
            out.append(eid)
    return out


def _fields_referenced_by_predicates(preds: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(preds, list):
        return out
    for p in preds:
        if not isinstance(p, dict):
            continue
        f = p.get("field")
        if not f:
            continue
        out.append(str(f))
    # stable unique preserving order
    seen = set()
    uniq: List[str] = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _rule_citations_for_rule_ids(rule_ids: Any, rules_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(rule_ids, list):
        return out
    for rid in rule_ids:
        rid_s = str(rid or "").strip()
        if not rid_s:
            continue
        r = rules_by_id.get(rid_s)
        if not r:
            out.append({"rule_id": rid_s, "doc_id": None, "loc": None})
            continue
        out.append({"rule_id": rid_s, "doc_id": r.get("doc_id"), "loc": r.get("loc")})
    return out


def _compute_edge_diff_for_option(
    *,
    baseline_ev: Dict[str, Any],
    option_ev: Dict[str, Any],
    graph_edges_by_id: Dict[str, Dict[str, Any]],
    rules_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute a compact, memo-friendly edge diff between baseline and an option.

    Goal: make "path_changed" auditable by showing exactly which edge(s) differ, why the branch changed
    (predicate traces + relevant request field deltas), and the edge-level citations (rule_id/doc_id/loc).
    """
    b_graph = baseline_ev.get("graph") or {}
    o_graph = option_ev.get("graph") or {}
    b_edges = b_graph.get("traversed_edges") or []
    o_edges = o_graph.get("traversed_edges") or []
    if not isinstance(b_edges, list):
        b_edges = []
    if not isinstance(o_edges, list):
        o_edges = []

    b_ids = _edge_id_list(baseline_ev)
    o_ids = _edge_id_list(option_ev)

    # common prefix
    i = 0
    n = min(len(b_ids), len(o_ids))
    while i < n and b_ids[i] == o_ids[i]:
        i += 1

    # common suffix (avoid overlap with prefix)
    j = 0
    while (j < (len(b_ids) - i)) and (j < (len(o_ids) - i)) and (b_ids[-1 - j] == o_ids[-1 - j]):
        j += 1

    b_mid_edges = b_edges[i : (len(b_edges) - j) if j else len(b_edges)]
    o_mid_edges = o_edges[i : (len(o_edges) - j) if j else len(o_edges)]

    b_req = baseline_ev.get("request") or {}
    o_req = option_ev.get("request") or {}

    def _compact_edge(e: Dict[str, Any], *, req_a: Dict[str, Any], req_b: Dict[str, Any]) -> Dict[str, Any]:
        eid = str(e.get("edge_id") or "")
        ge = graph_edges_by_id.get(eid) or {}
        criteria = ge.get("criteria") or []
        fields = _fields_referenced_by_predicates(criteria)
        field_deltas = []
        for f in fields:
            va = get_field(req_a, f)
            vb = get_field(req_b, f)
            if va != vb:
                field_deltas.append({"field": f, "from": va, "to": vb})
        rule_ids = e.get("rule_ids") or []
        return {
            "edge_id": eid,
            "from": e.get("from"),
            "to": e.get("to"),
            "criteria_traces": e.get("criteria_traces") or [],
            "criteria_fields": fields,
            "criteria_field_deltas": field_deltas,
            "rule_ids": rule_ids,
            "rule_citations": _rule_citations_for_rule_ids(rule_ids, rules_by_id),
        }

    b_mid = [_compact_edge(e, req_a=b_req, req_b=o_req) for e in b_mid_edges if isinstance(e, dict)]
    o_mid = [_compact_edge(e, req_a=b_req, req_b=o_req) for e in o_mid_edges if isinstance(e, dict)]

    return {
        "path_same": (b_ids == o_ids),
        "edge_ids_baseline": b_ids,
        "edge_ids_option": o_ids,
        "common_prefix_edge_ids": b_ids[:i],
        "common_suffix_edge_ids": (b_ids[-j:] if j else []),
        "diff_window": {
            "start_index": i,
            "baseline_segment_len": len(b_mid),
            "option_segment_len": len(o_mid),
        },
        "baseline_segment": b_mid,
        "option_segment": o_mid,
    }


def _fields_referenced_by_rule(rule: Dict[str, Any]) -> set[str]:
    dsl = ((rule.get("criteria") or {}).get("dsl") or {})
    fields: set[str] = set()
    if isinstance(dsl, dict):
        rf = dsl.get("required_fields") or []
        if isinstance(rf, list):
            fields |= {str(x) for x in rf if x}
        af = dsl.get("applicability_fields") or []
        if isinstance(af, list):
            fields |= {str(x) for x in af if x}
        preds = dsl.get("predicates") or []
        if isinstance(preds, list):
            for p in preds:
                if isinstance(p, dict) and p.get("field"):
                    fields.add(str(p.get("field")))
    return fields


def _analyze_levers(req: Dict[str, Any], graph: Dict[str, Any], rules_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    levers = graph.get("levers_catalog") or []
    if not isinstance(levers, list):
        return []

    allowed_classes = {"hard", "design", "external", "unknown"}

    out: List[Dict[str, Any]] = []
    for lv in levers:
        if not isinstance(lv, dict) or not lv.get("lever_id"):
            continue
        lever_id = str(lv.get("lever_id"))
        lever_class = str(lv.get("class") or lv.get("lever_class") or "unknown").strip().lower()
        if lever_class not in allowed_classes:
            lever_class = "unknown"
        label = str(lv.get("label") or "")
        req_fields = [str(x) for x in (lv.get("request_fields") or []) if x]
        # Presence is “exists” semantics (same as evaluator): None / "" / [] are treated as missing.
        present = []
        missing = []
        for f in req_fields:
            v = get_field(req, f)
            ok = (v is not None) and (v != "") and (v != [])
            (present if ok else missing).append(f)

        refs = []
        for rid, r in rules_by_id.items():
            if rid and (set(req_fields) & _fields_referenced_by_rule(r)):
                refs.append(str(rid))
        refs = sorted(set(refs))

        out.append(
            {
                "lever_id": lever_id,
                "lever_class": lever_class,
                "label": label,
                "request_fields": req_fields,
                "present_fields": present,
                "missing_fields": missing,
                "referenced_rule_ids": refs,
                "referenced_rule_count": len(refs),
            }
        )
    return out


def _bucket_rank(v: str) -> int:
    """
    Lower is better.
    """
    s = (v or "").strip().lower()
    if s == "low":
        return 0
    if s == "medium":
        return 1
    if s == "high":
        return 2
    return 3  # unknown/other


def _timeline_score(te: Dict[str, Any]) -> int:
    """
    Lower is better. Converts timeline_estimate into a ranking integer.
    Anchored estimates with longer study floors score higher (worse rank).
    Unanchored high-upgrade-exposure estimates also score higher.
    Intentionally simple and explainable.
    """
    status = str(te.get("status") or "unanchored").lower()
    months = te.get("months_to_study_floor")
    score = 0
    if status == "anchored" and months is not None:
        try:
            m = int(months)
        except Exception:
            m = 0
        if m > 24:
            score += 3
        elif m > 12:
            score += 2
        elif m > 6:
            score += 1
    else:
        # Unanchored: use confidence as a mild tiebreaker — low confidence = slight penalty
        confidence = str(te.get("confidence") or "low").lower()
        if confidence == "low":
            score += 1
    return score


def _top_risk_drivers(risk_trace: List[Dict[str, Any]], *, top_n: int = 6) -> List[Dict[str, Any]]:
    items = [x for x in (risk_trace or []) if isinstance(x, dict) and x.get("kind") in {"rule", "heuristic", "context"}]
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for x in items:
        c = x.get("contributions") or {}
        try:
            up = float(c.get("upgrade") or 0.0)
            wt = float(c.get("wait") or 0.0)
            op = float(c.get("ops") or 0.0)
        except Exception:
            up = wt = op = 0.0
        score = abs(up) + abs(wt) + abs(op)
        scored.append((score, x))
    scored.sort(key=lambda t: (-t[0], str(t[1].get("rule_id") or "")))
    out: List[Dict[str, Any]] = []
    for s, x in scored[: max(0, int(top_n))]:
        out.append(
            {
                "kind": x.get("kind"),
                "source": x.get("source"),
                "rule_id": x.get("rule_id"),
                "doc_id": x.get("doc_id"),
                "loc": x.get("loc"),
                "trigger_tags": x.get("trigger_tags") or [],
                "contributions": x.get("contributions") or {},
                "notes": x.get("notes") or [],
                "score_abs": s,
            }
        )
    return out


def _recommendation_from_candidates(
    *,
    baseline: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    req_is_energization: bool,
) -> Dict[str, Any]:
    """
    Bounded recommendation within available candidates (baseline + options).
    """
    def _score(s: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        miss = int(s.get("missing_inputs_count") or 0)
        en = s.get("energization") or {}
        cc = en.get("checklist_counts") or {}
        not_sat = int(cc.get("not_satisfied") or 0)
        missing = int(cc.get("missing") or 0)
        # readiness: lower not_sat/missing is better when energization is requested
        readiness_penalty = (not_sat * 2 + missing) if req_is_energization else 0

        rs = s.get("risk_scores") or {}
        try:
            up_sc = float(rs.get("upgrade_score") or 0.0)
            wt_sc = float(rs.get("wait_score") or 0.0)
            op_sc = float(rs.get("ops_score") or 0.0)
        except Exception:
            up_sc = wt_sc = op_sc = 0.0

        tb = s.get("timeline_estimate") or {}
        timeline = _timeline_score(tb)
        up = _bucket_rank(str(s.get("upgrade_exposure_bucket") or "unknown"))
        ops = _bucket_rank(str(s.get("operational_exposure_bucket") or "unknown"))
        # NOTE: risk score totals are used as a tiebreaker so lever-driven rules can
        # affect recommendations even when qualitative buckets don't change.
        risk_pressure = up_sc + wt_sc + op_sc
        return (miss, readiness_penalty, up, ops, int(round(risk_pressure * 1000)), timeline)

    ranked = []
    for c in candidates:
        ranked.append((_score(c.get("summary") or {}), c))

    def _source_priority(cand: Dict[str, Any]) -> int:
        src = str(cand.get("source") or "").strip()
        if not src and str(cand.get("option_id") or "") == "baseline":
            src = "baseline"
        if src == "user_provided_alternatives":
            return 0
        if src == "baseline":
            return 1
        if src == "system_generated_toggle":
            return 2
        return 3

    # Deterministic ordering:
    # 1) score tuple (lower is better)
    # 2) prefer user-provided alternatives over baseline over system-generated toggles
    # 3) stable option_id tie-break
    ranked.sort(key=lambda t: (t[0], _source_priority(t[1]), str(t[1].get("option_id") or "")))
    best = ranked[0][1] if ranked else None

    out = {
        "mode": "bounded_v0",
        "basis": {
            "primary": "minimize missing inputs",
            "then": "minimize energization checklist not_satisfied/missing" if req_is_energization else "then minimize bucket severity",
            "risk_tiebreak": "prefer lower upgrade/ops bucket and better timeline score",
            "context_note": "risk buckets include context snapshot adjustments when request.context is provided",
        },
        "recommended_option_id": (best.get("option_id") if isinstance(best, dict) else None),
        "recommended_is_baseline": bool(best and best.get("option_id") == "baseline"),
        "candidates_count": len(candidates),
        "candidates_ranked": [
            {
                "option_id": c.get("option_id"),
                "lever_id": c.get("lever_id"),
                "source": c.get("source"),
                "score": list(_score(c.get("summary") or {})),
                "summary": c.get("summary") or {},
            }
            for _sc, c in ranked[:8]
        ],
    }

    # Short rationale: compare best vs baseline
    if best and isinstance(best, dict):
        bsum = baseline.get("summary") or {}
        osum = best.get("summary") or {}
        out["rationale"] = [
            f"missing_inputs_count: {int(bsum.get('missing_inputs_count') or 0)} → {int(osum.get('missing_inputs_count') or 0)}",
            f"upgrade_bucket: {bsum.get('upgrade_exposure_bucket','unknown')} → {osum.get('upgrade_exposure_bucket','unknown')}",
            f"ops_bucket: {bsum.get('operational_exposure_bucket','unknown')} → {osum.get('operational_exposure_bucket','unknown')}",
            f"path: {'same' if str(bsum.get('path') or '') == str(osum.get('path') or '') else 'changed'}",
        ]
        if req_is_energization:
            bc = (bsum.get("energization") or {}).get("checklist_counts") or {}
            oc = (osum.get("energization") or {}).get("checklist_counts") or {}
            out["rationale"].append(f"energization not_satisfied: {int(bc.get('not_satisfied') or 0)} → {int(oc.get('not_satisfied') or 0)}")
    return out


def _decision_screening_status(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Screening decision: "intake readiness" style summary.
    """
    miss = ev.get("missing_inputs") or []
    miss_n = len(miss) if isinstance(miss, list) else 0
    out: Dict[str, Any] = {"mode": "screening", "status": "unknown", "reasons": []}
    if miss_n > 0:
        out["status"] = "conditional"
        out["reasons"].append(f"missing_inputs_count={miss_n}")
    else:
        out["status"] = "go"
        out["reasons"].append("no missing inputs detected")
    # Coverage hint
    risk = ev.get("risk") or {}
    if str(risk.get("status") or "") == "insufficient_rule_coverage":
        out["reasons"].append("insufficient_rule_coverage")
    return out


def _decision_energization_status(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Energization decision: readiness vs the requested/target energization window.
    """
    req = ev.get("request") or {}
    miss = ev.get("missing_inputs") or []
    miss_n = len(miss) if isinstance(miss, list) else 0
    out: Dict[str, Any] = {"mode": "energization", "status": "unknown", "reasons": []}

    if not is_energization_mode(req):
        out["status"] = "unknown"
        out["reasons"].append("energization_not_requested")
        return out

    w = req.get("requested_energization_window")
    if isinstance(w, str) and w.strip():
        out["reasons"].append(f"target_window={w.strip()}")

    # Energization mode: use checklist statuses
    checklist = ev.get("energization_checklist") or []
    miss_gate = 0
    not_sat_gate = 0
    unknown_gate = 0
    for it in checklist or []:
        if not isinstance(it, dict):
            continue
        s = str(it.get("status") or "").strip().lower()
        if s == "missing":
            miss_gate += 1
        elif s == "not_satisfied":
            not_sat_gate += 1
        elif s == "unknown":
            unknown_gate += 1

    if not checklist:
        out["status"] = "unknown"
        out["reasons"].append("no_checklist_rows")
        return out

    if (not_sat_gate == 0) and (miss_gate == 0) and (unknown_gate == 0) and (miss_n == 0):
        out["status"] = "ready"
        out["reasons"].append("all checklist gates satisfied")
    elif not_sat_gate > 0:
        out["status"] = "not_ready"
        out["reasons"].append(f"checklist_not_satisfied={not_sat_gate}")
    elif unknown_gate > 0 and (miss_gate == 0) and (not_sat_gate == 0):
        out["status"] = "unknown"
        out["reasons"].append(f"checklist_unknown={unknown_gate}")
        out["reasons"].append("confirm applicability for unknown gates")
        if miss_n:
            out["reasons"].append(f"missing_inputs_count={miss_n}")
    else:
        out["status"] = "conditional"
        out["reasons"].append(f"checklist_missing={miss_gate}")
        if unknown_gate:
            out["reasons"].append(f"checklist_unknown={unknown_gate}")
        if miss_n:
            out["reasons"].append(f"missing_inputs_count={miss_n}")
    return out


def _decision_status(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible single decision summary.
    """
    req = ev.get("request") or {}
    return _decision_energization_status(ev) if is_energization_mode(req) else _decision_screening_status(ev)


def _next_actions(ev: Dict[str, Any], *, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Memo-friendly next actions (bounded list).
    """
    req = ev.get("request") or {}
    is_en = is_energization_mode(req)
    actions: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _add(a: Dict[str, Any]) -> None:
        k = json.dumps(a, sort_keys=True, ensure_ascii=False)
        if k in seen_keys:
            return
        seen_keys.add(k)
        actions.append(a)

    # 1) Energization gates first (if requested)
    if is_en:
        for c in (ev.get("energization_checklist") or []):
            if not isinstance(c, dict):
                continue
            status = str(c.get("status") or "").strip().lower()
            if status not in {"missing", "not_satisfied"}:
                continue
            rid = str(c.get("rule_id") or "")
            if not rid:
                continue
            memo_guidance = c.get("memo_guidance") if isinstance(c.get("memo_guidance"), dict) else None
            missing_fields = [str(x) for x in (c.get("missing_fields") or []) if str(x).strip()]
            required_fields = [str(x) for x in (c.get("required_fields") or []) if str(x).strip()]
            traces = [str(x) for x in (c.get("traces") or []) if str(x).strip()]
            _add(
                {
                    "kind": "energization_gate",
                    "status": status,
                    "rule_id": rid,
                    "doc_id": c.get("doc_id"),
                    "loc": c.get("loc"),
                    "criteria_text": c.get("criteria_text"),
                    "ask_fields": missing_fields[:6],
                    "required_fields": required_fields[:12],
                    "conditions": traces[:4],
                    "memo_guidance": memo_guidance,
                }
            )
            if len(actions) >= max_items:
                return actions

    # 2) Missing inputs (fields)
    miss = ev.get("missing_inputs") or []
    miss2 = [str(x) for x in miss] if isinstance(miss, list) else []
    for f in miss2:
        if not f or f == "is_requesting_energization":
            continue
        _add({"kind": "provide_field", "field": f})
        if len(actions) >= max_items:
            break

    # 3) Strategic next actions — always relevant even when all inputs are complete.
    #    These keep the memo forward-looking rather than showing an empty list.
    risk = ev.get("risk") or {}
    te = risk.get("timeline_estimate") or {}

    # Suggest confirming batch_zero_eligible if not yet provided (timeline is unanchored)
    if te.get("status") == "unanchored" and req.get("batch_zero_eligible") is None:
        _add({
            "kind": "confirm_field",
            "field": "batch_zero_eligible",
            "rationale": "Timeline cannot be anchored without batch qualification status — confirm whether project qualifies for Batch Zero.",
            "source": "BATCH_001_BATCH_STUDY_PROCESS_PENDING_POLICY",
        })

    # Suggest monitoring PUCT rulemaking (always relevant while Project 58481 is open)
    _add({
        "kind": "monitor",
        "topic": "PUCT Project 58481 rulemaking (16 TAC §25.194)",
        "rationale": "Interconnection standards for large loads are being established — final rules may alter LLIS obligations and timelines.",
        "source": "PUCT_002_REGULATORY_CHANGE_MAY_ALTER_LLIS_OBLIGATIONS",
    })

    # Suggest monitoring batch study framework finalization
    _add({
        "kind": "monitor",
        "topic": "ERCOT batch study framework finalization",
        "rationale": "Batch study dates are preliminary per Workshop Slide 2 disclaimer — update when ERCOT publishes final batch rules at June 1 2026 Board meeting.",
        "source": "BATCH_004_WORKSHOP_PROCESS_ONGOING",
    })

    return actions


def evaluate_graph(req: Dict[str, Any], graph: Dict[str, Any], *, include_options: bool = True) -> Dict[str, Any]:
    req = normalize_request(req)
    edges: List[Dict[str, Any]] = graph.get("edges") or []
    nodes_by_id = {n.get("id"): n for n in (graph.get("nodes") or [])}
    rules_by_id = load_published_rules()
    docs_by_id = index_docs(load_doc_registry())
    graph_edges_by_id = {str(e.get("id")): e for e in edges if isinstance(e, dict) and e.get("id")}

    current = "N0"
    path_nodes: List[str] = [current]
    traversed_edges: List[Dict[str, Any]] = []
    missing_inputs: List[str] = []
    # Field-level sources for memo-grade uncertainty reporting.
    missing_meta: Dict[str, Dict[str, Any]] = {}
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
            if str(e.get("missing_policy") or "").strip().lower() != "ignore":
                missing_inputs.extend(missing_fields)
                # Keep compact missing provenance (edge-level attribution).
                for f in missing_fields:
                    mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                    mm["value"] = req.get(f) if f in req else None
                    # Edge predicates treat empty string/list as missing, but do not use completeness False-as-missing.
                    mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=False)
                    mm["sources"].append({"kind": "edge_criteria", "edge_id": str(e.get("id") or "")})
                    missing_meta[f] = mm
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
    lever_class_by_id: Dict[str, str] = {}
    levers_catalog = graph.get("levers_catalog") or []
    if isinstance(levers_catalog, list):
        for lv in levers_catalog:
            if not isinstance(lv, dict):
                continue
            lid = str(lv.get("lever_id") or "").strip()
            if not lid:
                continue
            cls = str(lv.get("class") or lv.get("lever_class") or "unknown").strip().lower()
            if cls not in {"hard", "design", "external", "unknown"}:
                cls = "unknown"
            lever_class_by_id[lid] = cls

    levers = []
    if req.get("energization_plan") == "phased":
        levers.append(
            {
                "lever_id": "phased_energization",
                "lever_class": lever_class_by_id.get("phased_energization", "unknown"),
                "status": "present_in_request",
                "note": "Request includes a phased energization plan (can be used as an option lever).",
            }
        )
    if isinstance(req.get("voltage_options_kv"), list) and len(req["voltage_options_kv"]) >= 2:
        levers.append(
            {
                "lever_id": "voltage_level_choice",
                "lever_class": lever_class_by_id.get("voltage_level_choice", "unknown"),
                "status": "present_in_request",
                "note": "Request includes multiple voltage options (can be evaluated as alternative options).",
            }
        )
    if ("multiple_poi" in req) or ("poi_count" in req):
        levers.append(
            {
                "lever_id": "poi_topology_choice",
                "lever_class": lever_class_by_id.get("poi_topology_choice", "unknown"),
                "status": "present_in_request",
                "note": "Request includes POI topology inputs (single vs multiple POIs), which can change screening path/options.",
            }
        )
    if "is_co_located" in req:
        levers.append(
            {
                "lever_id": "co_location_signal",
                "lever_class": lever_class_by_id.get("co_location_signal", "unknown"),
                "status": "present_in_request",
                "note": "Request includes a co-location/shared-infrastructure signal (configuration complexity lever).",
            }
        )
    if "export_capability" in req:
        levers.append(
            {
                "lever_id": "export_capability_choice",
                "lever_class": lever_class_by_id.get("export_capability_choice", "unknown"),
                "status": "present_in_request",
                "note": "Request includes an export capability posture input (configuration lever; can affect ops/study exposure).",
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
    # Deduplicate rules by rule_id to avoid double-counting risk contributions.
    seen_rids: set[str] = set()
    unique_signal_rules: List[Dict[str, Any]] = []
    for r in all_signal_rules:
        rid = str(r.get("rule_id") or "")
        if not rid:
            continue
        if rid in seen_rids:
            continue
        seen_rids.add(rid)
        unique_signal_rules.append(r)

    # Rule checks (predicate-backed), for memo-grade checklists.
    rule_checks: List[Dict[str, Any]] = []
    energization_checklist: List[Dict[str, Any]] = []

    for rid, r in rules_by_id.items():
        if not rule_predicates(r) and not required_fields_from_rule(r):
            continue
        chk = eval_rule_tri_state(r, req)
        mg = normalize_memo_guidance(r)
        if mg:
            chk["memo_guidance"] = mg
        # Carry minimal DSL structure for memo-grade lever guidance (does not affect evaluation).
        chk["predicates"] = rule_predicates(r)
        chk["fields_referenced"] = sorted(_fields_referenced_by_rule(r))
        rule_checks.append(chk)

        tags = set(r.get("trigger_tags") or [])
        stage = str(r.get("stage") or "")
        if ("energization_gate" in tags) or (stage.lower() == "energization"):
            # Only show energization checklist when user is explicitly requesting energization,
            # OR when the rule isn't gated (rare).
            if is_energization_mode(req) or not has_gate_predicate(r, field="is_requesting_energization", value=True):
                energization_checklist.append(chk)

    energization_checklist = dedupe_checklist(energization_checklist)

    # Missing inputs from rules with required_fields (published)
    #
    # Policy:
    # - Only enforce required_fields when the rule is "in play" (predicates evaluate true),
    #   except when the rule has no predicates (pure completeness requirement).
    # - Conditional gates: if applicability isn't known-true, skip enforcement (unknown/not_applicable
    #   should not become missing_inputs).
    # - Energization-gated rules: do not enforce outside energization mode.
    for r in all_signal_rules:
        rf = required_fields_from_rule(r)
        if not rf:
            continue

        dsl = ((r.get("criteria") or {}).get("dsl") or {})
        is_completeness = dsl.get("kind") == "completeness_check"

        # Conditional gates: do not treat unknown applicability as "missing inputs".
        # If applicability is False (not applicable) OR unknown, skip enforcement entirely.
        if dsl.get("kind") == "conditional_gate":
            app_fields = dsl.get("applicability_fields") or []
            if isinstance(app_fields, list) and app_fields:
                applicable_known_true = True
                for f_app in [str(x) for x in app_fields if x]:
                    v_app = get_field(req, f_app)
                    if v_app is False:
                        applicable_known_true = False
                        break
                    if _is_explicit_unknown_value(v_app):
                        applicable_known_true = False
                        break
                    if v_app is not True:
                        applicable_known_true = False
                        break
                if not applicable_known_true:
                    continue

        preds = rule_predicates(r)
        if preds:
            ok, _tr, _miss = criteria_satisfied(req, preds)
            if not ok:
                # Special-case energization gates: only relevant in energization mode.
                if has_gate_predicate(r, field="is_requesting_energization", value=True) and not is_energization_mode(req):
                    continue
                # If predicates are not satisfied (or cannot be evaluated), the rule is not in play
                # for required_fields enforcement in missing_inputs.
                continue
        for f in rf:
            # Same missing semantics as tri-state (but applied to memo-level missing_inputs list)
            if f not in req:
                missing_inputs.append(f)
                mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                mm["value"] = None
                mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=is_completeness)
                mm["sources"].append({"kind": "rule_required_fields", "rule_id": str(r.get("rule_id") or "")})
                missing_meta[f] = mm
                continue
            v = req.get(f)
            if v is None:
                missing_inputs.append(f)
                mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                mm["value"] = None
                mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=is_completeness)
                mm["sources"].append({"kind": "rule_required_fields", "rule_id": str(r.get("rule_id") or "")})
                missing_meta[f] = mm
                continue
            if isinstance(v, str) and v.strip() == "":
                missing_inputs.append(f)
                mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                mm["value"] = v
                mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=is_completeness)
                mm["sources"].append({"kind": "rule_required_fields", "rule_id": str(r.get("rule_id") or "")})
                missing_meta[f] = mm
                continue
            if is_completeness and (v is False):
                missing_inputs.append(f)
                mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                mm["value"] = v
                mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=is_completeness)
                mm["sources"].append({"kind": "rule_required_fields", "rule_id": str(r.get("rule_id") or "")})
                missing_meta[f] = mm
                continue
            if is_completeness and isinstance(v, list) and len(v) == 0:
                missing_inputs.append(f)
                mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": "", "value": None}
                mm["value"] = v
                mm["reason"] = mm["reason"] or _classify_missing_reason(req=req, field=f, is_completeness=is_completeness)
                mm["sources"].append({"kind": "rule_required_fields", "rule_id": str(r.get("rule_id") or "")})
                missing_meta[f] = mm
                continue

    ctx_snapshot = normalize_context_snapshot(req.get("context"))
    risk, risk_trace = risk_from_signals(
        req,
        unique_signal_rules,
        missing_inputs=sorted(set(missing_inputs)),
        context_snapshot=ctx_snapshot,
    )

    # Uncertainty buckets (memo-friendly, compact, evidence-backed):
    # - Missing: absent/empty/false-as-missing fields
    # - Unknown: explicit unknown markers (null / "unknown")
    # - Assumptions: explicitly labeled non-cited heuristics (still auditable)
    # - Operator discretion: signals that indicate engineering/operator judgement remains
    # - Queue-state dependent: signals tied to queue density/dependencies
    missing_inputs_uniq = sorted(set(missing_inputs))
    missing_details = []
    for f in missing_inputs_uniq:
        mm = missing_meta.get(f) or {"field": f, "sources": [], "reason": _classify_missing_reason(req=req, field=f, is_completeness=False), "value": (req.get(f) if f in req else None)}
        # stable + compact sources (dedupe by kind/id)
        seen_src = set()
        src2 = []
        for s in mm.get("sources") or []:
            k = json.dumps(s, sort_keys=True, ensure_ascii=False)
            if k in seen_src:
                continue
            seen_src.add(k)
            src2.append(s)
        mm2 = {
            "field": f,
            "reason": mm.get("reason") or "",
            "provided": f in req,
            "value": req.get(f) if f in req else None,
            "sources": src2,
        }
        missing_details.append(mm2)

    unknown_inputs: List[Dict[str, Any]] = []
    for k, v in sorted(req.items(), key=lambda kv: str(kv[0])):
        if _is_explicit_unknown_value(v):
            unknown_inputs.append({"field": str(k), "value": v})

    assumptions: List[Dict[str, Any]] = []
    for rt in (risk_trace or []):
        if not isinstance(rt, dict):
            continue
        if rt.get("source") == "non_cited_heuristic":
            assumptions.append(
                {
                    "rule_id": rt.get("rule_id"),
                    "loc": rt.get("loc"),
                    "trigger_tags": rt.get("trigger_tags") or [],
                    "notes": rt.get("notes") or [],
                }
            )

    OPERATOR_DISCRETION_TAGS = {"engineering_review", "operator_discretion", "voltage_selection_risk"}
    QUEUE_STATE_TAGS = {"queue_density", "queue_dependency", "must_study"}

    def _compact_signal(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rule_id": r.get("rule_id"),
            "doc_id": r.get("doc_id"),
            "loc": r.get("loc"),
            "trigger_tags": sorted(set(r.get("trigger_tags") or [])),
        }

    operator_discretion_signals = []
    queue_state_dependent_signals = []
    for r in unique_signal_rules:
        tags = set(r.get("trigger_tags") or [])
        if tags & OPERATOR_DISCRETION_TAGS:
            operator_discretion_signals.append(_compact_signal(r))
        if tags & QUEUE_STATE_TAGS:
            queue_state_dependent_signals.append(_compact_signal(r))

    # Attach provenance for any docs referenced by evidence
    used_doc_ids = sorted({e.get("doc_id") for e in (risk.get("evidence") or []) if e.get("doc_id")})
    used_docs = [doc_provenance_entry(docs_by_id[d]) for d in used_doc_ids if d in docs_by_id]

    out = {
        "evaluated_at": now_iso(),
        "request": req,
        "context_snapshot": ctx_snapshot,
        "graph": {
            "path_nodes": path_nodes,
            "path_node_labels": [nodes_by_id.get(n, {}).get("label", n) for n in path_nodes],
            "traversed_edges": traversed_edges,
            "graph_version": graph.get("version"),
        },
        "missing_inputs": sorted(set(missing_inputs)),
        "uncertainties": {
            "missing": missing_details,
            "unknown": unknown_inputs,
            "assumptions": assumptions,
            "operator_discretion": operator_discretion_signals,
            "queue_state_dependent": queue_state_dependent_signals,
        },
        "levers": levers,
        "levers_catalog_analysis": _analyze_levers(req, graph, rules_by_id),
        "options": [],
        "flags": flags,
        "rule_checks": rule_checks,
        "energization_checklist": energization_checklist,
        "risk": risk,
        "risk_trace": risk_trace,
        "provenance": {
            "doc_registry_count": len(load_doc_registry()),
            "graph_source": "graph/process_graph.yaml",
            "graph_sha256": sha256_cached(GRAPH_PATH),
            "rules_source": "rules/published.jsonl",
            "rules_sha256": sha256_cached(RULES_PUBLISHED),
            "docs": used_docs,
        },
    }

    if include_options:
        options: List[Dict[str, Any]] = []
        baseline_summary = _summarize_option_eval(out)

        # Lever 1: voltage choice (evaluate each option separately).
        vopts = get_field(req, "voltage_options_kv")
        if isinstance(vopts, list) and len(vopts) >= 2:
            for v in vopts:
                try:
                    v_num = float(v)
                except Exception:
                    continue
                req2 = copy.deepcopy(req)
                req2["voltage_options_kv"] = [v_num]
                # Keep alias in sync for readability in raw request JSON.
                req2["voltage_options"] = [v_num]
                ev2 = evaluate_graph(req2, graph, include_options=False)
                opt_summary = _summarize_option_eval(ev2)
                delta = _diff_option_summaries(baseline_summary, opt_summary)
                if delta.get("path_changed"):
                    delta["edge_diff"] = _compute_edge_diff_for_option(
                        baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                    )
                options.append(
                    {
                        "option_id": f"voltage_{int(v_num) if v_num.is_integer() else v_num}",
                        "lever_id": "voltage_level_choice",
                        "source": "user_provided_alternatives",
                        "patch": {"voltage_options_kv": [v_num]},
                        "summary": opt_summary,
                        "delta": delta,
                    }
                )

        # Lever 2: single vs phased plan (evaluate alternate plan shape).
        plan = req.get("energization_plan")
        if plan in ("single", "phased"):
            other = "phased" if plan == "single" else "single"
            req2 = copy.deepcopy(req)
            req2["energization_plan"] = other
            if other == "single":
                req2.pop("phases", None)
            ev2 = evaluate_graph(req2, graph, include_options=False)
            opt_summary = _summarize_option_eval(ev2)
            delta = _diff_option_summaries(baseline_summary, opt_summary)
            if delta.get("path_changed"):
                delta["edge_diff"] = _compute_edge_diff_for_option(
                    baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                )
            options.append(
                {
                    "option_id": f"energization_plan_{other}",
                    "lever_id": "phased_energization",
                    "source": "system_generated_toggle",
                    "patch": {"energization_plan": other},
                    "summary": opt_summary,
                    "delta": delta,
                }
            )

        # Lever 3: POI topology (single vs multiple POIs).
        mp = req.get("multiple_poi")
        if mp in (True, False):
            other_mp = not bool(mp)
            req2 = copy.deepcopy(req)
            req2["multiple_poi"] = other_mp
            # Keep poi_count aligned for readability (heuristic; does not claim correctness).
            if other_mp is True:
                try:
                    pc = int(req.get("poi_count") or 2)
                except Exception:
                    pc = 2
                req2["poi_count"] = max(2, pc)
            else:
                req2["poi_count"] = 1
            ev2 = evaluate_graph(req2, graph, include_options=False)
            opt_summary = _summarize_option_eval(ev2)
            delta = _diff_option_summaries(baseline_summary, opt_summary)
            if delta.get("path_changed"):
                delta["edge_diff"] = _compute_edge_diff_for_option(
                    baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                )
            options.append(
                {
                    "option_id": f"poi_topology_{'multi' if other_mp else 'single'}",
                    "lever_id": "poi_topology_choice",
                    "source": "system_generated_toggle",
                    "patch": {"multiple_poi": other_mp, "poi_count": req2.get("poi_count")},
                    "summary": opt_summary,
                    "delta": delta,
                }
            )

        # Lever 4: co-location signal.
        if req.get("is_co_located") in (True, False):
            other_col = not bool(req.get("is_co_located"))
            req2 = copy.deepcopy(req)
            req2["is_co_located"] = other_col
            ev2 = evaluate_graph(req2, graph, include_options=False)
            opt_summary = _summarize_option_eval(ev2)
            delta = _diff_option_summaries(baseline_summary, opt_summary)
            if delta.get("path_changed"):
                delta["edge_diff"] = _compute_edge_diff_for_option(
                    baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                )
            options.append(
                {
                    "option_id": f"co_located_{str(other_col).lower()}",
                    "lever_id": "co_location_signal",
                    "source": "system_generated_toggle",
                    "patch": {"is_co_located": other_col},
                    "summary": opt_summary,
                    "delta": delta,
                }
            )

        # Lever 5: export capability posture.
        exp = str(req.get("export_capability") or "").strip().lower()
        if exp in {"none", "possible", "planned", "unknown", ""}:
            other_exp = "planned" if exp in {"none", "unknown", ""} else "none"
            req2 = copy.deepcopy(req)
            req2["export_capability"] = other_exp
            ev2 = evaluate_graph(req2, graph, include_options=False)
            opt_summary = _summarize_option_eval(ev2)
            delta = _diff_option_summaries(baseline_summary, opt_summary)
            if delta.get("path_changed"):
                delta["edge_diff"] = _compute_edge_diff_for_option(
                    baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                )
            options.append(
                {
                    "option_id": f"export_{other_exp}",
                    "lever_id": "export_capability_choice",
                    "source": "system_generated_toggle",
                    "patch": {"export_capability": other_exp},
                    "summary": opt_summary,
                    "delta": delta,
                }
            )

        # Lever 6: voltage class
        #
        # IMPORTANT: avoid generating "class-only" toggles when we already have numeric voltage
        # evidence (voltage_choice_kv or voltage_options_kv) because it creates inconsistent
        # what-ifs (e.g., voltage_options=[138,345] but poi_voltage_class=distribution).
        #
        # We only allow this toggle when the request explicitly provides poi_voltage_class
        # without any numeric voltage choice/options.
        has_voltage_choice = req.get("voltage_choice_kv") is not None
        has_voltage_opts = isinstance(vopts, list) and len(vopts) > 0
        if (not has_voltage_choice) and (not has_voltage_opts):
            cls = str(req.get("poi_voltage_class") or "").strip().lower()
            if cls in {"distribution", "transmission"}:
                other_cls = "transmission" if cls == "distribution" else "distribution"
                req2 = copy.deepcopy(req)
                req2["poi_voltage_class"] = other_cls
                ev2 = evaluate_graph(req2, graph, include_options=False)
                opt_summary = _summarize_option_eval(ev2)
                delta = _diff_option_summaries(baseline_summary, opt_summary)
                if delta.get("path_changed"):
                    delta["edge_diff"] = _compute_edge_diff_for_option(
                        baseline_ev=out, option_ev=ev2, graph_edges_by_id=graph_edges_by_id, rules_by_id=rules_by_id
                    )
                options.append(
                    {
                        "option_id": f"voltage_class_{other_cls}",
                        "lever_id": "voltage_level_choice",
                        "source": "system_generated_toggle",
                        "patch": {"poi_voltage_class": other_cls},
                        "summary": opt_summary,
                        "delta": delta,
                    }
                )

        # Bounded recommendation among baseline + generated options
        base_candidate = {
            "option_id": "baseline",
            "lever_id": "baseline",
            "source": "baseline",
            "patch": {},
            "summary": baseline_summary,
            "delta": _diff_option_summaries(baseline_summary, baseline_summary),
        }
        # Always include baseline as first option so compact table can reference it
        out["options"] = [base_candidate] + options

        candidates = [base_candidate] + list(options)
        out["recommendation"] = _recommendation_from_candidates(
            baseline=base_candidate,
            candidates=candidates,
            req_is_energization=is_energization_mode(req),
        )

    # Decision summary + next actions + top drivers (memo-friendly)
    out["decision_screening"] = _decision_screening_status(out)
    out["decision_energization"] = _decision_energization_status(out)
    out["decision"] = out["decision_energization"] if is_energization_mode(req) else out["decision_screening"]
    out["next_actions"] = _next_actions(out, max_items=10)
    out["top_drivers"] = _top_risk_drivers(out.get("risk_trace") or [], top_n=6)

    return out


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
