from __future__ import annotations

"""
Tag taxonomy for tx-large-load-v0.

Goals:
- Keep trigger_tags consistent and interpretable across rules.
- Make it explicit which tags contribute to v0 risk buckets (upgrade / wait / ops).
- Separate BASELINE tags (inherent to every large load process) from INCREMENTAL
  tags (project-specific risk signals) so exposure buckets have meaningful variance.

This module is intentionally dependency-free so it can be imported by both:
- `src/evaluate.py` (risk scoring / memo evidence)
- `src/validate.py` (linting / warnings)
"""

from typing import Dict

# ────────────────────────────────────────────────────────────────────────────────
# BASELINE tags: inherent to every large load interconnection process.
# These represent standard study / process requirements that every project
# encounters.  They contribute to a separate "baseline_load" score that
# is reported for transparency but NOT used for the high/medium/low exposure
# bucket.  This prevents every project from being rated "high" simply because
# the LLIS process exists.
# ────────────────────────────────────────────────────────────────────────────────
BASELINE_TAGS: Dict[str, Dict[str, float]] = {
    # Standard study elements (every project must complete these)
    "steady_state": {"wait": 0.5},
    "study_dependency": {"wait": 0.5},
    "short_circuit": {"upgrade": 0.5, "wait": 0.5},    # standard LLIS study element; PG9_009/017
    "protection": {"upgrade": 0.5, "wait": 0.5},       # standard protection review; PG9_017
    "dynamic_stability": {"upgrade": 0.5, "wait": 0.5}, # standard dynamic study; PG9_018
    # Standard intake/process gates (every project must satisfy these)
    "process_latency": {"wait": 0.5},
    "intake_gate": {"wait": 0.5},
    "data_completeness": {"wait": 0.5},
    "commissioning_plan": {"wait": 0.5},
    "fee_requirement": {"wait": 0.5},
    "modeling_requirement": {"wait": 0.5},
    "telemetry_requirement": {"wait": 0.5},
    "monitoring_requirement": {"wait": 0.5},
    "dynamic_modeling": {"wait": 0.5},
    "metering_requirement": {"ops": 0.5},
    "financial_security": {"wait": 0.5},
    # Batch study process (applies to all projects in the current ERCOT regime)
    "batch_study": {"wait": 2.0},
    # Governance dependency (applies system-wide; not project-specific)
    "governance_dependency": {"wait": 1.5},
    # PUCT rulemaking (applies to all projects under current regime)
    "puct_rule_pending": {"wait": 1.0},
    # Timeline dependency (baseline; every project has schedule dependency)
    "timeline_dependency": {"wait": 0.5},
}

# ────────────────────────────────────────────────────────────────────────────────
# INCREMENTAL tags (TAG_RISK_CONTRIB): project-specific risk signals.
# Only these contribute to the upgrade/wait/ops exposure buckets.
# A tag here means "this project has ADDITIONAL risk beyond the baseline process."
# ────────────────────────────────────────────────────────────────────────────────
TAG_RISK_CONTRIB: Dict[str, Dict[str, float]] = {
    # ── wait / timeline pressure (incremental) ──
    "timeline_risk": {"wait": 1.0},
    "re_study": {"wait": 1.5},
    "cancellation_risk": {"wait": 2.0},
    "queue_density": {"wait": 1.0},
    "queue_dependency": {"wait": 1.0},
    "must_study": {"wait": 1.0},

    # ── upgrade exposure (incremental) ──
    "upgrade_exposure": {"upgrade": 0.5},
    "voltage_selection_risk": {"upgrade": 0.5},
    "weak_grid_risk": {"upgrade": 1.0},
    "new_substation": {"upgrade": 3.0},
    "new_line": {"upgrade": 3.0},
    "dynamic_study": {"upgrade": 1.0},
    "sso": {"upgrade": 1.0},

    # ── operations exposure (incremental) ──
    "curtailment_risk": {"ops": 2.0},
    "operational_constraint": {"ops": 1.0},
    "commissioning_limit": {"ops": 1.0},
    "compliance_risk": {"ops": 1.0},

    # ── fast track (negative — reduces pressure) ──
    "fast_track": {"wait": -0.5},

    # ── geographic / reliability constraints (project-specific) ──
    "geo_risk_far_west": {"upgrade": 1.5, "wait": 1.0},
    "study_assumption_change": {"wait": 1.5, "upgrade": 0.5},
    "restudy_risk": {"wait": 1.5},

    # ── regulatory / compliance (incremental on top of baseline puct_rule_pending) ──
    "compliance_dependency": {"wait": 0.5, "ops": 0.5},
}


# Tags that are valid and useful for evidence grouping / auditability,
# but intentionally do NOT contribute to v0 risk scoring.
KNOWN_NON_RISK_TAGS = {
    "process_trigger",
    "modification_trigger",
    "energization_gate",
    "engineering_review",
    "pscad_required",
    "mqt_required",
    "transmission_scope",
    # New non-risk labels added 2026-02-26
    "policy_update",        # marks rules that describe a policy/market process change (informational)
    "market_notice",        # marks rules sourced from ERCOT market notices (informational)
}

ALL_KNOWN_TAGS = set(TAG_RISK_CONTRIB.keys()) | set(KNOWN_NON_RISK_TAGS) | set(BASELINE_TAGS.keys())

