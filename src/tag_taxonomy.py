from __future__ import annotations

"""
Tag taxonomy for tx-large-load-v0.

Goals:
- Keep trigger_tags consistent and interpretable across rules.
- Make it explicit which tags contribute to v0 risk buckets (upgrade / wait / ops).

This module is intentionally dependency-free so it can be imported by both:
- `src/evaluate.py` (risk scoring / memo evidence)
- `src/validate.py` (linting / warnings)
"""

from typing import Dict

# Risk contribution mapping:
#   tag -> {bucket: delta_score}
#
# Buckets:
# - wait: timeline / process latency pressure
# - upgrade: upgrade exposure pressure
# - ops: operational exposure pressure
TAG_RISK_CONTRIB: Dict[str, Dict[str, float]] = {
    # wait / timeline pressure
    "process_latency": {"wait": 0.5},
    "timeline_risk": {"wait": 1.0},
    "timeline_dependency": {"wait": 0.5},
    "study_dependency": {"wait": 0.5},
    "re_study": {"wait": 1.5},
    "cancellation_risk": {"wait": 2.0},
    "queue_density": {"wait": 1.0},
    "queue_dependency": {"wait": 1.0},
    "must_study": {"wait": 1.0},
    "financial_security": {"wait": 0.5},
    "modeling_requirement": {"wait": 0.5},
    "telemetry_requirement": {"wait": 0.5},
    "monitoring_requirement": {"wait": 0.5},
    "dynamic_modeling": {"wait": 0.5},
    "commissioning_plan": {"wait": 0.5},
    "fee_requirement": {"wait": 0.5},
    "intake_gate": {"wait": 0.5},
    # Data completeness is handled as a special-case penalty when missing required_fields.
    # We still include a small baseline wait contribution to make its role visible.
    "data_completeness": {"wait": 0.5},

    # upgrade exposure
    "upgrade_exposure": {"upgrade": 0.5},
    "voltage_selection_risk": {"upgrade": 0.5},
    "weak_grid_risk": {"upgrade": 1.0},
    "protection": {"upgrade": 1.0},
    "short_circuit": {"upgrade": 1.0},
    "new_substation": {"upgrade": 3.0},
    "new_line": {"upgrade": 3.0},
    "dynamic_stability": {"upgrade": 1.0},
    "dynamic_study": {"upgrade": 1.0},
    "sso": {"upgrade": 1.0},

    # operations exposure
    "curtailment_risk": {"ops": 2.0},
    "operational_constraint": {"ops": 1.0},
    "commissioning_limit": {"ops": 1.0},
    "compliance_risk": {"ops": 1.0},
    "metering_requirement": {"ops": 0.5},
    # NOTE: energization_gate is intentionally not a risk contributor; it is a gating concept.

    # ── NEW v0 tags (added 2026-02-26) ──────────────────────────────────────
    # Process / governance dependency
    # batch_study: project timing may depend on ERCOT batch study cycle.
    "batch_study": {"wait": 2.0},
    # fast_track: fast-track/priority path available; reduces wait pressure (negative delta).
    "fast_track": {"wait": -0.5},
    # governance_dependency: outcome depends on ERCOT/PUCT policy decision not yet final.
    "governance_dependency": {"wait": 1.5},

    # Geographic / reliability constraint
    # geo_risk_far_west: Far West Texas area-specific reliability constraints (e.g., no-solar scenario).
    "geo_risk_far_west": {"upgrade": 1.5, "wait": 1.0},
    # study_assumption_change: study results may be invalidated by changed system assumptions (e.g., generator retirements, new constraints).
    "study_assumption_change": {"wait": 1.5, "upgrade": 0.5},
    # restudy_risk: elevated likelihood of a restudy being required (general; complements re_study which is event-triggered).
    "restudy_risk": {"wait": 1.5},

    # Regulatory / PUCT
    # puct_rule_pending: outcome depends on pending PUCT rulemaking (rule not yet effective).
    "puct_rule_pending": {"wait": 1.0},
    # compliance_dependency: project may be subject to compliance requirements whose final form is uncertain.
    "compliance_dependency": {"wait": 0.5, "ops": 0.5},
}


# Tags that are valid and useful for evidence grouping / auditability,
# but intentionally do NOT contribute to v0 risk scoring.
KNOWN_NON_RISK_TAGS = {
    "process_trigger",
    "modification_trigger",
    "energization_gate",
    "engineering_review",
    "steady_state",
    "pscad_required",
    "mqt_required",
    "transmission_scope",
    # New non-risk labels added 2026-02-26
    "policy_update",        # marks rules that describe a policy/market process change (informational)
    "market_notice",        # marks rules sourced from ERCOT market notices (informational)
}

ALL_KNOWN_TAGS = set(TAG_RISK_CONTRIB.keys()) | set(KNOWN_NON_RISK_TAGS)

