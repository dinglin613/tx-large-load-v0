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
}

ALL_KNOWN_TAGS = set(TAG_RISK_CONTRIB.keys()) | set(KNOWN_NON_RISK_TAGS)

