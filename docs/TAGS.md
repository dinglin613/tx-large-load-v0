# trigger_tags taxonomy (v0)

`trigger_tags` serve two roles:

- **Evidence labels**: help readers scan *why* something appeared (grouping/filtering in the memo).
- **Risk inputs (v0)**: some tags contribute to qualitative risk buckets via rule-driven scoring.

In v0, tags are intentionally simple. Not every tag contributes to risk.

## Risk-contributing tags

These tags contribute to v0 scoring (see `src/tag_taxonomy.py` for the authoritative mapping):

- **wait / timeline pressure**: `process_latency`, `timeline_risk`, `timeline_dependency`, `study_dependency`, `re_study`, `cancellation_risk`, `queue_density`, `queue_dependency`, `must_study`, `financial_security`, `modeling_requirement`, `telemetry_requirement`, `monitoring_requirement`, `dynamic_modeling`, `commissioning_plan`, `fee_requirement`, `intake_gate`, `data_completeness`
- **upgrade exposure**: `upgrade_exposure`, `voltage_selection_risk`, `weak_grid_risk`, `protection`, `short_circuit`, `new_substation`, `new_line`, `dynamic_stability`, `dynamic_study`, `sso`
- **ops exposure**: `curtailment_risk`, `operational_constraint`, `commissioning_limit`, `compliance_risk`, `metering_requirement`

Notes:
- `data_completeness` also triggers an additional explicit penalty when rule required_fields are missing (see `src/evaluate.py`).

## Evidence-only (non-risk) tags

These tags are valid and useful, but do not (currently) contribute to v0 risk scoring:

- `process_trigger`, `modification_trigger`
- `energization_gate`
- `engineering_review`
- `steady_state`
- `pscad_required`, `mqt_required`
- `transmission_scope`

## Validation behavior

`src/validate.py` emits warnings for:

- Tags used by **published** rules that are **not** in the taxonomy.
- Tags used by **published** rules that are **not** in the risk mapping (to prevent confusion about whether they affect buckets).

This is warning-only in v0: it is meant to support disciplined iteration without blocking progress.

