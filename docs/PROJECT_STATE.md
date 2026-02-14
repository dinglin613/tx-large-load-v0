# Project state (tx-large-load-v0)

This doc is a *living* snapshot intended to prevent “spec drift” between the code and how we describe it.

## What is already true in code (v0 contract)

- **Request → deterministic process path → Decision Memo**
  - `src/evaluate.py` traverses `graph/process_graph.yaml` deterministically: *first satisfied outgoing edge wins* (YAML order is semantic).
  - `src/render_memo.py` turns evaluation JSONL into single-file HTML memos.
- **Auditability**
  - Path edges carry `rule_ids`; memo includes an edge audit log with predicate traces and citations.
  - Risk output includes evidence items with `rule_id` / `doc_id` / `loc` / `trigger_tags`.
- **Rule-driven risk buckets (v0)**
  - Risk is qualitative and explainable; `risk_trace` records per-signal contributions and final bucket assignment.
  - If no published rules match/flag, risk explicitly reports `insufficient_rule_coverage` (heuristics may still apply, and are labeled).
- **Parallel flags**
  - Published rules with predicates that match the request (but are not on the chosen path) are emitted as `flags[]`.
- **Energization checklist (tri-state)**
  - When `is_requesting_energization=true`, rules tagged as energization gates are evaluated into `satisfied` / `missing` / `not_satisfied`.
  - Checklist rows are deduped to stay compact (prefers Planning Guide where overlaps exist).
- **Provenance**
  - Evaluation output includes `graph_sha256` + `rules_sha256`.
  - Documents are referenced via `registry/doc_registry.json` artifacts (`path` + `sha256`), and the memo embeds the provenance table.
- **Citation integrity (PDF)**
  - `src/citation_audit.py` validates published rules’ `loc` against local PDFs: artifact sha256 match, cited page range in bounds, and “anchor” tokens found in extracted PDF text.
  - Memo rendering runs the audit by default (`--citation-audit strict`).

## Clarifications (common “stale” assumptions)

- `graph/process_graph.yaml` is already multi-line, human-editable YAML. The main maintainability constraint is **edge ordering** (reordering edges can change behavior).
- `rules/published.jsonl` is no longer “just a few seed rules”: it already includes Planning Guide Section 9 clause-level anchors (`PG9_*`) plus Q&A, energization form, and TDSP examples. Coverage can still be improved, but the repo is past the “toy skeleton” stage.

## Known v0 limitations (intentionally accepted)

- **Coverage is not completeness**: v0 is a screening engine; many real-world decisions still depend on queue state, studies, and stakeholder approvals.
- **Some signals are heuristics**: non-cited heuristics are explicitly labeled (`doc_id=NON_CITED_HEURISTIC`) to keep the audit boundary clear.
- **Schema is necessary but not sufficient**: `schema/request.schema.json` defines field names and types, but the evaluator applies additional semantics (e.g., how `completeness_check` treats false/empty inputs).

## Near-term improvements that preserve the v0 shape

- Concentrate rules around a small set of “decision levers” that change outcomes (e.g., voltage choice, phased vs single) and ensure memos explain *why*.
- Expand request schema documentation (descriptions/examples) as the stable integration boundary for pilots.
- Keep “unknowns / assumptions” explicit in memo outputs (missing inputs + explicitly labeled heuristics), so readers can see what is *not* being claimed.

