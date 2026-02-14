## tx-large-load-v0

Goal: ship a **v0 ERCOT (Texas) Large Load** workflow that takes a request → evaluates a process graph → produces a **Decision Memo** with provenance hooks (doc + rule ids + versions).

### Compact spec (current behavior)

- **Deterministic traversal**: evaluate outgoing edges in YAML file order; **first satisfied edge wins**.
- **Audit-friendly evidence**: every surfaced signal includes `rule_id` / `doc_id` / `loc` / `trigger_tags`.
- **Risk buckets (v0)**: qualitative, rule-driven buckets with an evidence list and a reproducible `risk_trace`.
- **Parallel flags**: predicate-matched published rules that were **not** required on the main path.
- **Energization readiness**: when `is_requesting_energization=true`, render a compact **tri-state** checklist (`satisfied` / `missing` / `not_satisfied`) with dedupe that prefers Planning Guide citations.
- **Provenance**: include `graph_sha256` + `rules_sha256`, plus doc artifact paths + sha256 from the local registry.
- **Citation integrity**: optional strict audit verifies published rule `loc` against local PDFs (artifact sha256, page range, and anchor tokens).

### Repo map (where things live)

- **Docs + provenance**
  - `docs/raw/`: local PDFs (convention: `DOC_ID__<effective>.pdf`)
  - `registry/doc_registry.json`: doc registry (doc metadata + artifacts with `path`/`sha256`/`retrieved_date`)
  - `src/ingest.py`: scan `docs/raw/` and upsert artifact sha256 into registry
- **Rules**
  - `schema/rule.schema.json`: rule schema
  - `rules/published.jsonl`: authoritative rule set used by the evaluator + memo outputs
  - `src/review_gate.py`: minimal publish-quality gate (loc not TBD, confidence threshold, trigger_tags, doc_id exists, DSL sanity)
- **Process graph**
  - `graph/process_graph.yaml`: process nodes/edges + edge criteria + `rule_ids` citations (edge order matters)
- **Engine + outputs**
  - `schema/request.schema.json`: request schema (includes energization fields and v0 toggles)
  - `src/validate.py`: schema validation + DSL lint + graph reference checks (fail-fast)
  - `src/evaluate.py`: deterministic traversal + tri-state rule checks + flags + options + provenance hashes
  - `src/citation_audit.py`: audit published rule citations vs local PDFs
  - `src/render_memo.py`: render eval JSONL → single-file HTML memo (runs citation audit by default)

### What to do next (recommended order)

1. **Register docs**: keep `registry/doc_registry.json` updated with local artifacts (PDFs / snapshots) + `sha256`.
2. **Seed rules**: add a small set of high-signal rules into `rules/drafts.jsonl`, then promote to `rules/published.jsonl`.
3. **Run v0 demo**: run evaluator and render a memo from `tests/test_requests.jsonl`.

### Setup

- Install Python deps:

```bash
python -m pip install -r requirements.txt
```

### Ingest: fill registry artifact hashes

This scans `docs/raw/`, matches known filenames to the docs in `registry/doc_registry.json`, fills `artifacts[].sha256` and `retrieved_date`, and prints any **unregistered** files it found.

```bash
python src/ingest.py
```

### Find page/loc fast (PDF search)

Use this to locate keywords and get page numbers while writing rules:

```bash
python src/pdf_search.py --pdf "docs/raw/ERCOT_LARGE_LOAD_QA__2025-06-01.pdf" --re "Load Commissioning Plan|Survey"
```

### Evaluate + render memo (v0)

```bash
python src/validate.py --rules rules/published.jsonl --requests tests/test_requests.jsonl --graph graph/process_graph.yaml
python src/evaluate.py --in tests/test_requests.jsonl --out memo/outputs/evals.jsonl
python src/render_memo.py --in memo/outputs/evals.jsonl --out-dir memo/outputs
```

Notes:
- `render_memo.py` runs a **strict PDF citation audit** by default. To skip (e.g., no local PDFs), pass `--citation-audit off`.
- `docs/raw/` is gitignored by default; the registry stores the provenance to local copies.
- `rules/published.jsonl` is intentionally the only rule set used for memo outputs.
