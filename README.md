## tx-large-load-v0

Goal: ship a **v0 ERCOT (Texas) Large Load** workflow that takes a request → evaluates a process graph → produces a **Decision Memo** with provenance hooks (doc + rule ids + versions).

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

Notes:
- `docs/raw/` is gitignored by default; the registry stores the provenance to local copies.
- `rules/published.jsonl` is intentionally the only rule set used for memo outputs.
