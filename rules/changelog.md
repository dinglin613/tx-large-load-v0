# Rules Changelog

## 2026-02-26 — Batch study / Far West / PUCT / New Q&A (20 new rules)

**Motivation**: Gap-fill identified by demo readiness review:
1. Published rules lacked coverage of ERCOT's 2025-12 → 2026 policy shift to Batch Study Process.
2. Far West Texas reliability constraints (Market Notice M-A022326-01) were not represented.
3. PUCT rulemaking (Project 58481) dependency was not captured.
4. 2025-06-01 Q&A citations were not updated to Rev.12.15.25 (effective 2025-12-15).

**New rule IDs published (--allow-unsafe; sha256 pending PDF/snapshot download)**:
- `BATCH_001` – `BATCH_005`: ERCOT Batch Study Process policy and queue/cycle implications
- `FAR_WEST_001` – `FAR_WEST_004`: Far West Texas reliability constraints (M-A022326-01)
- `PUCT_001` – `PUCT_003`: PUCT Project 58481 rulemaking dependency
- `QA_REV1215_NET_METERING_SB6_APPROVAL_REQUIRED`: Updated citation to Rev.12.15.25
- `QA_REV1215_BATCH_STUDY_REFERENCE_PROCESS`: Rev.12.15.25 batch study process reference (PLACEHOLDER — confirm section after PDF download)
- `QA_REV1215_COLOCATION_APPLICABILITY_UPDATED`: Rev.12.15.25 co-location applicability
- `RESTUDY_001` – `RESTUDY_002`: Material change re-slotting under batch cycle; cohort queue disruption
- `MARKET_NOTICE_001` – `MARKET_NOTICE_002`: ERCOT market notice series M-A122325-01 through -05
- `POLICY_001`: Macro governance transition narrative (data center / batch study context)

**New tags added to tag_taxonomy.py**:
- Risk-contributing: `batch_study`, `fast_track`, `governance_dependency`, `geo_risk_far_west`, `study_assumption_change`, `restudy_risk`, `puct_rule_pending`, `compliance_dependency`
- Non-risk (evidence-only): `policy_update`, `market_notice`

**Action required before promoting to "hardened" published status**:
1. Download Rev.12.15.25 Q&A PDF → `docs/snapshots/ERCOT_LARGE_LOAD_QA_REV20251215__2025-12-15.pdf` → run `python src/ingest.py` to fill sha256.
2. Download / snapshot all Market Notices (M-A122325-01/02/03/05, M-A022326-01) → `docs/snapshots/` → `src/ingest.py`.
3. Download PUCT Project 58481 PDF → `docs/snapshots/` → `src/ingest.py`.
4. Download Batch Study PPT (PPTX or PDF export) → `docs/snapshots/` → `src/ingest.py`.

---

## 2026-02-26 — Citation audit + sha256 hardening (post-download)

**What was done**: All 9 pending snapshot files were downloaded and sha256 fields were populated in `doc_registry.json`.

**Downloads completed** (all to `docs/snapshots/`):
| File | sha256 (first 16) | Size |
|------|-------------------|------|
| `ERCOT_LARGE_LOAD_QA_REV20251215__2025-12-15.pdf` | `e1fba2a7986a8de3` | 246,070 B |
| `ERCOT_LLI_BATCH_STUDY_PPT_20260122__2026-01-22.pptx` | `03d4f5cfb192d48e` | 1,892,158 B |
| `ERCOT_LLI_WORKSHOP_20260226__2026-02-26.html` | `3d28c5a55d3ae2c8` | 61,006 B |
| `ERCOT_MARKET_NOTICE_A122325_01__2025-12-23.html` | `c1d2335acaaae1a4` | 8,934 B |
| `ERCOT_MARKET_NOTICE_A122325_02__2025-12-23.html` | `6e36dec03b5d2cde` | 7,258 B |
| `ERCOT_MARKET_NOTICE_A122325_03__2025-12-23.html` | `6539080a0125b311` | 8,328 B |
| `ERCOT_MARKET_NOTICE_A122325_05__2025-12-23.html` | `ee9e908b02f1ea62` | 6,264 B |
| `ERCOT_MARKET_NOTICE_A022326_01__2026-02-23.html` | `32956dda69810429` | 8,674 B |
| `PUCT_PROJECT_58481_LARGE_LOAD__2026-02-26.pdf` | `0f19c9ae3319a15b` | 333,246 B |

**Key findings from citation audit**:

1. **`QA_REV1215_BATCH_STUDY_REFERENCE_PROCESS` — CRITICAL CORRECTION**: The Q&A Rev.12.15.25 does NOT discuss the batch study process. The batch process was announced 8 days later on Dec 23, 2025 via M-A122325-01. Rule re-anchored to `ERCOT_MARKET_NOTICE_A122325_01` with corrected criteria text and confidence raised to 0.95.

2. **`ERCOT_LARGE_LOAD_QA_REV20251215` sha256 = same as `ERCOT_LARGE_LOAD_QA`**: ERCOT updated the document in-place at the same URL. The file is confirmed Rev.12.15.25 (cover page confirmed). Both doc_ids now point to the same underlying bytes; `ERCOT_LARGE_LOAD_QA` is the legacy path in `docs/raw/`, `ERCOT_LARGE_LOAD_QA_REV20251215` is the archival copy in `docs/snapshots/`.

3. **PUCT doc is Sierra Club docket comments**: The downloaded file (58481_91_1575552.PDF) is the Lone Star Chapter Sierra Club comments on the 2nd Discussion Draft (16 TAC §25.194, Jan 9 workshop). It is a valid docket filing and confirms the rulemaking is active and substantive. PUCT rule loc fields updated to clarify this is a docket filing, not the rule text itself.

4. **Q&A p.14 confirms PUCT Project 58481 dependency**: The Q&A Rev.12.15.25 explicitly states: "PUC is in the process of establishing new interconnection standards for large loads in Project No. 58481. That rulemaking could require changes to ERCOT's large load interconnection process and delay projects." This cross-validates `PUCT_001/002/003` via two independent sources.

5. **Far West county list confirmed**: M-A022326-01 names 38 specific counties. Rule `FAR_WEST_003` loc updated with the complete county list for traceability.

6. **M-A122325-05 "Batch Zero" key quote**: "PUC recently provided guidance to ERCOT and stakeholders to engage in an expedited process to develop revision requests that create a new Batch Zero study for consideration at the **June 1, 2026 ERCOT Board meeting**." This is a hard governance milestone. `MARKET_NOTICE_002` loc updated.

**Rules with loc updated (14 total)**:
`BATCH_001`, `BATCH_005`, `FAR_WEST_001`, `FAR_WEST_002`, `FAR_WEST_003`, `FAR_WEST_004`, `PUCT_001`, `PUCT_002`, `PUCT_003`, `QA_REV1215_NET_METERING_SB6_APPROVAL_REQUIRED`, `QA_REV1215_BATCH_STUDY_REFERENCE_PROCESS` (also doc_id changed), `QA_REV1215_COLOCATION_APPLICABILITY_UPDATED`, `RESTUDY_002`, `MARKET_NOTICE_001`, `MARKET_NOTICE_002`

**Remaining open items**:
- `BATCH_002`/`BATCH_003`/`RESTUDY_001` still cite `ERCOT_LLI_BATCH_STUDY_PPT_20260122` (PPTX). The PPTX is downloaded but not text-extractable without LibreOffice/python-pptx. Verify slide content manually or add python-pptx to requirements.txt.
- `BATCH_004` cites `ERCOT_LLI_WORKSHOP_20260226` (Workshop #3 HTML). The HTML is downloaded and is the ERCOT calendar page (61KB); content is JS-rendered. Citation is plausible as a date/event marker but anchor text cannot be verified from static HTML alone.
5. Update `QA_REV1215_BATCH_STUDY_REFERENCE_PROCESS` loc field once PDF is reviewed.
6. Optionally narrow `FAR_WEST_*` rules to a `region` request field once schema is extended.
7. Re-run `python src/review_gate.py check-published` to verify all doc_ids resolve correctly after snapshots are in place.

## 2026-02-13 — Initial v0 seed rules (93 rules)

Initial import of Planning Guide Section 9 clause-level rules (PG9_*),
ERCOT Large Load Q&A (LL_QA_*), Standalone Energization Request (LLI_EN_*),
Oncor 520-106 (ONC_*), and DWG Procedure Manual (DWG_*).
