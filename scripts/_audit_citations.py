"""Audit: check rule_id citations, option baseline, next_actions, timeline consistency."""
import json, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load published rule_ids
with open("rules/published.jsonl") as f:
    pub_ids = set()
    pub_rules = {}
    for line in f:
        r = json.loads(line)
        rid = r.get("rule_id", "")
        pub_ids.add(rid)
        pub_rules[rid] = r

# Load doc registry
with open("registry/doc_registry.json") as f:
    registry = json.load(f)
reg_ids = set(d.get("doc_id") for d in registry if isinstance(d, dict))

# Load evals
with open("memo/outputs/evals.jsonl") as f:
    evals = [json.loads(line) for line in f]

print(f"=== CITATION AUDIT ({len(evals)} evals, {len(pub_ids)} rules, {len(reg_ids)} docs) ===\n")

issues = []

for d in evals:
    name = d["request"]["project_name"]
    
    # 1. Check rule_id citations
    evidence = d.get("risk", {}).get("evidence", [])
    for e in evidence:
        rid = e.get("rule_id", "")
        if rid.startswith("HEUR_") or rid.startswith("CTX_"):
            continue
        if rid not in pub_ids:
            issues.append(f"ORPHAN rule_id: {name} -> {rid}")
        doc_id = e.get("doc_id", "")
        if doc_id and doc_id != "NON_CITED_HEURISTIC" and doc_id != "CONTEXT_SNAPSHOT" and doc_id not in reg_ids:
            issues.append(f"ORPHAN doc_id in evidence: {name} -> {doc_id}")
    
    # 2. Check baseline option missing
    opts = d.get("options", [])
    opt_ids = [o.get("option_id") for o in opts]
    rec_id = (d.get("recommendation", {}) or {}).get("recommended_option_id", "?")
    if "baseline" not in opt_ids:
        issues.append(f"MISSING baseline in options[]: {name} (rec={rec_id})")
    
    # 3. Check next_actions empty when it shouldn't be
    na = d.get("next_actions", [])
    miss = d.get("missing_inputs", [])
    if not na and not miss:
        # Check if there are high-risk flags that warrant actions
        risk = d.get("risk", {})
        up_b = risk.get("upgrade_exposure_bucket", "")
        if up_b in ("high", "medium"):
            issues.append(f"EMPTY next_actions but upgrade={up_b}: {name}")
    
    # 4. Check timeline consistency
    risk = d.get("risk", {})
    te = risk.get("timeline_estimate", {})
    te_status = te.get("status", "")
    te_floor = te.get("study_results_floor_date")
    te_cod = te.get("earliest_cod_quarter")
    batch_elig = d["request"].get("batch_zero_eligible")
    
    if batch_elig is not None and te_status != "anchored":
        issues.append(f"TIMELINE: batch_zero_eligible={batch_elig} but status={te_status}: {name}")
    if batch_elig is None and te_status == "anchored":
        issues.append(f"TIMELINE: no batch_zero_eligible but status=anchored: {name}")
    if te_status == "anchored" and not te_floor:
        issues.append(f"TIMELINE: anchored but no study_floor_date: {name}")
    if te_status == "anchored" and not te_cod:
        issues.append(f"TIMELINE: anchored but no earliest_cod_quarter: {name}")
    
    # 5. Check option summaries have valid buckets
    for o in opts:
        oid = o.get("option_id", "?")
        s = o.get("summary", {})
        up_b = s.get("upgrade_exposure_bucket", "")
        ops_b = s.get("operational_exposure_bucket", "")
        if up_b not in ("low", "medium", "high", "unknown"):
            issues.append(f"INVALID upgrade bucket in option {oid}: {up_b}: {name}")
        if ops_b not in ("low", "medium", "high", "unknown"):
            issues.append(f"INVALID ops bucket in option {oid}: {ops_b}: {name}")
    
    # 6. Check confidence field
    conf = te.get("confidence", "")
    if conf not in ("low", "medium", "medium-high", "high"):
        issues.append(f"INVALID confidence={conf}: {name}")
    
    # 7. Check decision present
    dec = d.get("decision", {})
    if not dec or not dec.get("mode"):
        issues.append(f"MISSING decision: {name}")
    
    # 8. Check top_drivers present
    td = d.get("top_drivers", [])
    if not td:
        issues.append(f"EMPTY top_drivers: {name}")
    
    # 9. Check graph traversal
    g = d.get("graph", {})
    if not g.get("path_node_labels"):
        issues.append(f"EMPTY graph path: {name}")
    if not g.get("traversed_edges"):
        issues.append(f"EMPTY graph edges: {name}")

# Print all issues
if issues:
    print(f"Found {len(issues)} issues:\n")
    for i, iss in enumerate(issues, 1):
        print(f"  {i:3d}. {iss}")
else:
    print("No issues found!")

print(f"\n=== SUMMARY ===")
print(f"Total evals: {len(evals)}")
print(f"Total issues: {len(issues)}")

# Categorize
cats = {}
for iss in issues:
    cat = iss.split(":")[0].strip()
    cats[cat] = cats.get(cat, 0) + 1
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
