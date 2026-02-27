"""Quick audit script – prints summary of all evals.

Structure reference (evals.jsonl top-level):
  request.project_name, risk.upgrade_score, risk.upgrade_exposure_bucket,
  risk.ops_score, risk.operational_exposure_bucket,
  risk.timeline_estimate.{confidence, earliest_cod_quarter, status},
  graph.path_node_labels, options[], missing_inputs[], risk_trace (list),
  rule_checks[], top_drivers[], decision.mode
"""
import json, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
with open("memo/outputs/evals.jsonl") as f:
    for line in f:
        d = json.loads(line)
        rows.append(d)

print(f"{'PROJECT':<55} {'UP_SC':>6} {'UP_BK':>6}  {'OP_SC':>6} {'OP_BK':>6}  {'CONF':<12}  {'COD':<12}  {'OPTS':>4}  {'MISS':>4}")
print("-" * 150)
for d in rows:
    req = d.get("request", {})
    risk = d.get("risk", {})
    te = risk.get("timeline_estimate", {})
    name = req.get("project_name", "?")
    up_sc = risk.get("upgrade_score", 0) or 0
    up_bk = risk.get("upgrade_exposure_bucket", "?")
    op_sc = risk.get("ops_score", 0) or 0
    op_bk = risk.get("operational_exposure_bucket", "?")
    conf = te.get("confidence", "?")
    cod = te.get("earliest_cod_quarter", "?")
    opts = len(d.get("options", []))
    miss = len(d.get("missing_inputs", []))
    print(f"{name:<55} {up_sc:>6.1f} {up_bk:>6}  {op_sc:>6.1f} {op_bk:>6}  {conf:<12}  {str(cod):<12}  {opts:>4}  {miss:>4}")

print(f"\nTotal evals: {len(rows)}")

# Check for anomalies
print("\n=== ANOMALY CHECKS ===")
anomalies = 0
for d in rows:
    req = d.get("request", {})
    risk = d.get("risk", {})
    te = risk.get("timeline_estimate", {})
    name = req.get("project_name", "?")
    up = risk.get("upgrade_score") or 0
    ops = risk.get("ops_score") or 0
    up_b = risk.get("upgrade_exposure_bucket", "?")
    ops_b = risk.get("operational_exposure_bucket", "?")

    # Check threshold consistency
    if up >= 5 and up_b != "high":
        print(f"  !! {name}: upgrade score {up} but bucket={up_b} (should be high)"); anomalies += 1
    if 2 <= up < 5 and up_b != "medium":
        print(f"  !! {name}: upgrade score {up} but bucket={up_b} (should be medium)"); anomalies += 1
    if up < 2 and up_b != "low":
        print(f"  !! {name}: upgrade score {up} but bucket={up_b} (should be low)"); anomalies += 1
    if ops >= 6 and ops_b != "high":
        print(f"  !! {name}: ops score {ops} but bucket={ops_b} (should be high)"); anomalies += 1
    if 3 <= ops < 6 and ops_b != "medium":
        print(f"  !! {name}: ops score {ops} but bucket={ops_b} (should be medium)"); anomalies += 1
    if ops < 3 and ops_b != "low":
        print(f"  !! {name}: ops score {ops} but bucket={ops_b} (should be low)"); anomalies += 1

    # Check confidence logic
    miss = len(d.get("missing_inputs", []))
    mat = len(req.get("material_change_flags", []))
    conf = te.get("confidence", "?")
    te_status = te.get("status", "")
    # Unanchored timeline legitimately drives confidence to low even with 0 missing inputs
    if miss == 0 and mat == 0 and conf != "medium-high" and te_status != "unanchored":
        print(f"  !! {name}: no missing inputs, no material changes, timeline={te_status}, but confidence={conf}"); anomalies += 1
    if miss > 5 and conf != "low":
        print(f"  !! {name}: {miss} missing inputs but confidence={conf}"); anomalies += 1

    # Check decision present
    dec = d.get("decision", {})
    if not dec or not dec.get("mode"):
        print(f"  !! {name}: missing decision"); anomalies += 1

    # Check top_drivers present
    if not d.get("top_drivers"):
        print(f"  !! {name}: empty top_drivers"); anomalies += 1

    # Check graph traversal
    g = d.get("graph", {})
    if not g.get("traversed_edges"):
        print(f"  !! {name}: empty graph edges"); anomalies += 1

    # Check risk_trace is non-empty list
    rt = d.get("risk_trace", [])
    if not isinstance(rt, list) or len(rt) == 0:
        print(f"  !! {name}: risk_trace empty or wrong type ({type(rt).__name__})"); anomalies += 1

    # Check rule_checks present
    rc = d.get("rule_checks", [])
    if not rc:
        print(f"  !! {name}: no rule_checks!"); anomalies += 1

print(f"\n=== OPTION-LEVEL CHECKS ===")
opt_anomalies = 0
for d in rows:
    req = d.get("request", {})
    name = req.get("project_name", "?")
    for i, opt in enumerate(d.get("options", [])):
        oid = opt.get("option_id", f"opt{i}")
        s = opt.get("summary", {})
        opt_up_b = s.get("upgrade_exposure_bucket", "?")
        opt_ops_b = s.get("operational_exposure_bucket", "?")
        valid_buckets = ("low", "medium", "high", "unknown")
        if opt_up_b not in valid_buckets:
            print(f"  !! {name}/{oid}: invalid upgrade bucket={opt_up_b}"); opt_anomalies += 1
        if opt_ops_b not in valid_buckets:
            print(f"  !! {name}/{oid}: invalid ops bucket={opt_ops_b}"); opt_anomalies += 1

print(f"\n=== SUMMARY ===")
print(f"Total evals: {len(rows)}")
print(f"Top-level anomalies: {anomalies}")
print(f"Option-level anomalies: {opt_anomalies}")
print("Done.")
