"""Quick audit: verify all IA/UX improvements are present in the local template and rendered output."""
import os

tpl = open("memo/templates/memo_template.html").read()

# Also check a rendered memo for things injected via Python (not template literals)
rendered_sample = ""
sample_path = "memo/outputs/Google_NorthTexas_250MW_BatchZeroA_2026Q4.html"
if not os.path.exists(sample_path):
    # Fallback: pick the first available HTML memo
    for fn in sorted(os.listdir("memo/outputs")):
        if fn.endswith(".html"):
            sample_path = f"memo/outputs/{fn}"
            break
if os.path.exists(sample_path):
    rendered_sample = open(sample_path).read()

checks = {
    "CSS exec/analyst hide rule":              "[data-mode=\"exec\"] .analyst-only { display: none" in tpl,
    "JS setMode() function":                   "function setMode(m)" in tpl,
    "Default URL mode param":                  "get(\"mode\")" in tpl,
    "sec-brief GONE from template":            'id="sec-brief"' not in tpl,
    "sec-summary GONE from template":          'id="sec-summary"' not in tpl,
    "sec-scope GONE from template":            'id="sec-scope"' not in tpl,
    "sec-decision present":                    'id="sec-decision"' in tpl,
    "decision_status_badge chips REMOVED":     '{{decision_status_badge}}' not in tpl,
    "decision_reason_chips REMOVED":           '{{decision_reason_chips}}' not in tpl,
    "Options at a glance (in rendered output)": "Options at a glance" in rendered_sample,
    "★ recommended option in rendered output":  "★" in rendered_sample,
    "sec-context analyst-only":                'class="card analyst-only" id="sec-context"' in tpl,
    "sec-path analyst-only":                   'class="card analyst-only" id="sec-path"' in tpl,
    "sec-missing analyst-only":                'class="card analyst-only" id="sec-missing"' in tpl,
    "sec-uncertainty analyst-only":            'class="card analyst-only" id="sec-uncertainty"' in tpl,
    "sec-levers analyst-only":                 'class="card analyst-only" id="sec-levers"' in tpl,
    "sec-flags analyst-only":                  'class="card analyst-only" id="sec-flags"' in tpl,
    "sec-risk analyst-only":                   'class="card analyst-only" id="sec-risk"' in tpl,
    "sec-provenance analyst-only":             'class="card analyst-only" id="sec-provenance"' in tpl,
    "full_graph_svg inside analyst-only":      'analyst-only' in tpl and 'full_graph_svg' in tpl,
    "risk_trace_json analyst-only":            'risk_trace_json' in tpl,
    "Project details kv table":                'Operator area' in tpl and 'class="kv"' in tpl,
    "Raw JSON in collapsed details":           'Show raw request JSON' in tpl,
    "Forwardable tagline":                     'Forwardable' in tpl,
    "One-line disclaimer":                     'Screening brief only' in tpl,
}

all_ok = True
for label, result in checks.items():
    icon = "OK " if result else "FAIL"
    if not result:
        all_ok = False
    print(f"  [{icon}]  {label}")

print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
