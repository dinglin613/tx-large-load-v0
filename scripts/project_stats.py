from __future__ import annotations

from pathlib import Path
import collections
import json

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def main() -> None:
    rules_path = REPO_ROOT / "rules" / "published.jsonl"
    graph_path = REPO_ROOT / "graph" / "process_graph.yaml"

    rules = read_jsonl(rules_path)
    published = [r for r in rules if r.get("review_status") == "published"]

    stage = collections.Counter([r.get("stage") or "(missing)" for r in published])
    operator_stack = collections.Counter([
        "/".join(r.get("operator_stack") or []) or "(missing)" for r in published
    ])
    doc_id = collections.Counter([r.get("doc_id") or "(missing)" for r in published])
    trigger_tags = collections.Counter([t for r in published for t in (r.get("trigger_tags") or [])])

    graph = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    levers = graph.get("levers_catalog") or []

    print("== Rules ==")
    print(f"published_rules: {len(published)}")
    print(f"distinct_stage: {len(stage)}")
    for k, v in stage.most_common(50):
        print(f"  stage {k}: {v}")

    print(f"distinct_operator_stack: {len(operator_stack)}")
    for k, v in operator_stack.most_common(50):
        print(f"  operator_stack {k}: {v}")

    print(f"distinct_doc_id: {len(doc_id)}")
    for k, v in doc_id.most_common(50):
        print(f"  doc_id {k}: {v}")

    print(f"distinct_trigger_tags: {len(trigger_tags)}")
    for k, v in trigger_tags.most_common(80):
        print(f"  trigger_tag {k}: {v}")

    print("\n== Graph ==")
    print(f"graph_nodes: {len(nodes)}")
    print(f"graph_edges: {len(edges)}")
    print(f"graph_levers: {len(levers)}")

    print("\n== Levers ==")
    for lv in levers:
        lever_id = lv.get("lever_id")
        lever_class = lv.get("class")
        print(f"  lever {lever_id} (class={lever_class})")


if __name__ == "__main__":
    main()
