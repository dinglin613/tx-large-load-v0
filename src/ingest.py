from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_RAW_DIR = REPO_ROOT / "docs" / "raw"
REGISTRY_PATH = REPO_ROOT / "registry" / "doc_registry.json"


def _iso_date_from_mtime(p: Path) -> str:
    dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return dt.date().isoformat()


def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class MatchRule:
    doc_id: str
    filename_contains: Tuple[str, ...]


DEFAULT_MATCH_RULES: List[MatchRule] = [
    MatchRule("ERCOT_PLANNING_GUIDE", ("Planning Guide",)),
    MatchRule("ERCOT_LARGE_LOAD", ("Large-Load-Interconnection-Process", "LLI_Standalone_Energization_Request")),
    MatchRule("ONCOR_STD_520_106", ("Oncor Standard 520-106", "520-106", "500-251")),
]


def load_registry() -> List[Dict[str, Any]]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: List[Dict[str, Any]]) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        f.write("\n")


def index_registry(registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in registry:
        out[d["doc_id"]] = d
    return out


def ensure_artifacts_shape(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts = doc.get("artifacts")
    if artifacts is None:
        artifacts = []
        doc["artifacts"] = artifacts
    if not isinstance(artifacts, list):
        raise ValueError(f"doc_id={doc.get('doc_id')} artifacts must be a list")
    return artifacts


def artifact_key(a: Dict[str, Any]) -> str:
    return str(a.get("path", "")).replace("\\", "/").lower()


def upsert_artifact(doc: Dict[str, Any], *, kind: str, path: Path) -> None:
    artifacts = ensure_artifacts_shape(doc)
    rel = path.relative_to(REPO_ROOT).as_posix()
    k = rel.lower()
    existing = {artifact_key(a): a for a in artifacts}
    a = existing.get(k)
    if a is None:
        a = {"kind": kind, "path": rel}
        artifacts.append(a)

    a["kind"] = kind
    a["path"] = rel
    a["size_bytes"] = int(path.stat().st_size)
    a["sha256"] = sha256_file(path)
    a["retrieved_date"] = a.get("retrieved_date") or _iso_date_from_mtime(path)


def match_doc_id(filename: str, match_rules: List[MatchRule]) -> Optional[str]:
    name = filename.lower()
    for r in match_rules:
        if any(token.lower() in name for token in r.filename_contains):
            return r.doc_id
    return None


def main() -> int:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")
    if not DOCS_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw docs dir: {DOCS_RAW_DIR}")

    registry = load_registry()
    by_id = index_registry(registry)

    raw_files = sorted([p for p in DOCS_RAW_DIR.glob("**/*") if p.is_file()])
    if not raw_files:
        print(f"No files found under {DOCS_RAW_DIR}")
        return 0

    unregistered: List[Path] = []
    updated_docs: set[str] = set()

    for p in raw_files:
        doc_id = match_doc_id(p.name, DEFAULT_MATCH_RULES)
        if doc_id is None or doc_id not in by_id:
            unregistered.append(p)
            continue
        upsert_artifact(by_id[doc_id], kind="pdf", path=p)
        updated_docs.add(doc_id)

    # Fill top-level convenience fields (backward compat)
    for doc_id in updated_docs:
        doc = by_id[doc_id]
        arts = ensure_artifacts_shape(doc)
        if arts:
            # keep original "hash" field populated with first artifact sha
            doc["hash"] = doc.get("hash") or arts[0].get("sha256", "")
            doc["retrieved_date"] = doc.get("retrieved_date") or arts[0].get("retrieved_date", "")

    save_registry(registry)

    print(f"Updated registry: {REGISTRY_PATH}")
    print(f"Matched docs: {sorted(updated_docs)}")
    if unregistered:
        print("\nUnregistered files found (consider adding to registry or match rules):")
        for p in unregistered:
            rel = p.relative_to(REPO_ROOT).as_posix()
            print(f"- {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
