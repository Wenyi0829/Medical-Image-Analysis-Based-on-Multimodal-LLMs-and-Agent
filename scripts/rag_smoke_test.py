#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal smoke test for OpenFDA label RAG.

This script does NOT call any remote API model. It validates:
1) retrieval works for fixed drug queries
2) each query returns top-k passages with key metadata
3) a basic answer template is produced with evidence
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from rag.medical_rag import MedicalRAG
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False


DEFAULT_QUERIES = [
    "warfarin interactions",
    "metformin contraindications",
    "isotretinoin pregnancy warning",
    "amiodarone adverse reactions",
    "ibuprofen dosage",
    "lisinopril renal dose adjustment",
    "clozapine black box warning",
    "valproate pregnancy risk",
    "digoxin toxicity signs",
    "rivaroxaban contraindications",
]


def build_basic_answer(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return (
            f"结论：未检索到与问题 `{query}` 直接相关的 OpenFDA 标签片段。\n\n"
            "依据要点：建议改写为药名 + 具体维度（如 contraindication/interaction/dosage）后重试。\n\n"
            "Evidence:\n"
            "- source=openfda_drug_label, set_id=None, effective_time=None, excerpt=None"
        )

    top = results[0]
    md = top.get("metadata", {}) or {}
    excerpt = (top.get("text", "") or "").strip()
    if len(excerpt) > 280:
        excerpt = excerpt[:280] + "..."

    answer = (
        "结论：根据 OpenFDA drug label 检索结果，当前问题可从标签中找到直接依据，请结合患者具体情况审慎使用。\n\n"
        "依据要点：\n"
        f"- 命中最高相关片段（score={top.get('score', 0.0):.4f}），提示与 `{query}` 相关的标签信息已覆盖。\n"
        "- 建议在临床使用前结合禁忌、相互作用和特殊人群用药条款综合判断。\n\n"
        "Evidence:\n"
        f"- source={md.get('source', 'openfda_drug_label')}, "
        f"set_id={md.get('set_id')}, "
        f"effective_time={md.get('effective_time')}, "
        f"excerpt={excerpt}"
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenFDA RAG minimal smoke test")
    parser.add_argument(
        "--index_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "rag_index", "openfda_label"),
        help="RAG index directory.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(PROJECT_ROOT, "rag", "corpus_manifest_openfda_label.json"),
        help="Manifest path for OpenFDA corpus.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Top-k retrieval.")
    parser.add_argument(
        "--queries_file",
        type=str,
        default="",
        help="Optional query file (.txt one per line or .jsonl with `query` field).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "eval_results", f"rag_smoke_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    if not _RAG_AVAILABLE:
        summary = {
            "status": "skipped",
            "reason": "missing_dependencies",
            "message": "RAG requires `faiss-cpu` and `sentence-transformers`.",
            "hint": "Install with: pip install faiss-cpu sentence-transformers",
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved smoke results to: {out_dir}")
        return

    queries = list(DEFAULT_QUERIES)
    if args.queries_file:
        loaded: List[str] = []
        if args.queries_file.endswith(".jsonl"):
            with open(args.queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    q = str(obj.get("query", "")).strip()
                    if q:
                        loaded.append(q)
        else:
            with open(args.queries_file, "r", encoding="utf-8") as f:
                for line in f:
                    q = line.strip()
                    if q:
                        loaded.append(q)
        if loaded:
            queries = loaded

    chunk_kwargs: Dict[str, Any] = {}
    cfg_path = os.path.join(args.index_dir, "medical_rag_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in (
                "embed_model_name",
                "chunk_size_words",
                "chunk_overlap_words",
                "min_chunk_words",
                "min_text_chars",
            ):
                if key in cfg:
                    chunk_kwargs[key] = cfg[key]
        except Exception:
            pass

    try:
        rag = MedicalRAG(
            index_dir=args.index_dir,
            corpus_manifest_path=args.manifest,
            force_rebuild=False,
            top_k=args.top_k,
            **chunk_kwargs,
        )
    except RuntimeError as e:
        summary = {
            "status": "skipped",
            "reason": "rag_unavailable",
            "message": str(e),
            "hint": "Install with: pip install faiss-cpu sentence-transformers",
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved smoke results to: {out_dir}")
        return

    records: List[Dict[str, Any]] = []
    hit_count = 0
    for q in queries:
        retrieved = rag.retrieve(q, top_k=args.top_k)
        hit = len(retrieved) > 0
        if hit:
            hit_count += 1
        answer = build_basic_answer(q, retrieved)
        records.append(
            {
                "query": q,
                "hit": hit,
                "top_k": [
                    {
                        "rank": i + 1,
                        "score": float(r.get("score", 0.0)),
                        "passage": r.get("text", ""),
                        "source": (r.get("metadata", {}) or {}).get("source"),
                        "set_id": (r.get("metadata", {}) or {}).get("set_id"),
                        "effective_time": (r.get("metadata", {}) or {}).get("effective_time"),
                    }
                    for i, r in enumerate(retrieved)
                ],
                "answer": answer,
            }
        )

    summary = {
        "num_queries": len(queries),
        "hit_count": hit_count,
        "hit_rate": (hit_count / len(queries)) if queries else 0.0,
        "top_k": args.top_k,
        "index_dir": args.index_dir,
        "manifest": args.manifest,
    }

    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved smoke results to: {out_dir}")


if __name__ == "__main__":
    main()

