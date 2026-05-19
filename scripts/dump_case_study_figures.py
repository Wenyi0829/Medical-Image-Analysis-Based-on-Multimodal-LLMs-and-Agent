#!/usr/bin/env python3
"""
Produce artifacts for thesis figures:
  Case 1: analyze_medical_image JSON (requires CUDA + project model).
  Case 2: search_drug_label retrieval + formatted Evidence block (CPU; avoids importing agent_quick_test).

Usage:
  python dump_case_study_figures.py --case drug
  python dump_case_study_figures.py --case image   # needs GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _format_drug_results(query: str, results: list) -> dict:
    formatted = []
    for i, r in enumerate(results):
        md = r.get("metadata", {}) or {}
        formatted.append(
            {
                "rank": i + 1,
                "score": round(r.get("score", 0.0), 4),
                "passage": r.get("text", ""),
                "source": md.get("source", "openfda_drug_label"),
                "set_id": md.get("set_id"),
                "effective_time": md.get("effective_time"),
            }
        )
    return {"status": "success", "query": query, "results": formatted}


def dump_drug_case(
    *,
    query: str,
    index_dir: str,
    manifest: str,
    out_json: str,
) -> None:
    from rag.medical_rag import MedicalRAG

    rag = MedicalRAG(
        index_dir=index_dir,
        corpus_manifest_path=manifest,
        force_rebuild=False,
    )
    results = rag.retrieve(query, top_k=3)
    payload = _format_drug_results(query, results)

    lines = []
    lines.append("=== Case study 2 (Drug-label RAG) ===\n")
    lines.append(f"User query (English):\n{query}\n")
    lines.append("Retrieved passages (top-3, truncated for display):\n")
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        pas = (item.get("passage") or "")[:520].replace("\n", " ")
        lines.append(
            f"  rank={item.get('rank')} score={item.get('score')} "
            f"set_id={item.get('set_id')} effective_time={item.get('effective_time')}\n"
            f"  excerpt: {pas}...\n"
        )

    first = (payload.get("results") or [{}])[0]
    excerpt = ((first.get("passage") or "")[:280]).replace("\n", " ")
    evidence_lines = [
        "### Brief answer (example framing for the figure)",
        "",
        "Warfarin sodium carries a boxed warning for bleeding risk. When antibiotics or antifungals",
        "are started or stopped, INR should be monitored closely because INR shifts have been reported,",
        "even though pharmacokinetic studies have not shown consistent effects on warfarin concentrations.",
        "",
        "**Evidence:**",
        f"- source={first.get('source', 'openfda_drug_label')}, set_id={first.get('set_id')}, "
        f"effective_time={first.get('effective_time')}, excerpt={excerpt}...",
        "",
    ]
    lines.extend(evidence_lines)

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case": "drug_rag",
                "query": query,
                "tool_response": payload,
                "figure_markdown": text,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(text)


def dump_image_case(
    *,
    image_path: str,
    question: str,
    base_model: str,
    lora_path: str,
    out_json: str,
) -> None:
    import agent_quick_test as agent  # noqa: PLC0415 — deferred heavy import

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; cannot run analyze_medical_image on this machine.", file=sys.stderr)
        sys.exit(2)

    model, processor = agent.load_finetuned_model(base_model, lora_path)
    agent.set_tool_context(model=model, processor=processor)
    agent.TOOL_CONTEXT["current_image_path"] = image_path
    agent.TOOL_CONTEXT["current_question"] = question

    raw = agent.tool_analyze_medical_image(
        image_path=image_path,
        analysis_type="general",
        focus=None,
    )
    payload = json.loads(raw)

    lines = []
    lines.append("=== Case study 1 (Multimodal image QA + uncertainty) ===\n")
    lines.append(f"User question (English):\n{question}\n")
    lines.append(f"Image file:\n{payload.get('image_path')}\n")
    lines.append("Structured tool output (JSON fields):\n")
    for k in (
        "modality",
        "anatomy",
        "laterality",
        "findings",
        "key_findings",
        "impression",
        "uncertainty",
    ):
        lines.append(f"  {k}: {payload.get(k)!r}")
    text = "\n".join(lines)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case": "image_qa",
                "question": question,
                "image_path": image_path,
                "tool_response": payload,
                "figure_markdown": text,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("drug", "image", "both"), default="both")
    p.add_argument(
        "--drug_query",
        default=(
            "What does the FDA label say about warfarin interactions with antibiotics? "
            "Summarize monitoring recommendations for a clinician."
        ),
    )
    p.add_argument(
        "--rag_index_dir",
        default=os.path.join(PROJECT_ROOT, "rag_index", "openfda_label"),
    )
    p.add_argument(
        "--rag_manifest_path",
        default=os.path.join(PROJECT_ROOT, "rag", "corpus_manifest_openfda_label.json"),
    )
    p.add_argument("--image_path", default="/home/wshenah/LLaVA-Med/data/images/18627621_F2.jpg")
    p.add_argument(
        "--image_question",
        default=(
            "This is a chest radiograph. Please summarize the main imaging findings and highlight "
            "any limitations of interpreting this single projection."
        ),
    )
    p.add_argument("--base_model", default=os.path.join(PROJECT_ROOT, "models", "Qwen3-VL-8B-Thinking"))
    p.add_argument(
        "--lora_path",
        default=os.path.join(PROJECT_ROOT, "lora", "v14-20260306-195347", "checkpoint-1200"),
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(PROJECT_ROOT, "eval_results", "case_study_figure_dump"),
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.case in ("drug", "both"):
        dump_drug_case(
            query=args.drug_query,
            index_dir=args.rag_index_dir,
            manifest=args.rag_manifest_path,
            out_json=os.path.join(args.out_dir, "case_study_drug_rag.json"),
        )
    if args.case in ("image", "both"):
        dump_image_case(
            image_path=args.image_path,
            question=args.image_question,
            base_model=args.base_model,
            lora_path=args.lora_path,
            out_json=os.path.join(args.out_dir, "case_study_image_qa.json"),
        )


if __name__ == "__main__":
    main()
