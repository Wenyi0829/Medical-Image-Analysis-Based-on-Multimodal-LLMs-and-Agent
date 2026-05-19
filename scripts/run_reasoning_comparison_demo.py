#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPT-oriented ablation runs on the same JSONL cases: baseline vs LoRA, RAG on vs off (drug),
tools on vs off (image). Writes one JSON with side-by-side `comparisons` per case.

Designed for small case counts (defense_demo_cases.jsonl); keep --max_new_tokens modest for queue time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import agent_quick_test as aq  # noqa: E402
from tool_eval.agent_patch import patch_agent_tools  # noqa: E402


FULL_TOOLS = ["analyze_medical_image", "search_drug_label", "safety_check_medical_answer"]


@contextmanager
def base_model_only(model: Any):
    """If model is a PEFT-wrapped model, temporarily disable adapters (baseline weights)."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def _load_rag(args: argparse.Namespace):
    if args.no_rag or not aq._RAG_AVAILABLE:
        return None
    from rag.medical_rag import MedicalRAG  # noqa: E402

    rag_kwargs: dict = {}
    corpus_manifest_path = args.rag_manifest_path
    corpus_sources_arg: list | None = None
    cfg_path = os.path.join(args.rag_index_dir, "medical_rag_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as cf:
                prev = json.load(cf)
            for key in (
                "chunk_size_words",
                "chunk_overlap_words",
                "min_chunk_words",
                "min_text_chars",
                "embed_model_name",
            ):
                if key in prev:
                    rag_kwargs[key] = prev[key]
            if isinstance(prev.get("corpus_sources"), list) and prev["corpus_sources"]:
                normalized_sources = []
                for s in prev["corpus_sources"]:
                    if not isinstance(s, dict):
                        continue
                    d = dict(s)
                    d["optional"] = bool(d.get("optional", False))
                    if d.get("format", "jsonl") == "jsonl" and not d.get("text_fields"):
                        d["text_fields"] = ["text"]
                    normalized_sources.append(d)
                corpus_sources_arg = normalized_sources
                corpus_manifest_path = None
        except Exception as exc:
            print(f"[comparison] RAG config read failed ({exc}); using manifest defaults.", file=sys.stderr)

    return MedicalRAG(
        index_dir=args.rag_index_dir,
        corpus_manifest_path=corpus_manifest_path,
        corpus_sources=corpus_sources_arg,
        force_rebuild=False,
        **rag_kwargs,
    )


def _run_one(
    model,
    processor,
    question: str,
    image: str | None,
    cfg: dict,
    *,
    rag_obj,
    enabled_tools: List[str],
    use_baseline_weights: bool,
) -> Tuple[str, List[str]]:
    with patch_agent_tools(aq, enabled_tools):
        aq.set_tool_context(model=model, processor=processor, rag=rag_obj)
        ctx = base_model_only(model) if use_baseline_weights else nullcontext()
        with ctx:
            return aq.run_agent_inference(model, processor, question, image, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reasoning comparison matrix for PPT demos.")
    parser.add_argument("--cases_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=aq.DEFAULT_CONFIG["base_model_path"])
    parser.add_argument("--lora_path", type=str, default=aq.DEFAULT_CONFIG["lora_ckpt_dir"])
    parser.add_argument("--rag_index_dir", type=str, default="/home/wshenah/project/rag_index/openfda_label")
    parser.add_argument(
        "--rag_manifest_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "rag", "corpus_manifest_openfda_label.json"),
    )
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--no_rag", action="store_true", help="Skip RAG load; drug runs use parametric knowledge only.")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline (LoRA-off) runs to save time.")
    args = parser.parse_args()

    cfg = aq.DEFAULT_CONFIG.copy()
    cfg["base_model_path"] = args.base_model
    cfg["lora_ckpt_dir"] = args.lora_path
    cfg["max_agent_steps"] = args.max_steps
    cfg["max_new_tokens"] = args.max_new_tokens

    samples = aq.parse_val_dataset(args.cases_jsonl, sample_size=None)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"[comparison] Loading model {cfg['base_model_path']} + LoRA {cfg['lora_ckpt_dir']} ...")
    model, processor = aq.load_finetuned_model(cfg["base_model_path"], cfg["lora_ckpt_dir"])

    rag_full = None if args.no_rag else _load_rag(args)
    if rag_full is None and not args.no_rag:
        print("[comparison] RAG unavailable; continuing without index.", file=sys.stderr)

    aq.set_tool_context(model=model, processor=processor, rag=rag_full)

    axis_notes = {
        "baseline_full_stack": "基座权重 + 全工具 + RAG（与微调后对照，展示微调）",
        "finetuned_full_stack": "LoRA 微调 + 全工具 + RAG（系统默认）",
        "finetuned_no_rag": "LoRA + 全工具，但关闭 RAG（药品问题：无说明书检索与 Evidence）",
        "finetuned_no_tools": "LoRA + RAG，但禁用所有工具（影像问题：不调用 analyze_medical_image，直接多模态问答）",
    }

    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "max_agent_steps": cfg["max_agent_steps"],
            "max_new_tokens": cfg["max_new_tokens"],
            "base_model_path": cfg["base_model_path"],
            "lora_ckpt_dir": cfg["lora_ckpt_dir"],
            "rag_index_dir": None if args.no_rag else args.rag_index_dir,
            "cases_jsonl": args.cases_jsonl,
        },
        "comparison_axis_notes_zh": axis_notes,
        "cases": [],
    }

    for i, s in enumerate(samples):
        question = s["question"]
        image = s.get("image")
        ref = (s.get("answer") or "")[:800]
        is_drug = aq._allow_drug_label_tool(question)
        is_mm = bool(image)

        print(f"[comparison] Case {i + 1}/{len(samples)} (drug={is_drug}, multimodal={is_mm}) ...")
        comparisons: Dict[str, Any] = {}

        # --- 微调前后：同一套全栈，仅切换基座 / LoRA 权重 ---
        if not args.skip_baseline:
            pred, trace = _run_one(
                model,
                processor,
                question,
                image,
                cfg,
                rag_obj=rag_full,
                enabled_tools=FULL_TOOLS,
                use_baseline_weights=True,
            )
            comparisons["baseline_full_stack"] = {
                "label_zh": axis_notes["baseline_full_stack"],
                "model_output": pred,
                "reasoning_trace": trace,
            }

        pred, trace = _run_one(
            model,
            processor,
            question,
            image,
            cfg,
            rag_obj=rag_full,
            enabled_tools=FULL_TOOLS,
            use_baseline_weights=False,
        )
        comparisons["finetuned_full_stack"] = {
            "label_zh": axis_notes["finetuned_full_stack"],
            "model_output": pred,
            "reasoning_trace": trace,
        }

        # --- RAG 前后：仅药品且已加载 RAG 时 ---
        if is_drug and rag_full is not None:
            pred, trace = _run_one(
                model,
                processor,
                question,
                image,
                cfg,
                rag_obj=None,
                enabled_tools=FULL_TOOLS,
                use_baseline_weights=False,
            )
            comparisons["finetuned_no_rag"] = {
                "label_zh": axis_notes["finetuned_no_rag"],
                "model_output": pred,
                "reasoning_trace": trace,
            }

        # --- 工具调用前后：多模态样本 ---
        # 影像题：RAG 对纯影像问题通常不参与；rag_full 可为 None（仅对比「有无工具」）
        if is_mm:
            pred, trace = _run_one(
                model,
                processor,
                question,
                image,
                cfg,
                rag_obj=rag_full,
                enabled_tools=[],
                use_baseline_weights=False,
            )
            comparisons["finetuned_no_tools"] = {
                "label_zh": axis_notes["finetuned_no_tools"],
                "model_output": pred,
                "reasoning_trace": trace,
            }

        out["cases"].append(
            {
                "case_id": i + 1,
                "flags": {"drug_related": is_drug, "multimodal": is_mm},
                "input": {"question": question, "image": image},
                "dataset_reference_excerpt": ref + ("..." if len(s.get("answer") or "") > 800 else ""),
                "comparisons": comparisons,
            }
        )

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[comparison] Wrote {args.output_json}")


if __name__ == "__main__":
    main()
