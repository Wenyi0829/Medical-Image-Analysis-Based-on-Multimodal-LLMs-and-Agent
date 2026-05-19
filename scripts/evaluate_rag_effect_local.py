#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate RAG contribution for local medical agent inference.

Runs two variants on the same dataset with real model inference:
1) rag_on: all tools enabled (including search_drug_label) with initialized RAG.
2) rag_off: same tool set but search_drug_label removed.

Outputs metric deltas and tool-call stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agent_quick_test as agent_module  # noqa: E402
from tool_eval.runner import evaluate_variant  # noqa: E402
from tool_eval.tool_stats import extract_tool_stats_from_traces  # noqa: E402


def load_model(model_path: str, lora_path: str | None = None):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    attn_impl = "flash_attention_2" if use_cuda else "eager"
    device_map = "auto" if use_cuda else None

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
        trust_remote_code=True,
    )
    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, processor


def maybe_init_rag(index_dir: str, manifest_path: str, rebuild: bool):
    if not getattr(agent_module, "_RAG_AVAILABLE", False):
        raise RuntimeError(
            "RAG deps unavailable in this environment. Install faiss-cpu and sentence-transformers."
        )
    # Reuse existing index config to avoid accidental full rebuild.
    cfg_path = os.path.join(index_dir, "medical_rag_config.json")
    rag_kwargs: Dict[str, Any] = {}
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
                    rag_kwargs[key] = cfg[key]
        except Exception:
            pass

    rag = agent_module.MedicalRAG(
        index_dir=index_dir,
        corpus_manifest_path=manifest_path,
        force_rebuild=rebuild,
        **rag_kwargs,
    )
    return rag


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG contribution (on/off) with local model inference.")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument(
        "--main_metric",
        type=str,
        default="rouge_l_mean",
        choices=["bleu_mean", "rouge_l_mean", "exact_match_mean", "medical_acc_mean"],
    )
    parser.add_argument(
        "--rag_index_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "rag_index", "openfda_label"),
    )
    parser.add_argument(
        "--rag_manifest_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "rag", "corpus_manifest_openfda_label.json"),
    )
    parser.add_argument("--rebuild_rag", action="store_true", default=False)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/wshenah/project/eval_results/rag_effect_local_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    samples = agent_module.parse_val_dataset(args.val_dataset, sample_size=args.sample_size)
    if not samples:
        raise RuntimeError(f"No samples loaded from: {args.val_dataset}")

    model, processor = load_model(args.base_model, args.lora_path)
    rag = maybe_init_rag(
        index_dir=args.rag_index_dir,
        manifest_path=args.rag_manifest_path,
        rebuild=args.rebuild_rag,
    )
    # Keep rag in global tool context; run_agent_inference updates per-sample model/image/question only.
    agent_module.set_tool_context(model=model, processor=processor, rag=rag)

    config: Dict[str, Any] = dict(agent_module.DEFAULT_CONFIG)
    config["max_agent_steps"] = args.max_steps

    all_tools = sorted(list(agent_module.TOOL_REGISTRY.keys()))
    rag_tool = "search_drug_label"
    if rag_tool not in all_tools:
        raise RuntimeError(
            f"Tool `{rag_tool}` not found in TOOL_REGISTRY. Current tools: {all_tools}"
        )

    rag_on_tools = list(all_tools)
    rag_off_tools = [t for t in all_tools if t != rag_tool]

    on_result = evaluate_variant(
        variant_name="rag_on",
        enabled_tool_names=rag_on_tools,
        samples=samples,
        model=model,
        processor=processor,
        config=config,
        output_dir=os.path.join(output_dir, "rag_on"),
        agent_module=agent_module,
        save_detail=True,
    )
    off_result = evaluate_variant(
        variant_name="rag_off",
        enabled_tool_names=rag_off_tools,
        samples=samples,
        model=model,
        processor=processor,
        config=config,
        output_dir=os.path.join(output_dir, "rag_off"),
        agent_module=agent_module,
        save_detail=True,
    )

    metric_keys = ["bleu_mean", "rouge_l_mean", "exact_match_mean", "medical_acc_mean"]
    rag_on_metrics = on_result.summary["metrics"]
    rag_off_metrics = off_result.summary["metrics"]
    deltas = {k: rag_on_metrics[k] - rag_off_metrics[k] for k in metric_keys}

    report = {
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "main_metric": args.main_metric,
        "main_delta_rag_on_minus_rag_off": deltas[args.main_metric],
        "side_deltas_rag_on_minus_rag_off": deltas,
        "rag_tool": rag_tool,
        "rag_on_tools": rag_on_tools,
        "rag_off_tools": rag_off_tools,
        "tool_stats_rag_on": extract_tool_stats_from_traces(on_result.results, rag_on_tools),
        "tool_stats_rag_off": extract_tool_stats_from_traces(off_result.results, rag_off_tools),
        "summary_rag_on": on_result.summary,
        "summary_rag_off": off_result.summary,
    }

    report_path = os.path.join(output_dir, "rag_effect_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. RAG effect report saved to: {report_path}")


if __name__ == "__main__":
    main()

