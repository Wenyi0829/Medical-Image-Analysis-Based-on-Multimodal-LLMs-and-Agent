#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation-based evaluation of tool usefulness for the functional-calling medical VQA agent.

Main metric (2): delta between full tools and no-tools (user-selected main_metric).
Secondary metric (1): tool call/execution statistics parsed from reasoning_trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agent_quick_test as agent_module  # noqa: E402
from tool_eval.runner import evaluate_variant  # noqa: E402
from tool_eval.tool_stats import extract_tool_stats_from_traces  # noqa: E402


def load_model(model_path: str, lora_path: str | None = None):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, processor


def get_registered_tools() -> List[str]:
    # Prefer TOOL_REGISTRY keys because those are actually executable.
    return sorted(list(agent_module.TOOL_REGISTRY.keys()))


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool usefulness for tool-calling agent.")
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
        help="Primary contribution metric for ablation full vs no_tools.",
    )
    parser.add_argument(
        "--remove_each",
        action="store_true",
        default=False,
        help="Also evaluate ablations removing each tool individually.",
    )
    parser.add_argument(
        "--save_reasoning_trace",
        action="store_true",
        default=False,
        help="Save full reasoning_trace for all variants (can be large). "
        "By default, only full variant keeps traces needed for tool stats.",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/wshenah/project/eval_results/tool_usefulness_local_{TIMESTAMP}"
    os.makedirs(output_dir, exist_ok=True)

    samples = agent_module.parse_val_dataset(args.val_dataset, sample_size=args.sample_size)
    if not samples:
        raise RuntimeError(f"No samples loaded from: {args.val_dataset}")

    model, processor = load_model(args.base_model, args.lora_path)

    config: Dict[str, Any] = dict(agent_module.DEFAULT_CONFIG)
    config["max_agent_steps"] = args.max_steps

    all_tools = get_registered_tools()
    no_tools: List[str] = []
    full_tools = all_tools

    variants: List[tuple[str, List[str]]] = [
        ("full", full_tools),
        ("no_tools", no_tools),
    ]

    if args.remove_each:
        for t in all_tools:
            variants.append((f"remove_{t}", [x for x in all_tools if x != t]))

    # Only full variant needs reasoning_trace for tool-level stats.
    save_detail_for_full = True
    save_detail_for_others = bool(args.save_reasoning_trace)

    variant_summaries: Dict[str, Dict[str, Any]] = {}
    variant_results: Dict[str, List[Dict[str, Any]]] = {}

    for name, enabled in variants:
        save_detail = save_detail_for_full if name == "full" else save_detail_for_others
        subdir = os.path.join(output_dir, name)
        vr = evaluate_variant(
            variant_name=name,
            enabled_tool_names=enabled,
            samples=samples,
            model=model,
            processor=processor,
            config=config,
            output_dir=subdir,
            agent_module=agent_module,
            save_detail=save_detail,
        )
        variant_summaries[name] = vr.summary
        variant_results[name] = vr.results

    # Contribution main metric delta (2)
    full_metric = variant_summaries["full"]["metrics"][args.main_metric]
    no_tools_metric = variant_summaries["no_tools"]["metrics"][args.main_metric]
    main_delta = full_metric - no_tools_metric

    # Side deltas for quick comparison
    metric_keys = ["bleu_mean", "rouge_l_mean", "exact_match_mean", "medical_acc_mean"]
    side_deltas = {k: variant_summaries["full"]["metrics"][k] - variant_summaries["no_tools"]["metrics"][k] for k in metric_keys}

    # Tool-level stats from full variant (1)
    full_results = variant_results["full"]
    # extract_tool_stats_from_traces expects reasoning_trace key; full_results keep it by default.
    tool_level_stats = extract_tool_stats_from_traces(full_results, all_tools)

    report = {
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "all_tools": all_tools,
        "main_metric": args.main_metric,
        "main_delta_full_minus_no_tools": main_delta,
        "side_deltas_full_minus_no_tools": side_deltas,
        "tool_level_stats_full": tool_level_stats,
        "variant_summaries": variant_summaries,
    }

    report_path = os.path.join(output_dir, "tool_usefulness_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. Report saved to: {report_path}")


if __name__ == "__main__":
    main()

