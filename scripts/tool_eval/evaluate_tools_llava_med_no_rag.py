#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch tool-calling evaluation on llava_med_qwen_format_fixed.json (JSON array).

Goal:
- Evaluate tool calling effectiveness excluding RAG.
- Variants:
  - no_tools
  - analyze_only
  - analyze_plus_safety

This script:
- Streams the huge JSON array without loading it all into RAM.
- Extracts VQA samples (question, answer, image).
- Runs the agent loop (local transformers model).
- Outputs:
  - per-variant results.jsonl
  - summary.json per variant
  - combined report.json + metrics.csv
  - a comparison plot (png)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agent_quick_test as agent_module  # noqa: E402
from tool_eval.agent_patch import patch_agent_tools  # noqa: E402
from tool_eval.metrics import evaluate_text_metrics, get_rouge  # noqa: E402
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


def _stream_json_array(path: str, *, chunk_size: int = 1 << 20) -> Generator[Any, None, None]:
    """
    Stream a JSON array from disk using json.JSONDecoder.raw_decode.

    Works for:
      [ {...}, {...}, ... ]

    Without loading the whole file. No external deps.
    """
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0
    started = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk

            while True:
                # skip whitespace
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1

                if not started:
                    if idx < len(buf) and buf[idx] == "[":
                        started = True
                        idx += 1
                    else:
                        # need more data
                        break

                # skip whitespace and optional commas
                while idx < len(buf) and (buf[idx].isspace() or buf[idx] == ","):
                    idx += 1

                if idx >= len(buf):
                    break

                if buf[idx] == "]":
                    return

                try:
                    obj, new_idx = decoder.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    # need more data
                    break
                yield obj
                idx = new_idx

            # trim consumed buffer occasionally
            if idx > 0 and idx > (1 << 20):
                buf = buf[idx:]
                idx = 0


def _extract_first_qa(messages: Any) -> Tuple[Optional[str], str, str]:
    """
    Extract first (image_path, question_text, answer_text) from a message list.
    Compatible with your agent_quick_test.parse_val_dataset conventions.
    """
    image_path = None
    question_text = ""
    answer_text = ""

    if not isinstance(messages, list):
        return None, "", ""

    current_q = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            current_q = content
        elif role == "assistant" and current_q:
            # parse user content blocks
            if isinstance(current_q, list):
                for item in current_q:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            image_path = item.get("image")
                        elif item.get("type") == "text":
                            question_text = item.get("text", "") or question_text
                    elif isinstance(item, str):
                        question_text = item or question_text
            elif isinstance(current_q, str):
                question_text = current_q

            # parse assistant content blocks
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    answer_text = str(first.get("text", "") or "")
                else:
                    answer_text = str(first)
            elif isinstance(content, str):
                answer_text = content
            break

    return image_path, (question_text or "").strip(), (answer_text or "").strip()


def load_llava_med_samples(
    path: str,
    *,
    limit: Optional[int] = None,
    start_offset: int = 0,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    seen = 0
    for obj in _stream_json_array(path):
        if not isinstance(obj, dict):
            continue
        messages = obj.get("messages")
        image, question, answer = _extract_first_qa(messages)
        if not question or not answer:
            continue

        if seen < start_offset:
            seen += 1
            continue
        seen += 1

        samples.append({"image": image, "question": question, "answer": answer})
        if limit is not None and len(samples) >= limit:
            break
    return samples


def mean_or_zero(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


_STEP_RE = re.compile(r"Step\s+(?P<n>\d+)\s+Model Output:")


def _agent_steps_from_trace(trace: List[str]) -> int:
    """
    Estimate agent loop steps from reasoning_trace.

    - Primary: max N from lines like "Step N Model Output: ..."
    - If deterministic safety is applied after the loop, count +1 "virtual step"
      so avg_steps isn't degenerate when we do post-check tooling.
    """
    max_step = 0
    has_det_safety = False
    for line in trace or []:
        m = _STEP_RE.search(line)
        if m:
            try:
                max_step = max(max_step, int(m.group("n")))
            except Exception:
                pass
        if "Deterministic Safety Tool Call:" in line:
            has_det_safety = True
    if max_step <= 0:
        max_step = 1 if trace else 0
    if has_det_safety:
        max_step += 1
    return int(max_step)


def evaluate_variant(
    *,
    name: str,
    enabled_tools: List[str],
    samples: List[Dict[str, Any]],
    model: Any,
    processor: Any,
    config: Dict[str, Any],
    out_dir: str,
    save_trace: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    rouge = get_rouge()

    metrics_acc: Dict[str, List[float]] = {"bleu": [], "rouge_l": [], "exact_match": [], "medical_acc": []}
    results: List[Dict[str, Any]] = []

    with patch_agent_tools(agent_module, enabled_tool_names=enabled_tools):
        for i, s in enumerate(samples):
            pred, trace = agent_module.run_agent_inference(
                model=model,
                processor=processor,
                question=s["question"],
                image_path=s.get("image"),
                config=config,
            )
            bleu, rouge_l, em, med_acc = evaluate_text_metrics(rouge, s["answer"], pred)
            metrics_acc["bleu"].append(bleu)
            metrics_acc["rouge_l"].append(rouge_l)
            metrics_acc["exact_match"].append(em)
            metrics_acc["medical_acc"].append(med_acc)

            rec: Dict[str, Any] = {
                "idx": i,
                "question": s["question"],
                "image": os.path.basename(s["image"]) if s.get("image") else "",
                "reference": s["answer"],
                "prediction": pred,
                "bleu": bleu,
                "rouge_l": rouge_l,
                "exact_match": em,
                "medical_acc": med_acc,
                # More meaningful than trace length: loop steps estimated from "Step N" lines.
                "steps": _agent_steps_from_trace(trace),
                "trace_len": len(trace),
            }
            if save_trace:
                rec["reasoning_trace"] = trace
            results.append(rec)

    # write results.jsonl
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "name": name,
        "num_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "tools_enabled": enabled_tools,
        "metrics": {
            "bleu_mean": mean_or_zero(metrics_acc["bleu"]),
            "rouge_l_mean": mean_or_zero(metrics_acc["rouge_l"]),
            "exact_match_mean": mean_or_zero(metrics_acc["exact_match"]),
            "medical_acc_mean": mean_or_zero(metrics_acc["medical_acc"]),
        },
        "avg_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "avg_trace_len": float(np.mean([r["trace_len"] for r in results])) if results else 0.0,
        "config": config,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results, summary


def _write_metrics_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_comparison(
    out_path: str,
    *,
    summaries: Dict[str, Dict[str, Any]],
    tool_stats: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(summaries.keys())
    metric_keys = ["rouge_l_mean", "bleu_mean", "exact_match_mean", "medical_acc_mean"]

    fig = plt.figure(figsize=(12, 7), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    # (1) Metrics bar plot
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(variants))
    width = 0.18
    for j, mk in enumerate(metric_keys):
        vals = [summaries[v]["metrics"][mk] for v in variants]
        ax0.bar(x + (j - 1.5) * width, vals, width=width, label=mk.replace("_mean", ""))
    ax0.set_xticks(x)
    ax0.set_xticklabels(variants, rotation=0)
    ax0.set_title("Metrics comparison (RAG excluded)")
    ax0.legend(ncol=4, fontsize=8)
    ax0.grid(axis="y", alpha=0.25)

    # (2) Avg steps
    ax1 = fig.add_subplot(gs[1, 0])
    steps = [summaries[v].get("avg_steps", 0.0) for v in variants]
    ax1.bar(variants, steps, color="#6666aa")
    ax1.set_title("Avg agent steps")
    ax1.grid(axis="y", alpha=0.25)

    # (3) Tool stats (calls/success)
    ax2 = fig.add_subplot(gs[1, 1])
    by_tool = (tool_stats or {}).get("by_tool", {}) or {}
    tool_names = list(by_tool.keys())
    calls = [by_tool[t].get("calls", 0) for t in tool_names]
    succ = [by_tool[t].get("success", 0) for t in tool_names]
    xx = np.arange(len(tool_names))
    ax2.bar(xx - 0.18, calls, width=0.36, label="calls")
    ax2.bar(xx + 0.18, succ, width=0.36, label="success")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(tool_names, rotation=30, ha="right")
    ax2.set_title("Tool calls/success (from full-trace variant)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tool evaluation on llava_med_qwen_format_fixed.json (RAG excluded).")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to llava_med_qwen_format_fixed.json")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to evaluate (streamed).")
    parser.add_argument("--start_offset", type=int, default=0, help="Skip first N usable samples.")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.output_dir:
        out_root = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(PROJECT_ROOT, "eval_results", f"tool_eval_no_rag_{ts}")
    os.makedirs(out_root, exist_ok=True)

    samples = load_llava_med_samples(args.dataset_json, limit=args.limit, start_offset=args.start_offset)
    if not samples:
        raise RuntimeError("No samples extracted from dataset (check format/path).")

    model, processor = load_model(args.base_model, args.lora_path)
    agent_module.set_tool_context(model=model, processor=processor, rag=None)

    config: Dict[str, Any] = dict(agent_module.DEFAULT_CONFIG)
    config["max_agent_steps"] = int(args.max_steps)
    config["max_new_tokens"] = int(args.max_new_tokens)

    # Evaluate 3 variants (RAG excluded by construction)
    variants: List[Tuple[str, List[str], bool]] = [
        # Save traces only for the richest variant to reduce disk IO and runtime.
        ("no_tools", [], False),
        ("analyze_only", ["analyze_medical_image"], False),
        ("analyze_plus_safety", ["analyze_medical_image", "safety_check_medical_answer"], True),
    ]

    summaries: Dict[str, Dict[str, Any]] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for name, tools, save_trace in variants:
        out_dir = os.path.join(out_root, name)
        results, summary = evaluate_variant(
            name=name,
            enabled_tools=tools,
            samples=samples,
            model=model,
            processor=processor,
            config=config,
            out_dir=out_dir,
            save_trace=save_trace,  # trace needed for tool_stats
        )
        summaries[name] = summary
        all_results[name] = results

    # Tool-level stats from analyze_plus_safety (the richest tool variant)
    stats_variant = "analyze_plus_safety"
    tool_names_for_stats = variants[-1][1]
    tool_stats = extract_tool_stats_from_traces(all_results[stats_variant], tool_names_for_stats)

    # Write combined report + CSV
    report = {
        "output_dir": out_root,
        "timestamp": datetime.now().isoformat(),
        "dataset_json": args.dataset_json,
        "num_samples": len(samples),
        "variants": {k: v.get("tools_enabled", []) for k, v in summaries.items()},
        "summaries": summaries,
        "tool_level_stats": tool_stats,
    }
    with open(os.path.join(out_root, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    rows = []
    for name, s in summaries.items():
        m = s.get("metrics", {})
        rows.append(
            {
                "variant": name,
                "num_samples": s.get("num_samples", 0),
                "bleu_mean": m.get("bleu_mean", 0.0),
                "rouge_l_mean": m.get("rouge_l_mean", 0.0),
                "exact_match_mean": m.get("exact_match_mean", 0.0),
                "medical_acc_mean": m.get("medical_acc_mean", 0.0),
                "avg_steps": s.get("avg_steps", 0.0),
                "avg_trace_len": s.get("avg_trace_len", 0.0),
            }
        )
    _write_metrics_csv(os.path.join(out_root, "metrics.csv"), rows)

    plot_comparison(os.path.join(out_root, "tool_eval_plot.png"), summaries=summaries, tool_stats=tool_stats)

    print(f"Done. Artifacts written to: {out_root}")


if __name__ == "__main__":
    main()

