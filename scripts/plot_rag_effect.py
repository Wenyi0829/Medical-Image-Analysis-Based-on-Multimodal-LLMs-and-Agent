#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot RAG on/off comparison from rag_effect_report.json.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_report(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_report(report_path: str, out_path: str):
    report = load_report(report_path)

    rag_on = report.get("summary_rag_on", {}).get("metrics", {})
    rag_off = report.get("summary_rag_off", {}).get("metrics", {})
    tool_on = report.get("tool_stats_rag_on", {})
    tool_off = report.get("tool_stats_rag_off", {})

    metric_keys = ["bleu_mean", "rouge_l_mean", "exact_match_mean", "medical_acc_mean"]
    labels = ["BLEU", "ROUGE-L", "ExactMatch", "MedAcc"]
    on_vals = [rag_on.get(k, 0.0) for k in metric_keys]
    off_vals = [rag_off.get(k, 0.0) for k in metric_keys]
    deltas = [a - b for a, b in zip(on_vals, off_vals)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: absolute metrics
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, on_vals, width, label="RAG ON")
    axes[0].bar(x + width / 2, off_vals, width, label="RAG OFF")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0, max(on_vals + off_vals + [0.01]) * 1.25)
    axes[0].set_title("Absolute Metrics")
    axes[0].legend()

    # Panel 2: deltas
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    axes[1].bar(labels, deltas, color=colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Delta (RAG ON - RAG OFF)")
    axes[1].tick_params(axis="x", rotation=20)

    # Panel 3: tool usage comparison
    names = ["tool_call_rate", "avg_tool_calls_per_sample"]
    on_tool_vals = [tool_on.get("tool_call_rate", 0.0), tool_on.get("avg_tool_calls_per_sample", 0.0)]
    off_tool_vals = [tool_off.get("tool_call_rate", 0.0), tool_off.get("avg_tool_calls_per_sample", 0.0)]
    x2 = np.arange(len(names))
    axes[2].bar(x2 - width / 2, on_tool_vals, width, label="RAG ON")
    axes[2].bar(x2 + width / 2, off_tool_vals, width, label="RAG OFF")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(names, rotation=20)
    axes[2].set_title("Tool Usage")
    axes[2].legend()

    main_metric = report.get("main_metric", "rouge_l_mean")
    main_delta = report.get("main_delta_rag_on_minus_rag_off", 0.0)
    fig.suptitle(f"RAG Effect Report ({main_metric} delta={main_delta:.4f})", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to rag_effect_report.json")
    parser.add_argument("--out", default="", help="Output png path")
    args = parser.parse_args()

    out = args.out
    if not out:
        out = os.path.join(os.path.dirname(args.report), "rag_effect_plot.png")
    plot_report(args.report, out)


if __name__ == "__main__":
    main()

