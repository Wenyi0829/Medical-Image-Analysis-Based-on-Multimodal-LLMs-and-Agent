#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretty plotting for tool evaluation results (defense-ready).

Input directory must contain:
  - metrics.csv
  - report.json

Outputs:
  - tool_eval_plot_pretty.png
  - tool_eval_plot_pretty.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

import numpy as np


def _read_metrics_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _short_variant_name(v: str) -> str:
    mapping = {
        "no_tools": "No tools",
        "analyze_only": "Analyze only",
        "analyze_plus_safety": "Analyze + Safety",
    }
    return mapping.get(v, v)


def plot(out_dir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = os.path.join(out_dir, "metrics.csv")
    report_path = os.path.join(out_dir, "report.json")

    rows = _read_metrics_csv(metrics_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Keep a stable variant order for presentations.
    order = ["no_tools", "analyze_only", "analyze_plus_safety"]
    rows_by_variant = {r["variant"]: r for r in rows if "variant" in r}
    variants = [v for v in order if v in rows_by_variant]

    # Extract metrics
    rouge = np.array([_to_float(rows_by_variant[v].get("rouge_l_mean")) for v in variants])
    bleu = np.array([_to_float(rows_by_variant[v].get("bleu_mean")) for v in variants])
    medacc = np.array([_to_float(rows_by_variant[v].get("medical_acc_mean")) for v in variants])
    steps = np.array([_to_float(rows_by_variant[v].get("avg_steps")) for v in variants])
    tracelen = np.array([_to_float(rows_by_variant[v].get("avg_trace_len")) for v in variants])

    tool_stats = report.get("tool_level_stats") or {}
    by_tool = (tool_stats.get("by_tool") or {}) if isinstance(tool_stats, dict) else {}
    safety_calls = _to_float(((by_tool.get("safety_check_medical_answer") or {}).get("calls")) if isinstance(by_tool, dict) else 0)
    n = int(report.get("num_samples") or 0)
    safety_rate = (safety_calls / n) if n else 0.0

    # Style: defense-friendly.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    colors = {
        "No tools": "#4C78A8",
        "Analyze only": "#F58518",
        "Analyze + Safety": "#54A24B",
    }
    labels = [_short_variant_name(v) for v in variants]
    bar_colors = [colors.get(lbl, "#777777") for lbl in labels]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], wspace=0.35, hspace=0.35)

    # --- Panel A: text metrics
    axA = fig.add_subplot(gs[0, :2])
    x = np.arange(len(labels))
    width = 0.23
    axA.bar(x - width, rouge, width=width, label="ROUGE-L", color="#4C78A8")
    axA.bar(x, bleu, width=width, label="BLEU", color="#F58518")
    axA.bar(x + width, medacc, width=width, label="Medical-Acc (keyword)", color="#54A24B")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=0)
    axA.set_ylim(0, max(0.35, float(np.max([rouge.max(initial=0), bleu.max(initial=0), medacc.max(initial=0)]) * 1.25)))
    axA.set_title("Tool-calling evaluation (RAG excluded) — metrics", pad=10, fontweight="bold")
    axA.legend(frameon=True, fontsize=9, loc="upper right")
    axA.grid(axis="y", alpha=0.25)

    # Add value labels
    def _annotate(ax, xs, ys):
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

    _annotate(axA, x - width, rouge)
    _annotate(axA, x, bleu)
    _annotate(axA, x + width, medacc)

    # --- Panel B: steps & trace length (proxy for tool usage complexity)
    axB = fig.add_subplot(gs[1, 0])
    axB.bar(labels, steps, color=bar_colors, alpha=0.9)
    axB.set_title("Avg agent steps", fontweight="bold")
    axB.set_ylabel("steps")
    axB.grid(axis="y", alpha=0.25)
    for lbl, v in zip(labels, steps):
        axB.text(lbl, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axC = fig.add_subplot(gs[1, 1])
    axC.bar(labels, tracelen, color=bar_colors, alpha=0.55)
    axC.set_title("Avg trace length", fontweight="bold")
    axC.set_ylabel("lines")
    axC.grid(axis="y", alpha=0.25)
    for lbl, v in zip(labels, tracelen):
        axC.text(lbl, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # --- Panel D: tool call counts
    axD = fig.add_subplot(gs[0, 2])
    tool_names = ["analyze_medical_image", "safety_check_medical_answer"]
    calls = [int((by_tool.get(t) or {}).get("calls") or 0) for t in tool_names] if isinstance(by_tool, dict) else [0, 0]
    succ = [int((by_tool.get(t) or {}).get("success") or 0) for t in tool_names] if isinstance(by_tool, dict) else [0, 0]
    xx = np.arange(len(tool_names))
    axD.bar(xx - 0.18, calls, width=0.36, label="calls", color="#7F7F7F")
    axD.bar(xx + 0.18, succ, width=0.36, label="success", color="#2CA02C")
    axD.set_xticks(xx)
    axD.set_xticklabels(["analyze", "safety"], rotation=0)
    axD.set_title("Tool execution stats", fontweight="bold")
    axD.legend(fontsize=9, frameon=True)
    axD.grid(axis="y", alpha=0.25)
    for i, (c, s) in enumerate(zip(calls, succ)):
        axD.text(i - 0.18, c + 0.3, str(c), ha="center", va="bottom", fontsize=9)
        axD.text(i + 0.18, s + 0.3, str(s), ha="center", va="bottom", fontsize=9)

    # Footer with run config
    cfg = (((report.get("summaries") or {}).get("analyze_plus_safety") or {}).get("config") or {}) if isinstance(report, dict) else {}
    max_steps = cfg.get("max_agent_steps")
    max_new_tokens = cfg.get("max_new_tokens")
    subtitle = f"N={n} | max_steps={max_steps} | max_new_tokens={max_new_tokens} | safety_call_rate≈{safety_rate:.1%}"
    fig.suptitle(subtitle, y=0.99, fontsize=11)

    png_path = os.path.join(out_dir, "tool_eval_plot_pretty.png")
    pdf_path = os.path.join(out_dir, "tool_eval_plot_pretty.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    ap = argparse.ArgumentParser(description="Pretty plot tool-eval directory.")
    ap.add_argument("--out_dir", type=str, required=True, help="tool_eval_no_rag_*/ directory containing metrics.csv/report.json")
    args = ap.parse_args()
    p = plot(args.out_dir)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()

