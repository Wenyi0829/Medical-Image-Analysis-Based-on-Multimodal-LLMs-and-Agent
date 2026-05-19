#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot drug-focused RAG report from rag_drug_report.json.
Uses non-interactive Agg backend (safe on headless clusters).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# DejaVu Sans ships with matplotlib — no extra fonts needed on compute nodes
_STYLE = {
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0f172a",
    "text.color": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#e2e8f0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
}


def _bar_labels(ax, rects, fmt="{:.3f}", fontsize=8, pad=3):
    """Annotate bar heights (skip if bar too short vs axis range)."""
    lim = ax.get_ylim()[1] - ax.get_ylim()[0]
    if lim <= 0:
        return
    for rect in rects:
        h = rect.get_height()
        if h <= 0:
            continue
        # Skip tiny bars to avoid clutter
        if h / lim < 0.02:
            continue
        ax.annotate(
            fmt.format(h),
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#475569",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to rag_drug_report.json")
    parser.add_argument("--out", default="", help="Output png path")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution (default 220)")
    args = parser.parse_args()

    plt.rcParams.update(_STYLE)

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    on = report.get("rag_on", {})
    off = report.get("rag_off", {})
    deltas = report.get("deltas", {})

    metrics = [
        ("tool_call_rate", on.get("tool_stats", {}).get("tool_call_rate", 0.0), off.get("tool_stats", {}).get("tool_call_rate", 0.0)),
        ("avg_tool_calls/sample", on.get("tool_stats", {}).get("avg_tool_calls_per_sample", 0.0), off.get("tool_stats", {}).get("avg_tool_calls_per_sample", 0.0)),
        ("evidence_rate", on.get("evidence_rate", 0.0), off.get("evidence_rate", 0.0)),
        ("retrieval_hit_rate", on.get("retrieval_hit_rate", 0.0), off.get("retrieval_hit_rate", 0.0)),
        ("BLEU vs retrieval", on.get("bleu_mean_vs_retrieval", 0.0), off.get("bleu_mean_vs_retrieval", 0.0)),
        ("ROUGE-L vs retrieval", on.get("rouge_l_mean_vs_retrieval", 0.0), off.get("rouge_l_mean_vs_retrieval", 0.0)),
    ]

    labels = [m[0] for m in metrics]
    on_vals = [m[1] for m in metrics]
    off_vals = [m[2] for m in metrics]
    diffs = [a - b for a, b in zip(on_vals, off_vals)]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 5.8),
        constrained_layout=False,
    )

    x = np.arange(len(labels))
    width = 0.36
    color_on = "#2563eb"
    color_off = "#94a3b8"
    edge = "#ffffff"

    r_on = axes[0].bar(
        x - width / 2,
        on_vals,
        width,
        label="RAG ON",
        color=color_on,
        edgecolor=edge,
        linewidth=0.8,
        zorder=3,
    )
    r_off = axes[0].bar(
        x + width / 2,
        off_vals,
        width,
        label="RAG OFF",
        color=color_off,
        edgecolor=edge,
        linewidth=0.8,
        zorder=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=22, ha="right")
    ymax = max(on_vals + off_vals + [0.01]) * 1.28
    axes[0].set_ylim(0, ymax)
    axes[0].set_title("Drug-focused metrics", fontweight="600", pad=12)
    axes[0].set_ylabel("Score (rate or similarity)")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="#e2e8f0", loc="upper right")
    axes[0].yaxis.grid(True, zorder=0)
    _bar_labels(axes[0], list(r_on) + list(r_off))

    pos_color = "#059669"
    neg_color = "#dc2626"
    delta_colors = [pos_color if d >= 0 else neg_color for d in diffs]
    r_delta = axes[1].bar(
        x,
        diffs,
        width=0.62,
        color=delta_colors,
        edgecolor=edge,
        linewidth=0.8,
        zorder=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=22, ha="right")
    axes[1].axhline(0, color="#64748b", linewidth=1.0, zorder=2)
    axes[1].set_title("Δ (RAG ON − RAG OFF)", fontweight="600", pad=12)
    axes[1].set_ylabel("Difference")
    axes[1].yaxis.grid(True, zorder=0)
    dmin, dmax = min(diffs + [0]), max(diffs + [0])
    pad_y = max(abs(dmax - dmin) * 0.12, 0.02)
    axes[1].set_ylim(dmin - pad_y, dmax + pad_y)
    _bar_labels(
        axes[1],
        list(r_delta),
        fmt="{:+.4f}" if max(abs(d) for d in diffs) < 0.5 else "{:+.3f}",
    )

    subtitle = (
        f"n = {report.get('num_queries')}  ·  "
        f"Δ evidence = {deltas.get('evidence_rate', 0.0):.3f}  ·  "
        f"Δ hit = {deltas.get('retrieval_hit_rate', 0.0):.3f}  ·  "
        f"Δ BLEU = {deltas.get('bleu_mean_vs_retrieval', 0.0):.4f}  ·  "
        f"Δ ROUGE-L = {deltas.get('rouge_l_mean_vs_retrieval', 0.0):.4f}"
    )
    fig.suptitle("RAG drug-focused evaluation", fontsize=14, fontweight="600", color="#0f172a", y=1.02)
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color="#64748b")

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out = args.out
    if not out:
        out = os.path.join(os.path.dirname(args.report), "rag_drug_plot.png")
    destdir = os.path.dirname(out)
    if destdir:
        os.makedirs(destdir, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved plot to: {out}")


if __name__ == "__main__":
    main()
