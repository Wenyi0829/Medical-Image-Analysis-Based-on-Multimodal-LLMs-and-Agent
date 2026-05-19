from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .agent_patch import patch_agent_tools
from .metrics import evaluate_text_metrics, get_rouge


@dataclass
class VariantResult:
    name: str
    enabled_tool_names: List[str]
    summary: Dict[str, Any]
    results: List[Dict[str, Any]]


def evaluate_variant(
    *,
    variant_name: str,
    enabled_tool_names: List[str],
    samples: List[Dict[str, Any]],
    model: Any,
    processor: Any,
    config: Dict[str, Any],
    output_dir: str,
    agent_module: Any,
    save_detail: bool,
) -> VariantResult:
    os.makedirs(output_dir, exist_ok=True)
    rouge = get_rouge()

    metrics_acc: Dict[str, List[float]] = {
        "bleu": [],
        "rouge_l": [],
        "exact_match": [],
        "medical_acc": [],
    }

    save_path = os.path.join(output_dir, "agent_eval_results.jsonl")
    results: List[Dict[str, Any]] = []

    with open(save_path, "w", encoding="utf-8") as f_out:
        with patch_agent_tools(agent_module, enabled_tool_names=enabled_tool_names):
            iterator = tqdm(samples, desc=f"Evaluating {variant_name}")
            for idx, sample in enumerate(iterator):
                question = sample["question"]
                image = sample["image"]
                reference = sample["answer"]

                pred, trace = agent_module.run_agent_inference(
                    model=model,
                    processor=processor,
                    question=question,
                    image_path=image,
                    config=config,
                )

                bleu, rouge_l, em, med_acc = evaluate_text_metrics(rouge, reference, pred)

                metrics_acc["bleu"].append(bleu)
                metrics_acc["rouge_l"].append(rouge_l)
                metrics_acc["exact_match"].append(em)
                metrics_acc["medical_acc"].append(med_acc)

                record: Dict[str, Any] = {
                    "idx": idx,
                    "question": question,
                    "image": os.path.basename(image) if image else "",
                    "reference": reference,
                    "prediction": pred,
                    "bleu": bleu,
                    "rouge_l": rouge_l,
                    "exact_match": em,
                    "medical_acc": med_acc,
                    "steps": len(trace),
                }
                if save_detail:
                    record["reasoning_trace"] = trace

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                results.append(record)

    def mean_or_zero(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "num_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "tools_enabled": enabled_tool_names,
        "metrics": {
            "bleu_mean": mean_or_zero(metrics_acc["bleu"]),
            "rouge_l_mean": mean_or_zero(metrics_acc["rouge_l"]),
            "exact_match_mean": mean_or_zero(metrics_acc["exact_match"]),
            "medical_acc_mean": mean_or_zero(metrics_acc["medical_acc"]),
        },
        "config": config,
    }
    return VariantResult(name=variant_name, enabled_tool_names=enabled_tool_names, summary=summary, results=results)

