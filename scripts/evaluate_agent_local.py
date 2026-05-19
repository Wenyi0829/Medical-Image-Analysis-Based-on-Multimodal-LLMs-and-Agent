#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local (local transformers) evaluation for the Medical VQA agent.

This script evaluates your functional-calling agent loop without using RAG as a tool.
It reuses the agent implementation in `scripts/agent_quick_test.py` to keep the tool
protocol consistent.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agent_quick_test as agent_module  # noqa: E402


def load_model(model_path: str, lora_path: str | None = None):
    """
    Load base model with optional LoRA adapter.
    """
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path:
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    model.eval()
    return model, processor


def calculate_bleu(reference: str, candidate: str) -> float:
    try:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if not ref_tokens or not cand_tokens:
            return 0.0
        return sentence_bleu([ref_tokens], cand_tokens)
    except Exception:
        return 0.0


def calculate_exact_match(reference: str, candidate: str) -> float:
    return 1.0 if reference.strip().lower() == candidate.strip().lower() else 0.0


def calculate_medical_accuracy(reference: str, candidate: str) -> float:
    medical_keywords = [
        "ct",
        "mri",
        "ultrasound",
        "x-ray",
        "pet",
        "tumor",
        "lesion",
        "mass",
        "nodule",
        "normal",
        "abnormal",
        "positive",
        "negative",
        "acute",
        "chronic",
        "benign",
        "malignant",
    ]
    ref_lower = reference.lower()
    cand_lower = candidate.lower()
    matches = sum(1 for kw in medical_keywords if (kw in ref_lower) == (kw in cand_lower))
    return matches / len(medical_keywords)


def parse_val_dataset(val_path: str, sample_size: int | None = None):
    """
    Use the dataset parsing logic already implemented for the agent.
    """
    return agent_module.parse_val_dataset(val_path, sample_size=sample_size)


def evaluate_agent(
    samples,
    model,
    processor,
    output_dir: str,
    config: dict,
):
    results = []
    metrics = {"bleu": [], "rouge_l": [], "exact_match": [], "medical_acc": []}
    rouger = Rouge()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "agent_eval_results.jsonl")

    with open(save_path, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(tqdm(samples, desc="Evaluating", file=sys.stdout)):
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

            bleu = calculate_bleu(reference, pred)
            rouge_l = rouger.get_scores([reference], [pred])[0]["rouge-l"]["f"]
            em = calculate_exact_match(reference, pred)
            med_acc = calculate_medical_accuracy(reference, pred)

            metrics["bleu"].append(bleu)
            metrics["rouge_l"].append(rouge_l)
            metrics["exact_match"].append(em)
            metrics["medical_acc"].append(med_acc)

            result = {
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
                "reasoning_trace": trace,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            results.append(result)

    summary = {
        "num_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "bleu_mean": float(np.mean(metrics["bleu"])) if metrics["bleu"] else 0.0,
            "rouge_l_mean": float(np.mean(metrics["rouge_l"])) if metrics["rouge_l"] else 0.0,
            "exact_match_mean": float(np.mean(metrics["exact_match"])) if metrics["exact_match"] else 0.0,
            "medical_acc_mean": float(np.mean(metrics["medical_acc"])) if metrics["medical_acc"] else 0.0,
        },
        "config": config,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return results, summary


def save_compact_table(results, output_dir: str, name: str):
    if not results:
        return
    df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local agent evaluation (functional calling).")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--base_model", type=str, required=True, help="Base model path (Qwen3-VL-8B).")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path; omit for baseline.")
    parser.add_argument("--val_dataset", type=str, required=True, help="Path to val_dataset.jsonl.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--eval_baseline", action="store_true", default=False, help="Also evaluate base model without LoRA.")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/wshenah/project/eval_results/agent_local_{TIMESTAMP}"

    samples = parse_val_dataset(args.val_dataset, sample_size=args.sample_size)
    if not samples:
        raise RuntimeError(f"No samples loaded from: {args.val_dataset}")

    config = dict(agent_module.DEFAULT_CONFIG)
    config["max_agent_steps"] = args.max_steps

    # Evaluate finetuned (if provided) or base (if lora_path omitted)
    finetuned_subdir = os.path.join(output_dir, "finetuned")
    model, processor = load_model(args.base_model, args.lora_path)
    finetuned_results, finetuned_summary = evaluate_agent(
        samples=samples,
        model=model,
        processor=processor,
        output_dir=finetuned_subdir,
        config=config,
    )
    save_compact_table(finetuned_results, finetuned_subdir, "agent_eval_results")

    # Optional baseline evaluation
    baseline_summary = None
    if args.eval_baseline and args.lora_path:
        baseline_subdir = os.path.join(output_dir, "baseline")
        base_model, base_processor = load_model(args.base_model, None)
        baseline_results, baseline_summary = evaluate_agent(
            samples=samples,
            model=base_model,
            processor=base_processor,
            output_dir=baseline_subdir,
            config=config,
        )
        save_compact_table(baseline_results, baseline_subdir, "agent_eval_results")

    # Write a top-level combined summary
    combined = {
        "timestamp": datetime.now().isoformat(),
        "samples": len(samples),
        "config": config,
        "finetuned": finetuned_summary,
        "baseline": baseline_summary,
    }
    combined_path = os.path.join(output_dir, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Done. Results written to: {output_dir}")

