#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drug-focused RAG evaluation (RAG ON vs RAG OFF) with real model inference.

Unlike dataset-based evaluation, we don't assume gold references.
We compare:
- tool usage stats (esp. search_drug_label call rate)
- evidence inclusion rate in final answers
- retrieval hit rate from tool responses

Outputs:
  - rag_drug_report.json
  - per-query jsonl (rag_on/results.jsonl, rag_off/results.jsonl)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agent_quick_test as agent_module  # noqa: E402
from tool_eval.agent_patch import patch_agent_tools  # noqa: E402
from tool_eval.metrics import evaluate_text_metrics, get_rouge  # noqa: E402
from tool_eval.tool_stats import extract_tool_stats_from_traces  # noqa: E402


DEFAULT_DRUG_QUERIES = [
    "warfarin interactions with antibiotics",
    "warfarin contraindications and bleeding risk",
    "metformin contraindications renal impairment eGFR",
    "metformin lactic acidosis warning",
    "isotretinoin pregnancy boxed warning iPLEDGE",
    "isotretinoin contraindications breastfeeding",
    "amiodarone adverse reactions pulmonary toxicity",
    "amiodarone drug interactions with warfarin or digoxin",
    "ibuprofen dosage adults max daily dose",
    "ibuprofen contraindications GI bleeding",
    "clozapine black box warning agranulocytosis myocarditis",
    "valproate pregnancy warning neural tube defects",
    "lisinopril pregnancy contraindication",
    "rivaroxaban contraindications renal dosing",
    "digoxin toxicity signs and interactions",
    "sertraline drug interactions MAOI serotonin syndrome",
    "atorvastatin contraindications liver disease",
    "allopurinol severe skin reactions warning",
    "prednisone adverse effects long term",
    "azithromycin QT prolongation warning",
]


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


def init_rag(index_dir: str, manifest_path: str):
    if not getattr(agent_module, "_RAG_AVAILABLE", False):
        raise RuntimeError("RAG deps unavailable. Install faiss-cpu and sentence-transformers.")

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

    return agent_module.MedicalRAG(
        index_dir=index_dir,
        corpus_manifest_path=manifest_path,
        force_rebuild=False,
        **rag_kwargs,
    )


def load_queries(path: str) -> List[str]:
    if not path:
        return list(DEFAULT_DRUG_QUERIES)
    qs: List[str] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = str(obj.get("query", "")).strip()
                if q:
                    qs.append(q)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    qs.append(q)
    return qs


def extract_retrieval_hit_from_trace(trace: List[str]) -> Tuple[bool, int]:
    """
    Determine whether search_drug_label returned non-empty results.
    We parse the tool result JSON substring when present.
    """
    last_tool = None
    for line in trace:
        if "Tool Call:" in line or "Deterministic Drug Tool Call:" in line:
            if "search_drug_label" in line:
                last_tool = "search_drug_label"
        if last_tool == "search_drug_label" and ("Tool Result:" in line or "Deterministic Drug Tool Result:" in line):
            # try parse first {...}
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                payload = json.loads(m.group(0))
            except Exception:
                continue
            results = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(results, list):
                return (len(results) > 0), len(results)
    return False, 0


def evidence_present(answer: str) -> bool:
    a = (answer or "").lower()
    return ("evidence:" in a) and ("source=" in a)


def extract_pseudo_reference_from_trace(trace: List[str], *, max_chars: int = 3500) -> str:
    """
    Build a pseudo-reference string from search_drug_label tool JSON in the trace.

    Used for BLEU/ROUGE vs model answer (overlap with retrieved passages; not a clinical gold label).
    """
    last_tool = None
    for line in trace:
        if "search_drug_label" in line and ("Tool Call:" in line or "Deterministic Drug Tool Call:" in line):
            last_tool = "search_drug_label"
        if last_tool == "search_drug_label" and ("Tool Result:" in line or "Deterministic Drug Tool Result:" in line):
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            try:
                payload = json.loads(m.group(0))
            except Exception:
                continue
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                continue
            parts: List[str] = []
            for item in results[:3]:
                if not isinstance(item, dict):
                    continue
                p = item.get("passage") or item.get("text") or ""
                p = str(p).strip()
                if p:
                    parts.append(p)
            ref = " ".join(parts).strip()
            if len(ref) > max_chars:
                ref = ref[:max_chars] + "..."
            return ref
    return ""


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def attach_overlap_metrics(
    results: List[Dict[str, Any]],
    pseudo_refs: List[str],
    rouge: Any,
) -> Tuple[float, float]:
    """Mutates each result with bleu_vs_retrieval / rouge_l_vs_retrieval. Returns (bleu_mean, rouge_l_mean)."""
    bleus: List[float] = []
    rouges: List[float] = []
    for r, ref in zip(results, pseudo_refs):
        pred = r.get("prediction") or ""
        ref = ref or ""
        b, rl, _, _ = evaluate_text_metrics(rouge, ref, pred)
        r["bleu_vs_retrieval"] = float(b)
        r["rouge_l_vs_retrieval"] = float(rl)
        bleus.append(float(b))
        rouges.append(float(rl))
    return mean(bleus), mean(rouges)


def run_variant(
    *,
    name: str,
    enabled_tools: List[str],
    queries: List[str],
    model: Any,
    processor: Any,
    config: Dict[str, Any],
    out_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    with patch_agent_tools(agent_module, enabled_tool_names=enabled_tools):
        for i, q in enumerate(queries):
            pred, trace = agent_module.run_agent_inference(
                model=model,
                processor=processor,
                question=q,
                image_path=None,
                config=config,
            )
            hit, n = extract_retrieval_hit_from_trace(trace)
            rec = {
                "idx": i,
                "query": q,
                "prediction": pred,
                "evidence_present": evidence_present(pred),
                "retrieval_hit": hit,
                "retrieval_results_count": n,
                "reasoning_trace": trace,
            }
            results.append(rec)

    tool_stats = extract_tool_stats_from_traces(results, enabled_tools)
    ev_rate = sum(1 for r in results if r["evidence_present"]) / len(results) if results else 0.0
    hit_rate = sum(1 for r in results if r["retrieval_hit"]) / len(results) if results else 0.0

    summary: Dict[str, Any] = {
        "name": name,
        "num_queries": len(results),
        "tools_enabled": enabled_tools,
        "tool_stats": tool_stats,
        "evidence_rate": ev_rate,
        "retrieval_hit_rate": hit_rate,
    }

    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results, summary


def rewrite_variant_artifacts(out_dir: str, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Drug-focused RAG evaluation (no gold references).")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--queries_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of queries to run.")
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
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    if not queries:
        raise RuntimeError("No queries loaded.")
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    if args.output_dir:
        out_root = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(PROJECT_ROOT, "eval_results", f"rag_drug_{ts}")
    os.makedirs(out_root, exist_ok=True)

    model, processor = load_model(args.base_model, args.lora_path)
    rag = init_rag(args.rag_index_dir, args.rag_manifest_path)
    agent_module.set_tool_context(model=model, processor=processor, rag=rag)

    config: Dict[str, Any] = dict(agent_module.DEFAULT_CONFIG)
    config["max_agent_steps"] = args.max_steps
    config["max_new_tokens"] = args.max_new_tokens

    all_tools = sorted(list(agent_module.TOOL_REGISTRY.keys()))
    rag_tool = "search_drug_label"
    if rag_tool not in all_tools:
        raise RuntimeError(f"Expected tool `{rag_tool}`. Current tools: {all_tools}")

    rag_on_tools = list(all_tools)
    rag_off_tools = [t for t in all_tools if t != rag_tool]

    on_dir = os.path.join(out_root, "rag_on")
    off_dir = os.path.join(out_root, "rag_off")

    on_results, on_summary = run_variant(
        name="rag_on",
        enabled_tools=rag_on_tools,
        queries=queries,
        model=model,
        processor=processor,
        config=config,
        out_dir=on_dir,
    )
    pseudo_refs = [extract_pseudo_reference_from_trace(r.get("reasoning_trace") or []) for r in on_results]

    rouge = get_rouge()
    on_bleu_mean, on_rl_mean = attach_overlap_metrics(on_results, pseudo_refs, rouge)
    on_summary["bleu_mean_vs_retrieval"] = on_bleu_mean
    on_summary["rouge_l_mean_vs_retrieval"] = on_rl_mean
    rewrite_variant_artifacts(on_dir, on_results, on_summary)

    off_results, off_summary = run_variant(
        name="rag_off",
        enabled_tools=rag_off_tools,
        queries=queries,
        model=model,
        processor=processor,
        config=config,
        out_dir=off_dir,
    )
    off_bleu_mean, off_rl_mean = attach_overlap_metrics(off_results, pseudo_refs, rouge)
    off_summary["bleu_mean_vs_retrieval"] = off_bleu_mean
    off_summary["rouge_l_mean_vs_retrieval"] = off_rl_mean
    rewrite_variant_artifacts(off_dir, off_results, off_summary)

    report = {
        "output_dir": out_root,
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(queries),
        "rag_tool": rag_tool,
        "pseudo_reference_note": (
            "BLEU and ROUGE-L are computed with reference = concatenated top-3 passages from "
            "RAG ON `search_drug_label` tool JSON (per query). RAG OFF uses the same reference "
            "for a paired comparison (overlap with retrieved label text; not a clinical gold label)."
        ),
        "rag_on": on_summary,
        "rag_off": off_summary,
        "deltas": {
            "evidence_rate": on_summary["evidence_rate"] - off_summary["evidence_rate"],
            "retrieval_hit_rate": on_summary["retrieval_hit_rate"] - off_summary["retrieval_hit_rate"],
            "tool_call_rate": on_summary["tool_stats"]["tool_call_rate"] - off_summary["tool_stats"]["tool_call_rate"],
            "avg_tool_calls_per_sample": on_summary["tool_stats"]["avg_tool_calls_per_sample"]
            - off_summary["tool_stats"]["avg_tool_calls_per_sample"],
            "bleu_mean_vs_retrieval": on_bleu_mean - off_bleu_mean,
            "rouge_l_mean_vs_retrieval": on_rl_mean - off_rl_mean,
        },
    }

    report_path = os.path.join(out_root, "rag_drug_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. Report saved to: {report_path}")


if __name__ == "__main__":
    main()

