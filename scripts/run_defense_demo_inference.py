#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-batch inference for graduation defense demos (RAG + tool traces).

Reads cases in the same JSONL shape as val_dataset.jsonl (see agent_quick_test.parse_val_dataset).
Writes one JSON file with per-case predictions and reasoning_trace.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import agent_quick_test as aq  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Defense demo: few-shot agent inference with traces.")
    parser.add_argument("--cases_jsonl", type=str, required=True, help="JSONL path (val_dataset-style).")
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
    parser.add_argument("--no_rag", action="store_true", help="Disable FAISS RAG (drug label tool will error).")
    args = parser.parse_args()

    cfg = aq.DEFAULT_CONFIG.copy()
    cfg["base_model_path"] = args.base_model
    cfg["lora_ckpt_dir"] = args.lora_path
    cfg["max_agent_steps"] = args.max_steps
    cfg["max_new_tokens"] = args.max_new_tokens

    samples = aq.parse_val_dataset(args.cases_jsonl, sample_size=None)
    if not samples:
        print("No samples loaded; check --cases_jsonl", file=sys.stderr)
        sys.exit(1)

    print(f"[demo] Loading model from {cfg['base_model_path']} ...")
    model, processor = aq.load_finetuned_model(cfg["base_model_path"], cfg["lora_ckpt_dir"])

    rag = None
    if not args.no_rag and aq._RAG_AVAILABLE:
        from rag.medical_rag import MedicalRAG  # noqa: E402

        # Match the on-disk index: chunk params AND corpus_sources fingerprint (manifest may add text_fields).
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
                    # medical_rag_config.json may omit jsonl text_fields; fingerprint requires them for OpenFDA.
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
                    print(f"[demo] Using corpus_sources + chunk params from {cfg_path} (avoids fingerprint mismatch).")
                elif rag_kwargs:
                    print(f"[demo] Using RAG chunk/embed params from {cfg_path}: {rag_kwargs}")
            except Exception as exc:
                print(f"[demo] Could not read prior RAG config ({exc}); using manifest defaults.", file=sys.stderr)

        print(f"[demo] Loading RAG index from {args.rag_index_dir} ...")
        rag = MedicalRAG(
            index_dir=args.rag_index_dir,
            corpus_manifest_path=corpus_manifest_path,
            corpus_sources=corpus_sources_arg,
            force_rebuild=False,
            **rag_kwargs,
        )
    elif args.no_rag:
        print("[demo] RAG disabled (--no_rag).")
    else:
        print("[demo] RAG unavailable (missing faiss / sentence-transformers).", file=sys.stderr)

    aq.set_tool_context(model=model, processor=processor, rag=rag)

    out: dict = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "max_agent_steps": cfg["max_agent_steps"],
            "max_new_tokens": cfg["max_new_tokens"],
            "base_model_path": cfg["base_model_path"],
            "lora_ckpt_dir": cfg["lora_ckpt_dir"],
            "rag_index_dir": None if args.no_rag else args.rag_index_dir,
            "rag_manifest_path": None if args.no_rag else args.rag_manifest_path,
        },
        "cases": [],
    }

    for i, s in enumerate(samples):
        print(f"[demo] Case {i + 1}/{len(samples)} ...")
        pred, trace = aq.run_agent_inference(
            model,
            processor,
            s["question"],
            s.get("image"),
            cfg,
        )
        ref = s.get("answer") or ""
        out["cases"].append(
            {
                "case_id": i + 1,
                "input": {
                    "question": s["question"],
                    "image": s.get("image"),
                },
                "dataset_reference_excerpt": ref[:800] + ("..." if len(ref) > 800 else ""),
                "model_output": pred,
                "reasoning_trace": trace,
            }
        )

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[demo] Wrote {args.output_json}")


if __name__ == "__main__":
    main()
