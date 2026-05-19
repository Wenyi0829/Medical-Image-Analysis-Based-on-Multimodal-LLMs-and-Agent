#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert OpenFDA bulk drug/label JSON zips into a single JSONL corpus for RAG.

Output JSONL schema (one record per label record):
  {
    "text": "<concatenated label sections>",
    "metadata": {
      "source": "openfda_drug_label",
      "openfda_id": "...",
      "set_id": "...",
      "spl_id": "...",
      "effective_time": "YYYYMMDD",
      "version": "...",
      "brand_name": [...],
      "generic_name": [...],
      "manufacturer_name": [...],
      "product_ndc": [...],
      "route": [...],
      "product_type": [...],
      "substance_name": [...],
      "pharm_class_epc": [...],
      "pharm_class_cs": [...],
      "unii": [...],
      "nui": [...],
    }
  }
"""

import argparse
import json
import os
import re
import sys
import zipfile
from typing import Any, Dict, Iterable, List, Optional


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return _norm(v)
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, str):
                t = _norm(x)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    return ""


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        t = _norm(v)
        return [t] if t else []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            if isinstance(x, str):
                t = _norm(x)
                if t:
                    out.append(t)
        return out
    return []


def _pick_openfda_fields(openfda: Any) -> Dict[str, Any]:
    if not isinstance(openfda, dict):
        return {}
    keys = [
        "brand_name",
        "generic_name",
        "manufacturer_name",
        "product_ndc",
        "route",
        "product_type",
        "substance_name",
        "pharm_class_epc",
        "pharm_class_cs",
        "unii",
        "nui",
        "package_ndc",
        "is_original_packager",
        "application_number",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        v = openfda.get(k)
        if isinstance(v, list):
            out[k] = v
        elif isinstance(v, str) and v.strip():
            out[k] = [v.strip()]
        elif isinstance(v, bool):
            out[k] = [v]
        elif v is not None and isinstance(v, (int, float)):
            out[k] = [v]
    return out


def build_text(record: Dict[str, Any]) -> str:
    # Keep a conservative, high-signal set of sections for medication safety + use.
    # Many of these are lists of strings in OpenFDA.
    sections = [
        ("INDICATIONS", "indications_and_usage"),
        ("DOSAGE", "dosage_and_administration"),
        ("CONTRAINDICATIONS", "contraindications"),
        ("WARNINGS", "boxed_warning"),
        ("WARNINGS", "warnings"),
        ("PRECAUTIONS", "precautions"),
        ("GENERAL PRECAUTIONS", "general_precautions"),
        ("ADVERSE REACTIONS", "adverse_reactions"),
        ("DRUG INTERACTIONS", "drug_interactions"),
        ("DESCRIPTION", "description"),
        ("HOW SUPPLIED", "how_supplied"),
        ("STORAGE AND HANDLING", "storage_and_handling"),
    ]

    parts: List[str] = []

    # Add a short header from OpenFDA harmonized fields to help retrieval.
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    brand = ", ".join(_as_list(openfda.get("brand_name")))[:200]
    generic = ", ".join(_as_list(openfda.get("generic_name")))[:200]
    mfr = ", ".join(_as_list(openfda.get("manufacturer_name")))[:200]
    header = " | ".join([x for x in [brand, generic, mfr] if x])
    if header:
        parts.append(_norm(header))

    for label, field in sections:
        txt = _as_text(record.get(field))
        if not txt:
            continue
        parts.append(f"[{label}] {txt}")

    return "\n".join(parts).strip()


def iter_zip_records(zip_path: str) -> Iterable[Dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if not names:
            return
        with z.open(names[0]) as f:
            data = json.load(f)
    results = data.get("results", [])
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                yield r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        default="/home/wshenah/project/rag_data/openfda/drug_label_zips",
        help="Directory containing drug-label-*.json.zip files.",
    )
    ap.add_argument(
        "--output_jsonl",
        type=str,
        default="/home/wshenah/project/rag_data/openfda/openfda_drug_label_corpus.jsonl",
        help="Output JSONL path.",
    )
    ap.add_argument("--min_chars", type=int, default=200, help="Minimum text chars per record.")
    args = ap.parse_args()

    in_dir = args.input_dir
    out_path = args.output_jsonl
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    zips = sorted(
        [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.startswith("drug-label-") and f.endswith(".json.zip")
        ]
    )
    if not zips:
        print(f"ERROR: no drug-label-*.json.zip found in: {in_dir}", file=sys.stderr)
        return 2

    total_in = 0
    total_out = 0
    total_skipped = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for zpath in zips:
            print(f"[openfda] reading {os.path.basename(zpath)}", file=sys.stderr)
            for rec in iter_zip_records(zpath):
                total_in += 1
                text = build_text(rec)
                if len(text) < args.min_chars:
                    total_skipped += 1
                    continue

                md: Dict[str, Any] = {
                    "source": "openfda_drug_label",
                    "openfda_id": rec.get("id"),
                    "set_id": rec.get("set_id"),
                    "spl_id": rec.get("spl_id"),
                    "effective_time": rec.get("effective_time"),
                    "version": rec.get("version"),
                }
                md.update(_pick_openfda_fields(rec.get("openfda")))

                obj = {"text": text, "metadata": md}
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_out += 1

    print(
        json.dumps(
            {
                "input_zips": len(zips),
                "records_in": total_in,
                "records_out": total_out,
                "records_skipped": total_skipped,
                "output_jsonl": out_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

