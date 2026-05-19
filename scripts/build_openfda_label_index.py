#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS index for the OpenFDA drug label JSONL corpus.

This is intended to be run on a Slurm compute node (GPU preferred).
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="/home/wshenah/project/rag/corpus_manifest_openfda_label.json",
        help="Corpus manifest JSON path.",
    )
    ap.add_argument(
        "--index_dir",
        type=str,
        default="/home/wshenah/project/rag_index/openfda_label",
        help="Directory to store/load the FAISS index.",
    )
    ap.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true", default=False)
    ap.add_argument("--chunk_size_words", type=int, default=800)
    ap.add_argument("--chunk_overlap_words", type=int, default=120)
    ap.add_argument("--min_chunk_words", type=int, default=30)
    ap.add_argument("--min_text_chars", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, "/home/wshenah/project")
    from rag.medical_rag import MedicalRAG  # noqa: E402

    os.makedirs(args.index_dir, exist_ok=True)

    rag = MedicalRAG(
        index_dir=args.index_dir,
        corpus_manifest_path=args.manifest,
        embed_model_name=args.embed_model,
        top_k=args.top_k,
        force_rebuild=args.rebuild,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        min_chunk_words=args.min_chunk_words,
        min_text_chars=args.min_text_chars,
    )
    print(
        {
            "index_dir": args.index_dir,
            "chunks": len(rag._docs),
            "ntotal": int(getattr(rag._index, "ntotal", 0)),
            "embed_model": args.embed_model,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

