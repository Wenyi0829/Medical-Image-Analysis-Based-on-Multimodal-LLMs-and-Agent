# Medical Image Analysis with Multimodal LLMs & Agent

Fine-tuning and inference for medical VQA with tool-augmented agents and optional RAG.

## Layout

- `rag/` — medical text retrieval (FAISS + sentence embeddings)
- `scripts/` — training, inference, evaluation, and SLURM jobs

## Setup

Place the base model under `models/` and datasets under `rag_data/` (not included in this repo). See scripts such as `inference.py` and `build_openfda_label_index.py` for paths and usage.

## Base model & data

- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
