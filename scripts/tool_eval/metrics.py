from __future__ import annotations

from typing import Tuple

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def calculate_bleu(reference: str, candidate: str) -> float:
    try:
        ref_tokens = (reference or "").lower().split()
        cand_tokens = (candidate or "").lower().split()
        if not ref_tokens or not cand_tokens:
            return 0.0
        return float(sentence_bleu([ref_tokens], cand_tokens))
    except Exception:
        return 0.0


def calculate_exact_match(reference: str, candidate: str) -> float:
    return 1.0 if (reference or "").strip().lower() == (candidate or "").strip().lower() else 0.0


def calculate_medical_accuracy(reference: str, candidate: str) -> float:
    # Very rough keyword coverage metric (kept consistent with your existing evaluation style).
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
    ref_lower = (reference or "").lower()
    cand_lower = (candidate or "").lower()
    matches = sum(1 for kw in medical_keywords if (kw in ref_lower) == (kw in cand_lower))
    return float(matches / len(medical_keywords)) if medical_keywords else 0.0


def get_rouge() -> Rouge:
    return Rouge()


def evaluate_text_metrics(rouge: Rouge, reference: str, prediction: str) -> Tuple[float, float, float, float]:
    ref_s = (reference or "").strip()
    pred_s = (prediction or "").strip()
    bleu = calculate_bleu(ref_s, pred_s)
    if not ref_s or not pred_s:
        rouge_l = 0.0
    else:
        try:
            rouge_l = float(rouge.get_scores([ref_s], [pred_s])[0]["rouge-l"]["f"])
        except Exception:
            rouge_l = 0.0
    em = calculate_exact_match(reference, prediction)
    med_acc = calculate_medical_accuracy(reference, prediction)
    return bleu, rouge_l, em, med_acc

