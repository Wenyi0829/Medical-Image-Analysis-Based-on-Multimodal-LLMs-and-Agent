#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical VQA Evaluation Script - Qwen3-VL-8B-Thinking
Supports subsampling for quick validation
"""

import os
import json
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import warnings
import numpy as np
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

# ============ Configuration ============
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/wshenah/project/hf_cache'

BASE_MODEL_PATH = '/home/wshenah/project/models/Qwen3-VL-8B-Thinking/'
LORA_CKPT_DIR = '/home/wshenah/project/lora/v14-20260306-195347/checkpoint-1200'
VAL_DATASET_PATH = '/home/wshenah/project/lora/v14-20260306-195347/val_dataset.jsonl'

# Generation config
temperature = 0.7
top_p = 0.9
max_length = 4096
max_new_tokens = 512

# ============ Model Loading ============
def load_model(model_path, lora_path=None):
    """Load model with optional LoRA adapter"""
    print(f"Loading base model: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return model, processor

# ============ Data Parsing ============
def parse_val_dataset(val_path, sample_size=None):
    """Parse validation dataset with optional subsampling"""
    samples = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            messages = sample['messages']
            
            qa_pairs = []
            current_q = None
            for msg in messages:
                if msg['role'] == 'user':
                    current_q = msg['content']
                elif msg['role'] == 'assistant' and current_q:
                    qa_pairs.append({'question': current_q, 'answer': msg['content']})
                    current_q = None
            
            if qa_pairs:
                qa = qa_pairs[0]
                image_path = None
                question_text = ""
                for item in qa['question']:
                    if item['type'] == 'image':
                        image_path = item['image']
                    elif item['type'] == 'text':
                        question_text = item['text']
                
                answer_text = qa['answer'][0]['text'] if qa['answer'] else ""
                
                samples.append({
                    'image': image_path,
                    'question': question_text,
                    'answer': answer_text
                })
    
    print(f"Loaded {len(samples)} validation samples")
    
    if sample_size and sample_size < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, sample_size)
        print(f"Subsampled to {len(samples)} samples for quick validation")
    
    return samples

# ============ Inference ============
def single_infer(model, processor, question, image_path):
    """Single sample inference"""
    
    content = []
    if image_path and os.path.exists(image_path):
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": question})
    
    messages = [{"role": "user", "content": content}]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    response = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    return response.strip()

# ============ Metrics Calculation ============
def calculate_bleu(reference, candidate):
    """Calculate BLEU score"""
    try:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        return sentence_bleu([ref_tokens], cand_tokens)
    except:
        return 0.0

def calculate_medical_accuracy(reference, candidate):
    """Calculate medical terminology accuracy"""
    medical_keywords = [
        'ct', 'mri', 'ultrasound', 'x-ray', 'pet',
        'tumor', 'lesion', 'mass', 'nodule',
        'normal', 'abnormal', 'positive', 'negative',
        'acute', 'chronic', 'benign', 'malignant'
    ]
    
    ref_lower = reference.lower()
    cand_lower = candidate.lower()
    
    matches = 0
    for kw in medical_keywords:
        ref_has = kw in ref_lower
        cand_has = kw in cand_lower
        if ref_has == cand_has:
            matches += 1
    
    return matches / len(medical_keywords)

def calculate_length_metrics(reference, candidate):
    """Calculate length-related metrics"""
    ref_words = len(reference.split())
    cand_words = len(candidate.split())
    
    return {
        'ref_length': ref_words,
        'cand_length': cand_words,
        'length_diff': abs(ref_words - cand_words)
    }

# ============ Main Evaluation ============
def evaluate_models(samples, baseline_model, baseline_processor,
                   finetuned_model, finetuned_processor, output_dir,
                   eval_baseline=True):
    """Evaluate models with optional baseline comparison"""
    
    results = []
    metrics = {
        'baseline': {'bleu': [], 'rouge_l': [], 'exact_match': [], 'medical_acc': []},
        'finetuned': {'bleu': [], 'rouge_l': [], 'exact_match': [], 'medical_acc': []}
    }
    
    rouger = Rouge()
    save_interval = max(10, len(samples) // 10)
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating", file=sys.stdout)):
        question = sample['question']
        image = sample['image']
        reference = sample['answer']
        
        baseline_pred = ""
        if eval_baseline:
            baseline_pred = single_infer(baseline_model, baseline_processor, question, image)
        
        finetuned_pred = single_infer(finetuned_model, finetuned_processor, question, image)
        
        baseline_bleu = calculate_bleu(reference, baseline_pred) if eval_baseline else 0.0
        finetuned_bleu = calculate_bleu(reference, finetuned_pred)
        
        baseline_rouge = rouger.get_scores([reference], [baseline_pred])[0]['rouge-l']['f'] if eval_baseline else 0.0
        finetuned_rouge = rouger.get_scores([reference], [finetuned_pred])[0]['rouge-l']['f']
        
        baseline_em = 1.0 if reference.strip().lower() == baseline_pred.strip().lower() else 0.0 if eval_baseline else 0.0
        finetuned_em = 1.0 if reference.strip().lower() == finetuned_pred.strip().lower() else 0.0
        
        baseline_med = calculate_medical_accuracy(reference, baseline_pred) if eval_baseline else 0.0
        finetuned_med = calculate_medical_accuracy(reference, finetuned_pred)
        
        baseline_len = calculate_length_metrics(reference, baseline_pred) if eval_baseline else {'ref_length': 0, 'cand_length': 0, 'length_diff': 0}
        finetuned_len = calculate_length_metrics(reference, finetuned_pred)
        
        if eval_baseline:
            metrics['baseline']['bleu'].append(baseline_bleu)
            metrics['baseline']['rouge_l'].append(baseline_rouge)
            metrics['baseline']['exact_match'].append(baseline_em)
            metrics['baseline']['medical_acc'].append(baseline_med)
        
        metrics['finetuned']['bleu'].append(finetuned_bleu)
        metrics['finetuned']['rouge_l'].append(finetuned_rouge)
        metrics['finetuned']['exact_match'].append(finetuned_em)
        metrics['finetuned']['medical_acc'].append(finetuned_med)
        
        result = {
            'idx': idx,
            'question': question,
            'image': os.path.basename(image) if image else '',
            'reference': reference,
            'baseline_pred': baseline_pred,
            'finetuned_pred': finetuned_pred,
            'baseline_bleu': baseline_bleu,
            'baseline_rouge_l': baseline_rouge,
            'baseline_em': baseline_em,
            'baseline_med_acc': baseline_med,
            'finetuned_bleu': finetuned_bleu,
            'finetuned_rouge_l': finetuned_rouge,
            'finetuned_em': finetuned_em,
            'finetuned_med_acc': finetuned_med,
            'bleu_improvement': finetuned_bleu - baseline_bleu if eval_baseline else 0.0,
            'rouge_improvement': finetuned_rouge - baseline_rouge if eval_baseline else 0.0,
        }
        result.update(baseline_len)
        result.update({f"finetuned_{k}": v for k, v in finetuned_len.items()})
        
        results.append(result)
        
        if (idx + 1) % save_interval == 0:
            save_results(results, metrics, output_dir, idx + 1, eval_baseline)
            print(f"Saved {idx + 1}/{len(samples)} samples")
    
    save_results(results, metrics, output_dir, len(samples), eval_baseline)
    
    return results, metrics

# ============ Save Results ============
def save_results(results, metrics, output_dir, num_samples, eval_baseline=True):
    """Save results in multiple formats"""
    
    json_path = os.path.join(output_dir, f'eval_results_{num_samples}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    csv_path = os.path.join(output_dir, f'eval_results_{num_samples}.csv')
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # ? 修复：统一使用 _mean 后缀
    summary = {
        'num_samples': num_samples,
        'timestamp': datetime.now().isoformat(),
        'eval_baseline': eval_baseline,
        'baseline': {},
        'finetuned': {}
    }
    
    if eval_baseline:
        summary['baseline'] = {
            'bleu_mean': float(np.mean(metrics['baseline']['bleu'])),
            'bleu_std': float(np.std(metrics['baseline']['bleu'])),
            'rouge_l_mean': float(np.mean(metrics['baseline']['rouge_l'])),
            'rouge_l_std': float(np.std(metrics['baseline']['rouge_l'])),
            'exact_match_mean': float(np.mean(metrics['baseline']['exact_match'])),  # ? 修复：rate -> mean
            'medical_acc_mean': float(np.mean(metrics['baseline']['medical_acc'])),
        }
    
    summary['finetuned'] = {
        'bleu_mean': float(np.mean(metrics['finetuned']['bleu'])),
        'bleu_std': float(np.std(metrics['finetuned']['bleu'])),
        'rouge_l_mean': float(np.mean(metrics['finetuned']['rouge_l'])),
        'rouge_l_std': float(np.std(metrics['finetuned']['rouge_l'])),
        'exact_match_mean': float(np.mean(metrics['finetuned']['exact_match'])),  # ? 修复：rate -> mean
        'medical_acc_mean': float(np.mean(metrics['finetuned']['medical_acc'])),
    }
    
    if eval_baseline:
        for metric in ['bleu', 'rouge_l', 'exact_match', 'medical_acc']:
            baseline_val = summary['baseline'][f'{metric}_mean']
            finetuned_val = summary['finetuned'][f'{metric}_mean']
            improvement = ((finetuned_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            summary[f'{metric}_improvement_pct'] = improvement
    
    summary_path = os.path.join(output_dir, f'summary_{num_samples}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    create_visualizations(metrics, output_dir, num_samples, eval_baseline)
    print_summary(summary, eval_baseline)

# ============ Visualizations ============
def create_visualizations(metrics, output_dir, num_samples, eval_baseline=True):
    """Create visualization plots"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if eval_baseline:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_to_plot = [
            ('bleu', 'BLEU Score', axes[0, 0]),
            ('rouge_l', 'ROUGE-L Score', axes[0, 1]),
            ('exact_match', 'Exact Match Rate', axes[1, 0]),
            ('medical_acc', 'Medical Accuracy', axes[1, 1])
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if isinstance(axes, plt.Axes):
            axes = [axes]
        metrics_to_plot = [
            ('bleu', 'BLEU Score', axes[0]),
            ('medical_acc', 'Medical Accuracy', axes[1])
        ]
    
    for metric_key, title, ax in metrics_to_plot:
        finetuned_vals = metrics['finetuned'][metric_key]
        
        if eval_baseline:
            baseline_vals = metrics['baseline'][metric_key]
            data = [baseline_vals, finetuned_vals]
            bp = ax.boxplot(data, labels=['Baseline', 'Fine-tuned'], patch_artist=True)
      
            colors = ['lightcoral', 'lightblue']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            baseline_mean = np.mean(baseline_vals)
            finetuned_mean = np.mean(finetuned_vals)
            improvement = ((finetuned_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            ax.text(1.5, 0.95, f'Fine-tuned: {finetuned_mean:.4f}\n(+{improvement:.2f}%)', 
                   transform=ax.transAxes, ha='center', va='top', fontsize=9)
        else:
            ax.boxplot([finetuned_vals], labels=['Fine-tuned'], patch_artist=True)

            ax.get_children()[0].set_facecolor('lightblue') 
            finetuned_mean = np.mean(finetuned_vals)
            ax.text(0.5, 0.95, f'Mean: {finetuned_mean:.4f}', 
                   transform=ax.transAxes, ha='center', va='top', fontsize=9)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
    
    plt.suptitle(f'Model Evaluation Results (n={num_samples})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'metrics_comparison_{num_samples}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============ Print Summary ============
def print_summary(summary, eval_baseline=True):
    """Print evaluation summary"""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nNumber of samples: {summary['num_samples']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Evaluate Baseline: {eval_baseline}")
    
    if eval_baseline and summary['baseline']:
        print("\n" + "-"*70)
        print("BASELINE MODEL:")
        print("-"*70)
        for metric, value in summary['baseline'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    print("\n" + "-"*70)
    print("FINE-TUNED MODEL:")
    print("-"*70)
    for metric, value in summary['finetuned'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    if eval_baseline:
        print("\n" + "="*70)
        print("PERFORMANCE IMPROVEMENT:")
        print("="*70)
        for metric in ['bleu', 'rouge_l', 'exact_match', 'medical_acc']:
            improvement = summary.get(f'{metric}_improvement_pct', 0)
            status = "[+]" if improvement > 0 else "[-]"
            print(f"  {status} {metric}: {improvement:+.2f}%")
    
    print("\n" + "="*70)

# ============ Main ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical VQA Evaluation')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--eval_baseline', action='store_true', default=True,
                       help='Evaluate baseline model (default: True)')
    parser.add_argument('--no-eval_baseline', action='store_false', dest='eval_baseline',
                       help='Skip baseline model evaluation')
    parser.add_argument('--base_model', type=str, default=BASE_MODEL_PATH,
                       help='Base model path')
    parser.add_argument('--lora_path', type=str, default=LORA_CKPT_DIR,
                       help='LoRA checkpoint path')
    parser.add_argument('--val_dataset', type=str, default=VAL_DATASET_PATH,
                       help='Validation dataset path')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')
    
    args = parser.parse_args()
    
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTPUT_DIR = f'/home/wshenah/project/eval_results/{TIMESTAMP}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting Evaluation")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Validation Set: {args.val_dataset}")
    print(f"Sample Size: {args.sample_size if args.sample_size else 'All'}")
    print(f"Evaluate Baseline: {args.eval_baseline}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    baseline_model, baseline_processor = None, None
    if args.eval_baseline:
        baseline_model, baseline_processor = load_model(args.base_model)
    
    finetuned_model, finetuned_processor = load_model(args.base_model, args.lora_path)
    
    samples = parse_val_dataset(args.val_dataset, args.sample_size)
    
    results, metrics = evaluate_models(
        samples,
        baseline_model, baseline_processor,
        finetuned_model, finetuned_processor,
        OUTPUT_DIR,
        eval_baseline=args.eval_baseline
    )
    
    print(f"\nEvaluation completed. Results saved to: {OUTPUT_DIR}")