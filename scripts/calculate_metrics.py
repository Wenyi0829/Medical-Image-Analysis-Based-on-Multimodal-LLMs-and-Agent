#!/usr/bin/env python3
import json
import argparse
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np

def calculate_metrics(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    metrics = {
        'bleu_scores': [],
        'rouge_scores': [],
        'inference_time_base': [],
        'inference_time_lora': []
    }
    
    rouge = Rouge()
    
    for item in results:

        reference = item['ground_truth'].split()
        hypothesis_base = item['base_response'].split()
        hypothesis_lora = item['lora_response'].split()
        
        try:
            bleu_base = sentence_bleu([reference], hypothesis_base)
            bleu_lora = sentence_bleu([reference], hypothesis_lora)
            metrics['bleu_scores'].append({
                'base': bleu_base,
                'lora': bleu_lora,
                'improvement': bleu_lora - bleu_base
            })
        except:
            pass

        try:
            rouge_scores = rouge.get_scores(
                item['lora_response'], 
                item['ground_truth']
            )[0]
            metrics['rouge_scores'].append(rouge_scores)
        except:
            pass
        
        metrics['inference_time_base'].append(item['base_inference_time'])
        metrics['inference_time_lora'].append(item['lora_inference_time'])

    summary = {
        'avg_bleu_base': np.mean([m['base'] for m in metrics['bleu_scores']]),
        'avg_bleu_lora': np.mean([m['lora'] for m in metrics['bleu_scores']]),
        'bleu_improvement': np.mean([m['improvement'] for m in metrics['bleu_scores']]),
        'avg_rouge_l': np.mean([m['rouge-l']['f'] for m in metrics['rouge_scores']]),
        'avg_rouge_1': np.mean([m['rouge-1']['f'] for m in metrics['rouge_scores']]),
        'avg_rouge_2': np.mean([m['rouge-2']['f'] for m in metrics['rouge_scores']]),
        'avg_inference_time_base': np.mean(metrics['inference_time_base']),
        'avg_inference_time_lora': np.mean(metrics['inference_time_lora'])
    }
    
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True)
    args = parser.parse_args()
    
    summary = calculate_metrics(args.results_path)
    print(json.dumps(summary, indent=2))