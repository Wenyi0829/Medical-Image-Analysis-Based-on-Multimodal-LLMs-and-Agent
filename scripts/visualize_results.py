import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    base_times = [r['base_inference_time'] for r in results]
    lora_times = [r['lora_inference_time'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist([base_times, lora_times], label=['Base', 'LoRA'], alpha=0.7)
    axes[0].set_xlabel('Inference Time (s)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Inference Time Distribution')
    axes[0].legend()

    sample_ids = np.random.choice(len(results), 5, replace=False)
    axes[1].axis('off')
    for i, idx in enumerate(sample_ids):
        r = results[idx]
        text = f"Sample {idx}:\nQ: {r['question'][:50]}...\n"
        text += f"Base: {r['base_response'][:100]}...\n"
        text += f"LoRA: {r['lora_response'][:100]}...\n\n"
        axes[1].text(0.1, 0.9-i*0.2, text, transform=axes[1].transAxes, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    print("Visualization saved to comparison_results.png")

if __name__ == '__main__':
    import sys
    plot_comparison(sys.argv[1])