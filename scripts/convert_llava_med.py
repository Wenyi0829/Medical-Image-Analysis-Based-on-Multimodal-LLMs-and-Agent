# coding=gbk
#!/usr/bin/env python
import json
import os

def convert_llava_to_qwen(input_path, output_path, image_dir):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        messages = []
        for turn in item['conversatons']:  # 注意原数据拼写是conversatons
            if turn['from'] == 'human':
                # 分离<image>占位符和文本
                text = turn['value'].replace('<image>', '').strip()
                content = [
                    {"type": "image", "image": os.path.join(image_dir, item['image'])},
                    {"type": "text", "text": text}
                ]
                messages.append({"role": "user", "content": content})
            else:  # gpt/assistant
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn['value']}]
                })
        
        converted.append({"messages": messages})
    
    with open(output_path, 'w') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"? Converted {len(converted)} samples to {output_path}")

if __name__ == "__main__":
    convert_llava_to_qwen(
        "data/instruct/llava_med_instruct_10k.json",
        "data/instruct/llava_med_qwen_format.json",
        "data/images"
    )