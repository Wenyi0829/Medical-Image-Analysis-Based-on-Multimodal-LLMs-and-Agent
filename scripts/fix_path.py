# coding=gbk
#!/usr/bin/env python
import json
import os

# 配置：您图片的实际绝对路径目录
IMAGE_DIR = "/home/wshenah/LLaVA-Med/data/images"  # ?? 请根据实际情况修改
INPUT_JSON = "/home/wshenah/LLaVA-Med/data/instruct/llava_med_qwen_format.json"
OUTPUT_JSON = "/home/wshenah/LLaVA-Med/data/instruct/llava_med_qwen_format_fixed.json"

def fix_paths():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = []
    valid_count = 0
    invalid_count = 0
    
    for item in data:
        messages = item.get('messages', [])
        sample_valid = True
        new_messages = []
        
        for msg in messages:
            content = msg.get('content', [])
            new_content = []
            
            for block in content:
                if block.get('type') == 'image':
                    img_path = block.get('image', '')
                    # 如果是相对路径，转换为绝对路径
                    if not os.path.isabs(img_path):
                        # 尝试拼接基础路径
                        abs_path = os.path.join(IMAGE_DIR, os.path.basename(img_path))
                    else:
                        abs_path = img_path
                    
                    # 检查文件是否存在
                    if os.path.exists(abs_path):
                        block['image'] = abs_path
                        new_content.append(block)
                    else:
                        # 图片不存在，标记样本无效
                        sample_valid = False
                        break
                else:
                    new_content.append(block)
            
            if not sample_valid:
                break
            msg['content'] = new_content
            new_messages.append(msg)
        
        if sample_valid and len(new_messages) > 0:
            fixed_data.append({"messages": new_messages})
            valid_count += 1
        else:
            invalid_count += 1
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"? 处理完成！")
    print(f"   原始样本：{len(data)}")
    print(f"   有效样本：{valid_count}")
    print(f"   无效样本：{invalid_count} (图片缺失或损坏)")
    print(f"   输出文件：{OUTPUT_JSON}")

if __name__ == "__main__":
    fix_paths()