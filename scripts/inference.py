import os
import torch
os.environ['HF_HOME'] = '/home/wshenah/project/hf_cache'

from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "/home/wshenah/project/models/Qwen3-VL-8B-Thinking"
processor = AutoProcessor.from_pretrained(model_path)
#model = AutoModelForImageTextToText.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
     model_path,
     dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))