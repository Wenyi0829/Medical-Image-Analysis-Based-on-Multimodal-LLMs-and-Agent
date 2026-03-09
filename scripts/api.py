from openai import OpenAI
import base64
import requests

client = OpenAI(
    api_key="EMPTY", 
    base_url="http://127.0.0.1:22002/v1",
    timeout=3600
)

# method 1: image URL
def chat_with_image_url(image_url, question):
    response = client.chat.completions.create(
        model="/home/wshenah/project/models/Qwen3-VL-8B-Thinking",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question}
            ]
        }],
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8
    )
    return response.choices[0].message.content

# method 2: local image
def chat_with_local_image(image_path, question):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="/home/wshenah/project/models/Qwen3-VL-8B-Thinking",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": question}
            ]
        }],
        max_tokens=2048
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    result = chat_with_image_url(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
        "What animal is on the candy?"
    )
    print(f"Response: {result}")