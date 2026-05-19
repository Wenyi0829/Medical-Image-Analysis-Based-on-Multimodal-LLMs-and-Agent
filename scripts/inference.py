import os
import sys
import torch

os.environ["HF_HOME"] = "/home/wshenah/project/hf_cache"

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "/home/wshenah/project/models/Qwen3-VL-8B-Thinking"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

def _parse_user_line(line: str) -> tuple[str | None, str]:
    """
    Supported formats:
      - "hello"                         -> (None, "hello")
      - "img:/abs/path.jpg 这是什么？"    -> ("/abs/path.jpg", "这是什么？")
    """
    s = line.strip()
    if not s:
        return None, ""
    if s.lower().startswith("img:"):
        rest = s[4:].lstrip()
        if not rest:
            return None, ""
        parts = rest.split(maxsplit=1)
        img_path = parts[0]
        text = parts[1] if len(parts) > 1 else ""
        return img_path, text
    return None, s


def run_chat():
    if not sys.stdin.isatty():
        print(
            "未检测到交互式终端（stdin 不是 TTY）。"
            "请用 `srun ... --pty /bin/bash` 登录计算节点后再运行本脚本。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "模型已加载。支持多轮图文对话。\n"
        "- 纯文本：直接输入问题\n"
        '- 带图：输入 `img:/绝对路径/xxx.jpg 你的问题`（图片路径可无空格；问题可留空）\n'
        "- 退出：exit / quit / q\n"
    )
    messages = []

    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("已退出。")
            break

        img_path, text = _parse_user_line(user_input)
        content = []
        if img_path:
            if not os.path.exists(img_path):
                print(f"assistant> 找不到图片文件：{img_path}\n")
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"assistant> 读取图片失败：{img_path}（{e}）\n")
                continue
            content.append({"type": "image", "image": image})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            print("assistant> 请输入文本，或使用 `img:/path/to.jpg 你的问题`。\n")
            continue

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512)

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        response_text = processor.decode(new_tokens, skip_special_tokens=True)
        print(f"assistant> {response_text}\n")

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            }
        )


if __name__ == "__main__":
    run_chat()
