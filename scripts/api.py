import argparse
import base64
import mimetypes
import os
from typing import Optional

from openai import OpenAI


DEFAULT_MODEL = "/home/wshenah/project/models/Qwen3-VL-8B-Thinking"


def build_base_url() -> str:
    explicit = os.getenv("VLLM_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port = os.getenv("VLLM_PORT", "22002")
    return f"http://{host}:{port}/v1"


BASE_URL = build_base_url()
MODEL_NAME = os.getenv("VLLM_MODEL", DEFAULT_MODEL)

client = OpenAI(
    api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    base_url=BASE_URL,
    timeout=3600,
)


def _local_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _build_user_content(question: str, image_url: Optional[str] = None, image_path: Optional[str] = None):
    content = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    elif image_path:
        content.append({"type": "image_url", "image_url": {"url": _local_image_to_data_url(image_path)}})
    content.append({"type": "text", "text": question})
    return content


def chat_once(question: str, image_url: Optional[str] = None, image_path: Optional[str] = None):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": _build_user_content(question, image_url, image_path)}],
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
    )
    return response.choices[0].message.content


def launch_web_ui() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: gradio. Install with `pip install gradio` then rerun with `--web`."
        ) from exc

    def _respond(message, image, history):
        messages = []
        for turn in history:
            user_text = turn[0] or ""
            assistant_text = turn[1] or ""
            if user_text:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        user_content = []
        if image is not None:
            user_content.append({"type": "image_url", "image_url": {"url": _local_image_to_data_url(image)}})
        user_content.append({"type": "text", "text": message})
        messages.append({"role": "user", "content": user_content})

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
        )
        return resp.choices[0].message.content

    with gr.Blocks(title="vLLM Chat UI") as demo:
        gr.Markdown(f"## vLLM Chat UI\nBase URL: `{BASE_URL}`\n\nModel: `{MODEL_NAME}`")
        chatbot = gr.Chatbot(height=520)
        with gr.Row():
            txt = gr.Textbox(label="Message", placeholder="Ask anything...", scale=5)
            img = gr.Image(type="filepath", label="Optional image", scale=2)
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

        def _on_send(message, image, history):
            if not message.strip():
                return "", history
            answer = _respond(message, image, history)
            history = history + [[message, answer]]
            return "", history

        send.click(_on_send, inputs=[txt, img, chatbot], outputs=[txt, chatbot])
        txt.submit(_on_send, inputs=[txt, img, chatbot], outputs=[txt, chatbot])
        clear.click(lambda: [], outputs=chatbot)

    web_host = os.getenv("WEB_HOST", "127.0.0.1")
    web_port = int(os.getenv("WEB_PORT", "7860"))
    demo.launch(server_name=web_host, server_port=web_port, share=False)


def main():
    parser = argparse.ArgumentParser(description="Qwen vLLM API helper")
    parser.add_argument("--web", action="store_true", help="Launch a browser UI (Gradio).")
    parser.add_argument("--question", type=str, default="What animal is on the candy?")
    parser.add_argument("--image-url", type=str, default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG")
    parser.add_argument("--image-path", type=str, default=None)
    args = parser.parse_args()

    if args.web:
        launch_web_ui()
        return

    result = chat_once(args.question, image_url=args.image_url, image_path=args.image_path)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Response: {result}")


if __name__ == "__main__":
    main()