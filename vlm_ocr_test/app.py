from __future__ import annotations

import os
import socket
import threading
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from mineru.backend.vlm.vlm_analyze import ModelSingleton, predictor_execution_guard
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_image_tools import load_images_from_pdf_core
from mineru_vl_utils import MinerUSamplingParams


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_VLM_PROMPT = (
    "请提取这页图片中的全部可见文字内容。"
    "只输出文字本身，按自然阅读顺序换行，不要解释，不要输出 bbox。"
)
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7868
VLM_ENGINE = "transformers"
VLM_TEXT_MODE = "文字识别"
VLM_LAYOUT_MODE = "版面检测"

_ocr_models: dict[str, PytorchPaddleOCR] = {}
_ocr_lock = threading.Lock()
_vlm_lock = threading.Lock()


def find_available_port(start_port: int, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((DEFAULT_SERVER_NAME, port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {start_port}-{start_port + attempts - 1}")


def launch_demo() -> None:
    server_name = os.getenv("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME)
    requested_port = int(os.getenv("GRADIO_SERVER_PORT", str(DEFAULT_SERVER_PORT)))
    server_port = find_available_port(requested_port)
    if server_port != requested_port:
        print(f"Port {requested_port} is busy, using {server_port} instead.")
    demo.launch(server_name=server_name, server_port=server_port)


def validate_file(file_path: str | None) -> Path:
    if not file_path:
        raise gr.Error("请先上传 PDF 或图片文件。")

    path = Path(file_path)
    if not path.exists():
        raise gr.Error(f"文件不存在: {path}")

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise gr.Error(f"不支持的文件类型: {path.suffix}。支持: {allowed}")

    return path


def normalize_page_range(start_page_id: int | float, end_page_id: int | float) -> tuple[int, int]:
    start = max(0, int(start_page_id or 0))
    end = int(end_page_id if end_page_id is not None else 99999)
    end = max(start, end)
    return start, end


def load_page_images(
    file_path: str | None,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
) -> list[Image.Image]:
    path = validate_file(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        start, end = normalize_page_range(start_page_id, end_page_id)
        images = load_images_from_pdf_core(
            path.read_bytes(),
            dpi=int(dpi or 200),
            start_page_id=start,
            end_page_id=end,
            image_type=ImageType.PIL,
        )
        page_images = [item["img_pil"].convert("RGB") for item in images]
    else:
        page_images = [Image.open(path).convert("RGB")]

    if not page_images:
        raise gr.Error("没有读取到可解析页面。")
    return page_images


def get_ocr_model(language: str) -> PytorchPaddleOCR:
    language = language or "ch"
    with _ocr_lock:
        if language not in _ocr_models:
            _ocr_models[language] = PytorchPaddleOCR(lang=language)
        return _ocr_models[language]


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_direct_ocr(
    file_path: str | None,
    language: str,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
    include_page_headers: bool,
) -> str:
    images = load_page_images(file_path, start_page_id, end_page_id, dpi)
    ocr_model = get_ocr_model(language)

    pages: list[str] = []
    for index, image in enumerate(images, start=1):
        result = ocr_model.ocr(pil_to_bgr(image), det=True, rec=True)
        lines: list[str] = []
        if result and result[0]:
            for item in result[0]:
                if not item or len(item) < 2:
                    continue
                rec = item[1]
                if isinstance(rec, (list, tuple)) and rec:
                    text = str(rec[0]).strip()
                    if text:
                        lines.append(text)

        page_text = "\n".join(lines).strip()
        if include_page_headers and len(images) > 1:
            page_text = f"--- Page {index} ---\n{page_text}"
        pages.append(page_text)

    text = "\n\n".join(page for page in pages if page.strip()).strip()
    return text or "[OCR 没有识别到文字]"


def get_vlm_predictor():
    return ModelSingleton().get_model(VLM_ENGINE, None, None)


def run_direct_vlm(
    file_path: str | None,
    prompt: str,
    vlm_mode: str,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
    max_new_tokens: int | float,
    include_page_headers: bool,
) -> str:
    images = load_page_images(file_path, start_page_id, end_page_id, dpi)

    with _vlm_lock:
        predictor = get_vlm_predictor()
        with predictor_execution_guard(predictor):
            if vlm_mode == VLM_LAYOUT_MODE:
                outputs = predictor.batch_layout_detect(images)
            else:
                actual_prompt = (prompt or DEFAULT_VLM_PROMPT).strip()
                sampling_params = MinerUSamplingParams(
                    temperature=0.0,
                    top_p=0.01,
                    top_k=1,
                    max_new_tokens=int(max_new_tokens or 4096),
                )
                outputs = predictor.client.batch_predict(
                    images,
                    [actual_prompt] * len(images),
                    sampling_params=sampling_params,
                )

    pages = []
    for index, output in enumerate(outputs, start=1):
        if vlm_mode == VLM_LAYOUT_MODE:
            page_text = format_layout_output(output)
        else:
            page_text = str(output or "").strip()
        if include_page_headers and len(outputs) > 1:
            page_text = f"--- Page {index} ---\n{page_text}"
        pages.append(page_text)

    text = "\n\n".join(page for page in pages if page.strip()).strip()
    return text or "[VLM 没有返回文字]"


def format_layout_output(blocks) -> str:
    rows = []
    for index, block in enumerate(blocks or [], start=1):
        rows.append(
            {
                "index": index,
                "type": block.get("type"),
                "bbox": block.get("bbox"),
                "angle": block.get("angle"),
                "merge_prev": block.get("merge_prev"),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2) if rows else "[]"


def parse_vlm_chat(
    message: str,
    history: list[dict[str, str]] | None,
    file_path: str | None,
    vlm_mode: str,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
    max_new_tokens: int | float,
    include_page_headers: bool,
) -> tuple[list[dict[str, str]], str]:
    history = history or []
    user_prompt = (message or DEFAULT_VLM_PROMPT).strip()
    history.append({"role": "user", "content": user_prompt})

    try:
        text = run_direct_vlm(
            file_path=file_path,
            prompt=user_prompt,
            vlm_mode=vlm_mode,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            dpi=dpi,
            max_new_tokens=max_new_tokens,
            include_page_headers=include_page_headers,
        )
    except Exception as exc:
        text = f"VLM 调用失败: {exc}"

    history.append({"role": "assistant", "content": text})
    return history, ""


def clear_model_cache() -> str:
    with _ocr_lock:
        _ocr_models.clear()
    ModelSingleton().shutdown()
    return "模型缓存已清空。"


with gr.Blocks(title="MinerU VLM/OCR 单模型测试") as demo:
    gr.Markdown("# MinerU VLM/OCR 单模型测试")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            shared_file = gr.File(
                label="PDF 或图片",
                file_types=sorted(SUPPORTED_SUFFIXES),
                type="filepath",
            )
            start_page_id = gr.Number(
                label="起始页",
                value=0,
                precision=0,
                minimum=0,
            )
            end_page_id = gr.Number(
                label="结束页",
                value=99999,
                precision=0,
                minimum=0,
            )
            dpi = gr.Number(
                label="PDF 渲染 DPI",
                value=200,
                precision=0,
                minimum=72,
            )
            include_page_headers = gr.Checkbox(label="多页输出页码", value=True)
            vlm_mode = gr.Radio(
                label="VLM 模式",
                choices=[VLM_TEXT_MODE, VLM_LAYOUT_MODE],
                value=VLM_TEXT_MODE,
            )
            max_new_tokens = gr.Number(
                label="VLM 最大输出 token",
                value=4096,
                precision=0,
                minimum=128,
            )
            ocr_language = gr.Dropdown(
                label="OCR 语言",
                choices=[
                    "ch",
                    "ch_lite",
                    "ch_server",
                    "en",
                    "japan",
                    "korean",
                    "chinese_cht",
                    "latin",
                    "arabic",
                    "cyrillic",
                    "devanagari",
                ],
                value="ch",
            )
            ocr_button = gr.Button("开始 OCR", variant="primary")
            clear_button = gr.Button("清空模型缓存")
            cache_status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=3, min_width=520):
            with gr.Tabs():
                with gr.Tab("VLM 对话"):
                    vlm_chat = gr.Chatbot(label="VLM 输出", height=560)
                    vlm_message = gr.Textbox(
                        label="Prompt",
                        value=DEFAULT_VLM_PROMPT,
                        lines=3,
                    )
                    with gr.Row():
                        vlm_send = gr.Button("发送", variant="primary")
                        vlm_clear = gr.Button("清空对话")

                with gr.Tab("OCR 直接返回"):
                    ocr_output = gr.Textbox(
                        label="OCR 文本",
                        lines=30,
                    )

    clear_button.click(clear_model_cache, outputs=[cache_status])
    vlm_send.click(
        parse_vlm_chat,
        inputs=[
            vlm_message,
            vlm_chat,
            shared_file,
            vlm_mode,
            start_page_id,
            end_page_id,
            dpi,
            max_new_tokens,
            include_page_headers,
        ],
        outputs=[vlm_chat, vlm_message],
    )
    vlm_message.submit(
        parse_vlm_chat,
        inputs=[
            vlm_message,
            vlm_chat,
            shared_file,
            vlm_mode,
            start_page_id,
            end_page_id,
            dpi,
            max_new_tokens,
            include_page_headers,
        ],
        outputs=[vlm_chat, vlm_message],
    )
    vlm_clear.click(lambda: [], outputs=[vlm_chat])
    ocr_button.click(
        run_direct_ocr,
        inputs=[
            shared_file,
            ocr_language,
            start_page_id,
            end_page_id,
            dpi,
            include_page_headers,
        ],
        outputs=[ocr_output],
    )


if __name__ == "__main__":
    launch_demo()
