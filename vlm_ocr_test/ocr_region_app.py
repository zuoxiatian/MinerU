from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_image_tools import load_images_from_pdf_core


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
DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_SERVER_PORT = 7870

_ocr_models: dict[str, PytorchPaddleOCR] = {}
_ocr_lock = threading.Lock()


def find_available_port(start_port: int, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((DEFAULT_SERVER_NAME, port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {start_port}-{start_port + attempts - 1}")


def launch_demo() -> None:
    port = find_available_port(DEFAULT_SERVER_PORT)
    if port != DEFAULT_SERVER_PORT:
        print(f"Port {DEFAULT_SERVER_PORT} is busy, using {port} instead.")
    demo.launch(server_name=DEFAULT_SERVER_NAME, server_port=port)


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
    end = int(end_page_id if end_page_id is not None else start)
    return start, max(start, end)


def load_page_images(
    file_path: str | None,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
) -> list[Image.Image]:
    path = validate_file(file_path)
    if path.suffix.lower() == ".pdf":
        start, end = normalize_page_range(start_page_id, end_page_id)
        images = load_images_from_pdf_core(
            path.read_bytes(),
            dpi=int(dpi or 200),
            start_page_id=start,
            end_page_id=end,
            image_type=ImageType.PIL,
        )
        return [item["img_pil"].convert("RGB") for item in images]
    return [Image.open(path).convert("RGB")]


def get_ocr_model(language: str) -> PytorchPaddleOCR:
    language = language or "ch"
    with _ocr_lock:
        if language not in _ocr_models:
            _ocr_models[language] = PytorchPaddleOCR(lang=language)
        return _ocr_models[language]


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def polygon_to_record(points: np.ndarray, image_size: tuple[int, int], index: int) -> dict:
    width, height = image_size
    xs = points[:, 0]
    ys = points[:, 1]
    bbox = [
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs)),
        float(np.max(ys)),
    ]
    return {
        "index": index,
        "polygon": [[float(x), float(y)] for x, y in points.tolist()],
        "bbox": bbox,
        "normalized_bbox": [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ],
    }


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def x_overlap_ratio(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    smaller_width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / smaller_width


def vertical_gap(a: list[float], b: list[float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def should_merge_boxes(
    a: list[float],
    b: list[float],
    max_vertical_gap_px: float,
    min_x_overlap_ratio: float,
) -> bool:
    if vertical_gap(a, b) > max_vertical_gap_px:
        return False
    if x_overlap_ratio(a, b) >= min_x_overlap_ratio:
        return True

    a_center_x = (a[0] + a[2]) / 2
    b_center_x = (b[0] + b[2]) / 2
    max_width = max(a[2] - a[0], b[2] - b[0])
    return abs(a_center_x - b_center_x) <= max_width * 0.35


def union_bbox(a: list[float], b: list[float]) -> list[float]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def merge_adjacent_records(
    records: list[dict],
    max_vertical_gap_px: float,
    min_x_overlap_ratio: float,
    image_size: tuple[int, int],
) -> list[dict]:
    groups = [
        {
            "bbox": list(record["bbox"]),
            "children": [record["index"]],
        }
        for record in records
    ]

    changed = True
    while changed:
        changed = False
        merged_groups = []
        used = [False] * len(groups)
        for i, group in enumerate(groups):
            if used[i]:
                continue
            current = {
                "bbox": list(group["bbox"]),
                "children": list(group["children"]),
            }
            used[i] = True
            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                if should_merge_boxes(
                    current["bbox"],
                    groups[j]["bbox"],
                    max_vertical_gap_px,
                    min_x_overlap_ratio,
                ):
                    current["bbox"] = union_bbox(current["bbox"], groups[j]["bbox"])
                    current["children"].extend(groups[j]["children"])
                    used[j] = True
                    changed = True
            merged_groups.append(current)
        groups = merged_groups

    width, height = image_size
    groups.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    merged_records = []
    for index, group in enumerate(groups, start=1):
        bbox = group["bbox"]
        merged_records.append(
            {
                "index": index,
                "bbox": bbox,
                "normalized_bbox": [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ],
                "children": sorted(group["children"]),
            }
        )
    return merged_records


def draw_regions(
    image: Image.Image,
    records: list[dict],
    merged_records: list[dict],
    line_width: int,
    show_raw: bool,
    show_merged: bool,
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    if show_raw:
        for record in records:
            polygon = [(float(x), float(y)) for x, y in record["polygon"]]
            if len(polygon) < 4:
                continue
            draw.line([*polygon, polygon[0]], fill=(255, 0, 0), width=int(line_width))
            x, y = polygon[0]
            label = f"R{record['index']}"
            text_bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(text_bbox, fill=(255, 0, 0))
            draw.text((x, y), label, fill=(255, 255, 255), font=font)

    if show_merged:
        for record in merged_records:
            x1, y1, x2, y2 = record["bbox"]
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 102, 255),
                width=max(1, int(line_width) + 1),
            )
            label = f"M{record['index']}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill=(0, 102, 255))
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return annotated


def detect_ocr_regions(
    file_path: str | None,
    language: str,
    start_page_id: int | float,
    end_page_id: int | float,
    dpi: int | float,
    line_width: int | float,
    merge_enabled: bool,
    max_vertical_gap_px: int | float,
    min_x_overlap_ratio: float,
    show_raw: bool,
    show_merged: bool,
) -> tuple[list[Image.Image], str]:
    images = load_page_images(file_path, start_page_id, end_page_id, dpi)
    ocr_model = get_ocr_model(language)

    annotated_pages: list[Image.Image] = []
    json_pages: list[dict] = []
    for page_index, image in enumerate(images, start=1):
        det_result = ocr_model.ocr(pil_to_bgr(image), det=True, rec=False)
        records = []
        if det_result and det_result[0]:
            for index, box in enumerate(det_result[0], start=1):
                points = np.asarray(box, dtype=float)
                if points.shape != (4, 2):
                    continue
                records.append(polygon_to_record(points, image.size, index))

        merged_records = (
            merge_adjacent_records(
                records,
                float(max_vertical_gap_px or 12),
                float(min_x_overlap_ratio or 0.25),
                image.size,
            )
            if merge_enabled
            else []
        )

        annotated_pages.append(
            draw_regions(
                image,
                records,
                merged_records,
                int(line_width or 3),
                show_raw,
                show_merged,
            )
        )
        json_pages.append(
            {
                "page": page_index,
                "width": image.width,
                "height": image.height,
                "count": len(records),
                "regions": records,
                "merged_count": len(merged_records),
                "merged_regions": merged_records,
            }
        )

    return annotated_pages, json.dumps(json_pages, ensure_ascii=False, indent=2)


with gr.Blocks(title="OCR 文字区域检测测试") as demo:
    gr.Markdown("# OCR 文字区域检测测试")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            file_input = gr.File(
                label="PDF 或图片",
                file_types=sorted(SUPPORTED_SUFFIXES),
                type="filepath",
            )
            language = gr.Dropdown(
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
            start_page_id = gr.Number(label="起始页", value=0, precision=0, minimum=0)
            end_page_id = gr.Number(label="结束页", value=0, precision=0, minimum=0)
            dpi = gr.Number(label="PDF 渲染 DPI", value=200, precision=0, minimum=72)
            line_width = gr.Number(label="框线宽度", value=3, precision=0, minimum=1)
            merge_enabled = gr.Checkbox(label="合并上下相邻框", value=True)
            max_vertical_gap_px = gr.Number(
                label="最大垂直间距 px",
                value=12,
                precision=0,
                minimum=0,
            )
            min_x_overlap_ratio = gr.Number(
                label="最小横向重叠比例",
                value=0.25,
                minimum=0,
                maximum=1,
            )
            show_raw = gr.Checkbox(label="显示原始框(红)", value=True)
            show_merged = gr.Checkbox(label="显示合并框(蓝)", value=True)
            detect_button = gr.Button("检测文字区域", variant="primary")

        with gr.Column(scale=3, min_width=620):
            annotated_output = gr.Gallery(
                label="标注结果",
                columns=1,
                height=620,
            )
            json_output = gr.Textbox(label="检测框 JSON", lines=18)

    detect_button.click(
        detect_ocr_regions,
        inputs=[
            file_input,
            language,
            start_page_id,
            end_page_id,
            dpi,
            line_width,
            merge_enabled,
            max_vertical_gap_px,
            min_x_overlap_ratio,
            show_raw,
            show_merged,
        ],
        outputs=[annotated_output, json_output],
    )


if __name__ == "__main__":
    launch_demo()
