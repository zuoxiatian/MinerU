# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image

from mineru.backend.vlm_ocr_native.bbox import pdf_to_unit_bbox
from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR


_OCR_MODELS: dict[str, PytorchPaddleOCR] = {}
_OCR_LOCK = threading.Lock()


def get_ocr_model(language: str | None = None) -> PytorchPaddleOCR:
    language = language or os.getenv("MINERU_VLM_OCR_NATIVE_LANG", "ch")
    with _OCR_LOCK:
        if language not in _OCR_MODELS:
            _OCR_MODELS[language] = PytorchPaddleOCR(lang=language)
        return _OCR_MODELS[language]


def extract_ocr_text_blocks(
    image: Image.Image,
    *,
    scale: float,
    page_width: int,
    page_height: int,
    language: str | None = None,
    max_vertical_gap_px: float | None = None,
    min_x_overlap_ratio: float | None = None,
    layout_hint: dict | None = None,
    type_regions: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    records = detect_ocr_records(image, language=language)
    merged_records = merge_adjacent_records(
        records,
        float(max_vertical_gap_px if max_vertical_gap_px is not None else os.getenv("MINERU_VLM_OCR_NATIVE_MERGE_GAP_PX", 12)),
        float(min_x_overlap_ratio if min_x_overlap_ratio is not None else os.getenv("MINERU_VLM_OCR_NATIVE_MIN_X_OVERLAP", 0.25)),
        image.size,
    )
    merged_records = _sort_merged_records_by_layout_hint(merged_records, scale, layout_hint)

    blocks = []
    raw_by_index = {record["index"]: record for record in records}
    for index, merged in enumerate(merged_records, start=1):
        children = [raw_by_index[child] for child in merged["children"] if child in raw_by_index]
        content = _join_child_text(children)
        if not content.strip():
            continue
        pdf_bbox = [value / max(scale, 1e-6) for value in merged["bbox"]]
        block_type, matched_region = _classify_block_type_by_regions(pdf_bbox, type_regions)
        blocks.append(
            {
                "type": block_type,
                "bbox": pdf_to_unit_bbox(pdf_bbox, page_width, page_height),
                "angle": 0,
                "content": content,
                "index": len(blocks) + 1,
                "source": "ocr",
                "_ocr_children": merged["children"],
                "_ocr_type_region": matched_region,
            }
        )

    debug = {
        "raw_count": len(records),
        "merged_count": len(merged_records),
        "raw_regions": records,
        "merged_regions": merged_records,
        "layout_hint": layout_hint or {},
        "type_regions": type_regions or [],
    }
    return blocks, debug


def detect_ocr_records(image: Image.Image, language: str | None = None) -> list[dict]:
    ocr_model = get_ocr_model(language)
    result = ocr_model.ocr(_pil_to_bgr(image), det=True, rec=True)
    records = []
    if not result or not result[0]:
        return records
    for index, item in enumerate(result[0], start=1):
        record = _ocr_item_to_record(item, image.size, index)
        if record is not None:
            records.append(record)
    return records


def _ocr_item_to_record(item: Any, image_size: tuple[int, int], index: int) -> dict | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    points = np.asarray(item[0], dtype=float)
    if points.shape != (4, 2):
        return None
    rec = item[1]
    if isinstance(rec, (list, tuple)) and rec:
        text = str(rec[0] or "")
        score = float(rec[1]) if len(rec) > 1 else None
    else:
        text = str(rec or "")
        score = None
    record = polygon_to_record(points, image_size, index)
    record["content"] = text
    record["score"] = score
    return record


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


def _sort_merged_records_by_layout_hint(
    records: list[dict],
    scale: float,
    layout_hint: dict | None,
) -> list[dict]:
    if not records:
        return records
    if not isinstance(layout_hint, dict) or layout_hint.get("flow") != "double_column":
        return records
    columns = layout_hint.get("columns") or []
    if len(columns) < 2:
        return records

    column_centers = []
    for order, column in enumerate(columns):
        bbox = column.get("bbox") if isinstance(column, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        column_centers.append((order, (float(bbox[0]) + float(bbox[2])) / 2))
    if len(column_centers) < 2:
        return records

    safe_scale = max(scale, 1e-6)

    def sort_key(record: dict):
        bbox = record.get("bbox") or [0, 0, 0, 0]
        pdf_center_x = ((float(bbox[0]) + float(bbox[2])) / 2) / safe_scale
        column_order = min(column_centers, key=lambda item: abs(pdf_center_x - item[1]))[0]
        return (column_order, float(bbox[1]), float(bbox[0]))

    sorted_records = sorted(records, key=sort_key)
    return [
        {
            **record,
            "index": index,
        }
        for index, record in enumerate(sorted_records, start=1)
    ]


def _classify_block_type_by_regions(
    pdf_bbox: list[float],
    type_regions: list[dict] | None,
) -> tuple[str, dict | None]:
    if not type_regions:
        return "text", None
    center_x = (pdf_bbox[0] + pdf_bbox[2]) / 2
    center_y = (pdf_bbox[1] + pdf_bbox[3]) / 2
    best_region = None
    best_area = None
    for region in type_regions:
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not (float(bbox[0]) <= center_x <= float(bbox[2]) and float(bbox[1]) <= center_y <= float(bbox[3])):
            continue
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        if best_area is None or area < best_area:
            best_region = region
            best_area = area
    if best_region is None:
        return "text", None
    block_type = str(best_region.get("type") or "text")
    return block_type, {
        "type": block_type,
        "bbox": [round(float(value), 3) for value in best_region.get("bbox", [])],
        "source": "vlm_layout",
    }


def _join_child_text(children: list[dict]) -> str:
    if not children:
        return ""
    children = sorted(children, key=lambda item: (round(float(item["bbox"][1]) / 12) * 12, float(item["bbox"][0])))
    lines: list[list[dict]] = []
    for child in children:
        if not lines:
            lines.append([child])
            continue
        prev = lines[-1][-1]
        prev_h = max(1.0, prev["bbox"][3] - prev["bbox"][1])
        if abs(child["bbox"][1] - prev["bbox"][1]) <= prev_h * 0.6:
            lines[-1].append(child)
        else:
            lines.append([child])

    line_texts = []
    for line in lines:
        line.sort(key=lambda item: item["bbox"][0])
        text = "".join(str(item.get("content") or "").strip() for item in line).strip()
        if text:
            line_texts.append(text)
    return "\n".join(line_texts)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
