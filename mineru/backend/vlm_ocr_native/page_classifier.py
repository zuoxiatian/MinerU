# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IMAGE_TYPES = {"image", "image_block", "chart"}
TEXT_TYPES = {
    "text",
    "title",
    "ref_text",
    "phonetic",
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "list",
    "list_item",
    "image_caption",
    "table_caption",
    "code_caption",
    "image_footnote",
    "table_footnote",
}


@dataclass(frozen=True)
class PageClassifyResult:
    is_comic_like: bool
    largest_image_block_area_ratio: float
    total_image_block_area_ratio: float
    max_column_image_block_area_ratio: float
    column_image_block_area_ratios: list[dict]
    image_block_count: int
    text_block_count: int
    reason: str


@dataclass(frozen=True)
class PageLayoutResult:
    flow: str
    source: str
    reading_order: str
    columns: list[dict]
    reason: str


def classify_layout_blocks(
    layout_blocks: list[Any],
    *,
    page_width: int | float,
    page_height: int | float,
    large_image_ratio_threshold: float = 0.75,
    total_image_ratio_threshold: float = 0.65,
    column_image_ratio_threshold: float = 0.60,
) -> PageClassifyResult:
    image_block_count = 0
    text_block_count = 0
    largest_image_ratio = 0.0
    total_image_ratio = 0.0

    for raw_block in layout_blocks or []:
        block = _block_to_dict(raw_block)
        block_type = str(block.get("type") or "").lower()
        if block_type in IMAGE_TYPES:
            image_block_count += 1
            image_ratio = _bbox_area_ratio(block.get("bbox"), page_width, page_height)
            total_image_ratio += image_ratio
            largest_image_ratio = max(
                largest_image_ratio,
                image_ratio,
            )
        elif block_type in TEXT_TYPES:
            text_block_count += 1

    total_image_ratio = min(1.0, total_image_ratio)
    page_layout = analyze_layout_flow(
        layout_blocks,
        page_width=page_width,
        page_height=page_height,
    )
    column_image_ratios = (
        _column_image_area_ratios(layout_blocks, page_layout, page_width, page_height)
        if page_layout.flow == "double_column"
        else []
    )
    max_column_image_ratio = max(
        [item["image_block_area_ratio"] for item in column_image_ratios],
        default=0.0,
    )

    if largest_image_ratio >= large_image_ratio_threshold:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            total_image_block_area_ratio=round(total_image_ratio, 6),
            max_column_image_block_area_ratio=round(max_column_image_ratio, 6),
            column_image_block_area_ratios=column_image_ratios,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="largest_image_block",
        )
    if image_block_count > text_block_count:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            total_image_block_area_ratio=round(total_image_ratio, 6),
            max_column_image_block_area_ratio=round(max_column_image_ratio, 6),
            column_image_block_area_ratios=column_image_ratios,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="image_block_count",
        )
    if total_image_ratio >= total_image_ratio_threshold:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            total_image_block_area_ratio=round(total_image_ratio, 6),
            max_column_image_block_area_ratio=round(max_column_image_ratio, 6),
            column_image_block_area_ratios=column_image_ratios,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="total_image_block_area",
        )
    if max_column_image_ratio >= column_image_ratio_threshold:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            total_image_block_area_ratio=round(total_image_ratio, 6),
            max_column_image_block_area_ratio=round(max_column_image_ratio, 6),
            column_image_block_area_ratios=column_image_ratios,
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="column_image_block_area",
        )
    return PageClassifyResult(
        is_comic_like=False,
        largest_image_block_area_ratio=round(largest_image_ratio, 6),
        total_image_block_area_ratio=round(total_image_ratio, 6),
        max_column_image_block_area_ratio=round(max_column_image_ratio, 6),
        column_image_block_area_ratios=column_image_ratios,
        image_block_count=image_block_count,
        text_block_count=text_block_count,
        reason="normal",
    )


def analyze_layout_flow(
    layout_blocks: list[Any],
    *,
    page_width: int | float,
    page_height: int | float,
    reading_order: str = "left_to_right",
) -> PageLayoutResult:
    candidates = _layout_flow_candidates(layout_blocks, page_width, page_height)
    if len(candidates) < 4:
        return PageLayoutResult(
            flow="single_column",
            source="vlm_layout",
            reading_order=reading_order,
            columns=[],
            reason="not_enough_vlm_blocks",
        )

    left = [item for item in candidates if item["center_x"] <= float(page_width) * 0.5]
    right = [item for item in candidates if item["center_x"] > float(page_width) * 0.5]
    if len(left) < 2 or len(right) < 2:
        return PageLayoutResult(
            flow="single_column",
            source="vlm_layout",
            reading_order=reading_order,
            columns=[],
            reason="unbalanced_vlm_blocks",
        )

    left_column = _column_record(1, left, page_width, page_height)
    right_column = _column_record(2, right, page_width, page_height)
    center_gap = right_column["center_x"] - left_column["center_x"]
    min_center_gap = float(page_width) * 0.25
    y_overlap = _range_overlap_ratio(
        [left_column["bbox"][1], left_column["bbox"][3]],
        [right_column["bbox"][1], right_column["bbox"][3]],
    )
    if center_gap < min_center_gap or y_overlap < 0.25:
        return PageLayoutResult(
            flow="single_column",
            source="vlm_layout",
            reading_order=reading_order,
            columns=[],
            reason="weak_column_geometry",
        )

    columns = [left_column, right_column]
    if reading_order == "right_to_left":
        columns = [right_column, left_column]
        columns = [dict(column, index=index) for index, column in enumerate(columns, start=1)]

    return PageLayoutResult(
        flow="double_column",
        source="vlm_layout",
        reading_order=reading_order,
        columns=columns,
        reason="vlm_blocks_split_by_page_center",
    )


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    try:
        return dict(block)
    except Exception:
        return {
            "type": getattr(block, "type", None),
            "bbox": getattr(block, "bbox", None),
        }


def _bbox_area_ratio(
    bbox,
    page_width: int | float,
    page_height: int | float,
) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return 0.0

    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        page_area = 1.0
    else:
        page_area = max(1.0, float(page_width) * float(page_height))
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(1.0, area / page_area)


def _layout_flow_candidates(
    layout_blocks: list[Any],
    page_width: int | float,
    page_height: int | float,
) -> list[dict]:
    candidates = []
    for raw_block in layout_blocks or []:
        block = _block_to_dict(raw_block)
        block_type = str(block.get("type") or "").lower()
        bbox = _to_pdf_bbox(block.get("bbox"), page_width, page_height)
        if bbox is None:
            continue
        area_ratio = _bbox_area_ratio(block.get("bbox"), page_width, page_height)
        if block_type in IMAGE_TYPES and area_ratio >= 0.75:
            continue
        if block_type not in TEXT_TYPES and block_type not in IMAGE_TYPES:
            continue
        candidates.append(
            {
                "type": block_type,
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "center_y": (bbox[1] + bbox[3]) / 2,
            }
        )
    return candidates


def _to_pdf_bbox(
    bbox,
    page_width: int | float,
    page_height: int | float,
) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        return [
            x0 * float(page_width),
            y0 * float(page_height),
            x1 * float(page_width),
            y1 * float(page_height),
        ]
    return [x0, y0, x1, y1]


def _column_record(
    index: int,
    blocks: list[dict],
    page_width: int | float,
    page_height: int | float,
) -> dict:
    x0 = min(item["bbox"][0] for item in blocks)
    y0 = min(item["bbox"][1] for item in blocks)
    x1 = max(item["bbox"][2] for item in blocks)
    y1 = max(item["bbox"][3] for item in blocks)
    bbox = [x0, y0, x1, y1]
    return {
        "index": index,
        "bbox": bbox,
        "normalized_bbox": [
            round(x0 / max(1.0, float(page_width)), 6),
            round(y0 / max(1.0, float(page_height)), 6),
            round(x1 / max(1.0, float(page_width)), 6),
            round(y1 / max(1.0, float(page_height)), 6),
        ],
        "center_x": (x0 + x1) / 2,
        "count": len(blocks),
    }


def _range_overlap_ratio(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    smaller = max(1.0, min(left[1] - left[0], right[1] - right[0]))
    return overlap / smaller


def _column_image_area_ratios(
    layout_blocks: list[Any],
    page_layout: PageLayoutResult,
    page_width: int | float,
    page_height: int | float,
) -> list[dict]:
    ratios = []
    image_blocks = []
    for raw_block in layout_blocks or []:
        block = _block_to_dict(raw_block)
        block_type = str(block.get("type") or "").lower()
        if block_type not in IMAGE_TYPES:
            continue
        bbox = _to_pdf_bbox(block.get("bbox"), page_width, page_height)
        if bbox is None:
            continue
        image_blocks.append(
            {
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
                "area": max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]),
            }
        )

    for column in page_layout.columns:
        bbox = column.get("bbox") or [0, 0, 0, 0]
        column_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        image_area = 0.0
        image_count = 0
        for image_block in image_blocks:
            if bbox[0] <= image_block["center_x"] <= bbox[2]:
                image_area += image_block["area"]
                image_count += 1
        ratios.append(
            {
                "column_index": column.get("index"),
                "image_block_count": image_count,
                "image_block_area_ratio": round(min(1.0, image_area / column_area), 6),
            }
        )
    return ratios
