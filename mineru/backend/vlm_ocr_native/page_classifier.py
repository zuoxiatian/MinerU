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
    image_block_count: int
    text_block_count: int
    reason: str


def classify_layout_blocks(
    layout_blocks: list[Any],
    *,
    page_width: int | float,
    page_height: int | float,
    large_image_ratio_threshold: float = 0.75,
) -> PageClassifyResult:
    image_block_count = 0
    text_block_count = 0
    largest_image_ratio = 0.0

    for raw_block in layout_blocks or []:
        block = _block_to_dict(raw_block)
        block_type = str(block.get("type") or "").lower()
        if block_type in IMAGE_TYPES:
            image_block_count += 1
            largest_image_ratio = max(
                largest_image_ratio,
                _bbox_area_ratio(block.get("bbox"), page_width, page_height),
            )
        elif block_type in TEXT_TYPES:
            text_block_count += 1

    if largest_image_ratio >= large_image_ratio_threshold:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="largest_image_block",
        )
    if image_block_count > text_block_count:
        return PageClassifyResult(
            is_comic_like=True,
            largest_image_block_area_ratio=round(largest_image_ratio, 6),
            image_block_count=image_block_count,
            text_block_count=text_block_count,
            reason="image_block_count",
        )
    return PageClassifyResult(
        is_comic_like=False,
        largest_image_block_area_ratio=round(largest_image_ratio, 6),
        image_block_count=image_block_count,
        text_block_count=text_block_count,
        reason="normal",
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
