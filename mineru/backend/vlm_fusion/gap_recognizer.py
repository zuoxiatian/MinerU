# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re

from PIL import Image

from mineru.backend.vlm_fusion.block_builder import block_pdf_bbox, is_textual_type
from mineru.backend.vlm_fusion.config import FusionConfig
from mineru.backend.vlm_fusion.native_text import detect_native_gaps_for_block
from mineru.backend.vlm_fusion.schemas import NativeGap, NativeSpan


_DROP_GAP_VALUES = {
    "",
    "[Non-Text]",
    "Non-Text",
    "\u65e0",
    "\u65e0\u6587\u672c",
    "\u6ca1\u6709\u6587\u5b57",
    "\u6ca1\u6709\u6587\u672c",
}
_EDGE_PUNCTUATION = (
    " \t\r\n,"
    "\uff0c.\u3002!\uff01?\uff1f:\uff1a;\uff1b\u3001"
    "'\"\u201c\u201d\u2018\u2019()\uff08\uff09[]\u3010\u3011{}"
)


def detect_native_gaps(
    vlm_blocks: list[dict],
    native_spans: list[NativeSpan],
    page_width: int,
    page_height: int,
    config: FusionConfig,
) -> list[NativeGap]:
    if not config.native_gap_enable:
        return []
    gaps: list[NativeGap] = []
    for block_index, block in enumerate(vlm_blocks):
        block_type = block.get("type", "text")
        if not is_textual_type(block_type):
            continue
        effective_index = int(block.get("index", block_index + 1))
        block_bbox = block_pdf_bbox(block, page_width, page_height)
        gaps.extend(
            detect_native_gaps_for_block(
                effective_index,
                block_bbox,
                native_spans,
                config.native_overlap_threshold,
                width_ratio=config.native_gap_width_ratio,
                min_width=config.native_gap_min_width,
            )
        )
    return gaps


def crop_gap_image(
    page_image: Image.Image,
    gap: NativeGap,
    scale: float,
    config: FusionConfig,
) -> Image.Image | None:
    x0, y0, x1, y1 = gap.bbox
    line_height = max(1.0, y1 - y0)
    padding = line_height * config.native_gap_crop_padding_ratio
    crop_box = (
        max(0, int((x0 - padding * 0.35) * scale)),
        max(0, int((y0 - padding) * scale)),
        min(page_image.width, int((x1 + padding * 0.35) * scale)),
        min(page_image.height, int((y1 + padding) * scale)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None
    return page_image.crop(crop_box)


def normalize_gap_text(text: str | None, max_chars: int) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\s+", "", str(text))
    normalized = normalized.strip(_EDGE_PUNCTUATION)
    if normalized in _DROP_GAP_VALUES:
        return ""
    if len(normalized) > max_chars:
        return ""
    visible_count = sum(1 for char in normalized if not char.isspace())
    if visible_count == 0:
        return ""
    return normalized


def recognize_native_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: FusionConfig,
) -> list[NativeGap]:
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = crop_gap_image(page_image, gap, scale, config)
        if crop is None:
            continue
        gap_images.append(crop)
        gap_refs.append(gap)
    if not gap_images:
        return gaps

    try:
        outputs = predictor.batch_content_extract(gap_images, types=["text"] * len(gap_images))
    finally:
        for image in gap_images:
            image.close()

    for gap, output in zip(gap_refs, outputs):
        gap.content = normalize_gap_text(
            None if output is None else str(output),
            config.native_gap_max_chars,
        )
    return gaps


async def aio_recognize_native_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: FusionConfig,
) -> list[NativeGap]:
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = crop_gap_image(page_image, gap, scale, config)
        if crop is None:
            continue
        gap_images.append(crop)
        gap_refs.append(gap)
    if not gap_images:
        return gaps

    try:
        outputs = await predictor.aio_batch_content_extract(
            gap_images,
            types=["text"] * len(gap_images),
        )
    finally:
        for image in gap_images:
            image.close()

    for gap, output in zip(gap_refs, outputs):
        gap.content = normalize_gap_text(
            None if output is None else str(output),
            config.native_gap_max_chars,
        )
    return gaps
