# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re

from PIL import Image

from mineru.backend.vlm_ocr_native.schemas import NativeGap, NativeSpan
from mineru.backend.vlm_ocr_native.config import NativeCorrectionConfig


TEXTUAL_TYPES = {
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


def detect_native_correction_gaps(
    vlm_blocks: list[dict],
    native_spans: list[NativeSpan],
    page_width: int,
    page_height: int,
    config: NativeCorrectionConfig,
) -> list[NativeGap]:
    if not config.native_gap_enable:
        return []
    gaps: list[NativeGap] = []
    for block_index, block in enumerate(vlm_blocks):
        if not _should_detect_gap_for_block(block):
            continue
        effective_index = int(block.get("index", block_index + 1))
        block_bbox = block_pdf_bbox(block, page_width, page_height)
        gaps.extend(
            detect_native_gaps_for_block(
                effective_index,
                block_bbox,
                native_spans,
                overlap_threshold=0.45,
                width_ratio=config.native_gap_width_ratio,
                min_width=config.native_gap_min_width,
            )
        )
    return gaps


def recognize_native_correction_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: NativeCorrectionConfig,
) -> list[NativeGap]:
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = _crop_gap_image(page_image, gap, scale, config)
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
        gap.content = _normalize_chinese_gap_text(
            None if output is None else str(output),
            _gap_max_chars(gap, config.native_gap_max_chars),
        )
        if gap.content:
            gap.source = "vlm_ocr_native_gap"
    return gaps


async def aio_recognize_native_correction_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: NativeCorrectionConfig,
) -> list[NativeGap]:
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = _crop_gap_image(page_image, gap, scale, config)
        if crop is None:
            continue
        gap_images.append(crop)
        gap_refs.append(gap)
    if not gap_images:
        return gaps

    try:
        outputs = await predictor.aio_batch_content_extract(gap_images, types=["text"] * len(gap_images))
    finally:
        for image in gap_images:
            image.close()

    for gap, output in zip(gap_refs, outputs):
        gap.content = _normalize_chinese_gap_text(
            None if output is None else str(output),
            _gap_max_chars(gap, config.native_gap_max_chars),
        )
        if gap.content:
            gap.source = "vlm_ocr_native_gap"
    return gaps


def _should_detect_gap_for_block(block: dict) -> bool:
    block_type = block.get("type", "text")
    if is_textual_type(block_type):
        return True
    return block_type == "image" and block.get("sub_type") == "text_image" and bool((block.get("content") or "").strip())


def detect_native_gaps_for_block(
    block_index: int,
    block_bbox: list[float],
    native_spans: list[NativeSpan],
    overlap_threshold: float,
    *,
    width_ratio: float,
    min_width: float,
) -> list[NativeGap]:
    matched = match_native_to_bbox(block_bbox, native_spans, overlap_threshold)
    gaps: list[NativeGap] = []
    for line in group_native_spans_to_lines(matched):
        if len(line) < 2:
            continue
        char_widths = [
            (span.bbox[2] - span.bbox[0]) / max(1, len(span.content.strip()))
            for span in line
            if span.content.strip()
        ]
        if not char_widths:
            continue
        char_widths.sort()
        median_char_width = char_widths[(len(char_widths) - 1) // 2]
        gap_threshold = max(min_width, median_char_width * width_ratio)

        for left, right in zip(line, line[1:]):
            gap_width = right.bbox[0] - left.bbox[2]
            left_text = left.content.strip()
            right_text = right.content.strip()
            left_char_width = (left.bbox[2] - left.bbox[0]) / max(1, len(left_text))
            right_char_width = (right.bbox[2] - right.bbox[0]) / max(1, len(right_text))
            local_char_width = max(1.0, min(left_char_width, right_char_width, median_char_width))
            pair_gap_threshold = gap_threshold
            if min(len(left_text), len(right_text)) <= 2:
                short_span_threshold = max(min_width, min(left_char_width, right_char_width) * 2.0)
                pair_gap_threshold = min(pair_gap_threshold, short_span_threshold)
            if gap_width < pair_gap_threshold:
                continue
            y0 = max(min(left.bbox[1], right.bbox[1]), block_bbox[1])
            y1 = min(max(left.bbox[3], right.bbox[3]), block_bbox[3])
            if y1 <= y0:
                continue
            gaps.append(
                NativeGap(
                    bbox=[
                        max(left.bbox[2], block_bbox[0]),
                        y0,
                        min(right.bbox[0], block_bbox[2]),
                        y1,
                    ],
                    block_index=block_index,
                    left_span_uid=left.uid,
                    right_span_uid=right.uid,
                    max_chars=max(1, round(gap_width / local_char_width)),
                )
            )
    return gaps


def match_native_to_bbox(
    block_bbox: list[float],
    native_spans: list[NativeSpan],
    overlap_threshold: float,
) -> list[NativeSpan]:
    matched = []
    for span in native_spans:
        if span.consumed:
            continue
        if _span_overlap_ratio(span.bbox, block_bbox) >= overlap_threshold or center_in_bbox(span.bbox, block_bbox):
            matched.append(span)
    matched.sort(key=lambda span: reading_order_key({"bbox": span.bbox}))
    return matched


def group_native_spans_to_lines(spans: list[NativeSpan]) -> list[list[NativeSpan]]:
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda span: reading_order_key({"bbox": span.bbox}))
    lines: list[list[NativeSpan]] = []
    for span in sorted_spans:
        if not lines:
            lines.append([span])
            continue
        prev = lines[-1][-1]
        prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
        if abs(span.bbox[1] - prev.bbox[1]) <= prev_h * 0.6:
            lines[-1].append(span)
        else:
            lines.append([span])
    for line in lines:
        line.sort(key=lambda span: span.bbox[0])
    return lines


def block_pdf_bbox(block: dict, page_width: int, page_height: int) -> list[float]:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return [
        float(bbox[0]) * page_width,
        float(bbox[1]) * page_height,
        float(bbox[2]) * page_width,
        float(bbox[3]) * page_height,
    ]


def is_textual_type(block_type: str) -> bool:
    return normalize_block_type(block_type) in TEXTUAL_TYPES


def normalize_block_type(block_type: str) -> str:
    if block_type == "list_item":
        return "list"
    return block_type


def _span_overlap_ratio(span_bbox: list[float], block_bbox: list[float]) -> float:
    span_area = max(1.0, (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1]))
    return intersection_area(span_bbox, block_bbox) / span_area


def intersection_area(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def center_in_bbox(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def reading_order_key(block: dict):
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return (round(float(bbox[1]) / 12) * 12, float(bbox[0]))


def _crop_gap_image(
    page_image: Image.Image,
    gap: NativeGap,
    scale: float,
    config: NativeCorrectionConfig,
) -> Image.Image | None:
    x0, y0, x1, y1 = gap.bbox
    line_height = max(1.0, y1 - y0)
    padding = line_height * config.native_gap_crop_padding_ratio
    crop_box = (
        max(0, int(x0 * scale)),
        max(0, int((y0 - padding) * scale)),
        min(page_image.width, int(x1 * scale)),
        min(page_image.height, int((y1 + padding) * scale)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None
    return page_image.crop(crop_box)


def _normalize_chinese_gap_text(text: str | None, max_chars: int) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\s+", "", str(text))
    chinese = "".join(char for char in normalized if _is_cjk(char))
    if not chinese:
        return ""
    return chinese[:max_chars]


def _gap_max_chars(gap: NativeGap, configured_max_chars: int) -> int:
    if gap.max_chars > 0:
        return max(1, min(configured_max_chars, gap.max_chars))
    return configured_max_chars


def _is_cjk(char: str) -> bool:
    return (
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )
