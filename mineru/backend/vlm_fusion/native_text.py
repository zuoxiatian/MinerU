# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from difflib import SequenceMatcher

from loguru import logger

from mineru.backend.vlm_fusion.bbox import center_in_bbox, intersection_area, reading_order_key
from mineru.backend.vlm_fusion.schemas import BBox, NativeGap, NativeMatch, NativeSpan
from mineru.utils.pdf_text_tool import get_lines_from_chars, get_page_chars


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _coerce_bbox(value) -> BBox | None:
    if value is None:
        return None
    if hasattr(value, "bbox"):
        value = value.bbox
    if isinstance(value, dict):
        value = value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _line_text(line: dict) -> str:
    text = line.get("text")
    if isinstance(text, str):
        return text
    spans = line.get("spans") or []
    parts = []
    for span in spans:
        span_text = span.get("text") or span.get("content")
        if isinstance(span_text, str):
            parts.append(span_text)
    return "".join(parts)


def extract_native_spans(pdf_page) -> list[NativeSpan]:
    try:
        page_chars = get_page_chars(pdf_page)
        lines = get_lines_from_chars(page_chars["chars"])
    except Exception as exc:
        logger.debug(f"Failed to extract native PDF text: {exc}")
        return []

    native_spans: list[NativeSpan] = []
    next_uid = 0
    for line in lines:
        line_bbox = _coerce_bbox(line.get("bbox"))
        spans = line.get("spans") or []
        added_span = False
        for span in spans:
            bbox = _coerce_bbox(span.get("bbox")) or line_bbox
            content = span.get("text") or span.get("content") or ""
            if bbox and str(content).strip():
                native_spans.append(NativeSpan(bbox=bbox, content=str(content), uid=next_uid))
                next_uid += 1
                added_span = True
        if not added_span and line_bbox:
            content = _line_text(line)
            if content.strip():
                native_spans.append(NativeSpan(bbox=line_bbox, content=content, uid=next_uid))
                next_uid += 1
    return native_spans


def _span_overlap_ratio(span_bbox: BBox, block_bbox: BBox) -> float:
    span_area = max(1.0, (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1]))
    return intersection_area(span_bbox, block_bbox) / span_area


def match_native_to_bbox(
    block_bbox: BBox,
    native_spans: list[NativeSpan],
    overlap_threshold: float,
) -> list[NativeSpan]:
    matched = []
    for span in native_spans:
        if span.consumed:
            continue
        if (
            _span_overlap_ratio(span.bbox, block_bbox) >= overlap_threshold
            or center_in_bbox(span.bbox, block_bbox)
        ):
            matched.append(span)
    matched.sort(key=lambda span: reading_order_key({"bbox": span.bbox}))
    return matched


def join_native_spans(spans: list[NativeSpan]) -> str:
    if not spans:
        return ""
    lines: list[list[NativeSpan]] = []
    for span in spans:
        if not lines:
            lines.append([span])
            continue
        prev = lines[-1][-1]
        prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
        if abs(span.bbox[1] - prev.bbox[1]) <= prev_h * 0.6:
            lines[-1].append(span)
        else:
            lines.append([span])

    line_texts = []
    for line in lines:
        line.sort(key=lambda span: span.bbox[0])
        line_texts.append("".join(span.content for span in line).strip())
    return "\n".join(text for text in line_texts if text)


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


def join_native_spans_with_gaps(spans: list[NativeSpan], gaps: list[NativeGap]) -> str:
    if not spans:
        return ""
    gaps_by_left_uid: dict[int, list[NativeGap]] = {}
    for gap in gaps:
        if not gap.content.strip():
            continue
        gaps_by_left_uid.setdefault(gap.left_span_uid, []).append(gap)
    for gap_list in gaps_by_left_uid.values():
        gap_list.sort(key=lambda gap: gap.bbox[0])

    line_texts = []
    for line in group_native_spans_to_lines(spans):
        parts = []
        for span in line:
            parts.append(span.content)
            for gap in gaps_by_left_uid.get(span.uid, []):
                parts.append(gap.content)
        line_text = "".join(parts).strip()
        if line_text:
            line_texts.append(line_text)
    return "\n".join(line_texts)


def score_native_text_quality(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    control_count = len(_CONTROL_RE.findall(stripped))
    replacement_count = stripped.count("\ufffd")
    visible_count = sum(1 for char in stripped if not char.isspace())
    if visible_count == 0:
        return 0.0
    bad_ratio = (control_count + replacement_count) / max(1, len(stripped))
    quality = 1.0 - bad_ratio
    if visible_count <= 1:
        quality *= 0.8
    return max(0.0, min(1.0, quality))


def build_native_match(
    block_bbox: BBox,
    native_spans: list[NativeSpan],
    overlap_threshold: float,
) -> NativeMatch:
    spans = match_native_to_bbox(block_bbox, native_spans, overlap_threshold)
    content = join_native_spans(spans)
    quality = score_native_text_quality(content)
    return NativeMatch(
        spans=spans,
        content=content,
        quality=quality,
        reliable=quality >= 0.85 and bool(content.strip()),
    )


def detect_native_gaps_for_block(
    block_index: int,
    block_bbox: BBox,
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
        median_char_width = char_widths[len(char_widths) // 2]
        gap_threshold = max(min_width, median_char_width * width_ratio)

        for left, right in zip(line, line[1:]):
            gap_width = right.bbox[0] - left.bbox[2]
            if gap_width < gap_threshold:
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
                )
            )
    return gaps


def text_similarity(left: str, right: str) -> float:
    left_norm = re.sub(r"\s+", "", left or "")
    right_norm = re.sub(r"\s+", "", right or "")
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()
