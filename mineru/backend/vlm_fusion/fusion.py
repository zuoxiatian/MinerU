# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_fusion.bbox import coverage_by_boxes
from mineru.backend.vlm_fusion.block_builder import (
    block_pdf_bbox,
    build_native_text_block,
    build_supplement_block,
    build_vlm_fallback_block,
    copy_model_block,
    is_structural_type,
    is_textual_type,
)
from mineru.backend.vlm_fusion.config import FusionConfig
from mineru.backend.vlm_fusion.native_text import (
    build_native_match,
    join_native_spans,
    join_native_spans_with_gaps,
    text_similarity,
)
from mineru.backend.vlm_fusion.schemas import FusionMetrics, PageFusionContext


def _row_order_key(pdf_bbox):
    return (round(float(pdf_bbox[1]) / 12) * 12, float(pdf_bbox[0]))


def _detect_reading_columns(blocks: list[dict], page_width: int, page_height: int):
    items = []
    for block in blocks:
        bbox = block_pdf_bbox(block, page_width, page_height)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0 or height <= 0:
            continue
        if width >= page_width * 0.7:
            continue
        items.append((block, bbox, (bbox[0] + bbox[2]) / 2))

    if len(items) < 4:
        return []

    items.sort(key=lambda item: item[2])
    gaps = [
        (items[idx + 1][2] - items[idx][2], idx)
        for idx in range(len(items) - 1)
    ]
    if not gaps:
        return []

    largest_gap, split_idx = max(gaps, key=lambda item: item[0])
    if largest_gap < page_width * 0.1:
        return []

    left_items = items[: split_idx + 1]
    right_items = items[split_idx + 1 :]
    if len(left_items) < 2 or len(right_items) < 2:
        return []

    return [
        {
            "x0": min(item[1][0] for item in group),
            "x1": max(item[1][2] for item in group),
            "center": sum(item[2] for item in group) / len(group),
        }
        for group in (left_items, right_items)
    ]


def _sort_fused_blocks(blocks: list[dict], page_width: int, page_height: int):
    columns = _detect_reading_columns(blocks, page_width, page_height)
    if not columns:
        blocks.sort(key=lambda block: _row_order_key(block_pdf_bbox(block, page_width, page_height)))
        return

    columns.sort(key=lambda column: column["center"])

    def column_order_key(block):
        bbox = block_pdf_bbox(block, page_width, page_height)
        center_x = (bbox[0] + bbox[2]) / 2
        block_width = bbox[2] - bbox[0]
        if block_width >= page_width * 0.7 and bbox[1] < page_height * 0.2:
            column_index = -1
        else:
            column_index = min(
                range(len(columns)),
                key=lambda idx: abs(center_x - columns[idx]["center"]),
            )
        row_y, row_x = _row_order_key(bbox)
        return (column_index, row_y, row_x)

    blocks.sort(key=column_order_key)


def _next_index(blocks: list[dict]) -> int:
    indexes = [int(block.get("index", idx + 1)) for idx, block in enumerate(blocks)]
    return (max(indexes) + 1) if indexes else 1


def _is_duplicate_visual_candidate(candidate, locked_spans, config: FusionConfig):
    coverage = coverage_by_boxes(candidate.bbox, [span.bbox for span in locked_spans])
    native_text = join_native_spans(locked_spans)
    similarity = text_similarity(candidate.content, native_text)
    return coverage >= config.duplicate_threshold and similarity >= 0.55, coverage, similarity


def fuse_page(context: PageFusionContext, config: FusionConfig) -> tuple[list[dict], FusionMetrics]:
    metrics = FusionMetrics(
        page_idx=context.page_index,
        layout_block_count=len(context.layout_blocks),
        native_span_count=len(context.native_spans),
        vlm_content_block_count=len(context.vlm_blocks),
        visual_text_candidate_count=len(context.visual_candidates),
        native_gap_count=len(context.native_gaps),
        native_gap_filled_count=sum(1 for gap in context.native_gaps if gap.content.strip()),
    )

    fused_blocks: list[dict] = []
    locked_native_spans = []
    emitted_vlm_indices = set()

    for index, block in enumerate(context.vlm_blocks):
        block.setdefault("index", index + 1)
        block_type = block.get("type", "text")
        if is_textual_type(block_type):
            block_bbox = block_pdf_bbox(block, context.page_width, context.page_height)
            native_match = build_native_match(
                block_bbox,
                context.native_spans,
                config.native_overlap_threshold,
            )
            vlm_content = block.get("content") or ""
            if native_match.reliable:
                block_gaps = [
                    gap
                    for gap in context.native_gaps
                    if gap.block_index == block.get("index", index + 1) and gap.content.strip()
                ]
                native_content = (
                    join_native_spans_with_gaps(native_match.spans, block_gaps)
                    if block_gaps
                    else native_match.content
                )
                for span in native_match.spans:
                    span.consumed = True
                locked_native_spans.extend(native_match.spans)
                emitted_vlm_indices.add(block.get("index", index + 1))
                metrics.native_locked_count += 1
                if vlm_content and text_similarity(native_match.content, vlm_content) < 0.55:
                    metrics.source_conflict_count += 1
                fused_blocks.append(
                    build_native_text_block(
                        block,
                        native_content,
                        {
                            "decision": "native_locked",
                            "native_quality": native_match.quality,
                            "vlm_text": vlm_content,
                            "native_gap_count": len(block_gaps),
                        } if config.debug else None,
                    )
                )
            else:
                if vlm_content:
                    for span in native_match.spans:
                        span.consumed = True
                emitted_vlm_indices.add(block.get("index", index + 1))
                fused_blocks.append(
                    build_vlm_fallback_block(
                        block,
                        {
                            "decision": "vlm_fallback",
                            "native_quality": native_match.quality,
                            "native_text": native_match.content,
                        } if config.debug else None,
                    )
                )
        elif is_structural_type(block_type):
            fused_blocks.append(copy_model_block(block))
        else:
            fused_blocks.append(copy_model_block(block))

    metrics.native_consumed_count = sum(1 for span in context.native_spans if span.consumed)

    if config.visual_supplement:
        next_index = _next_index(fused_blocks)
        for candidate in context.visual_candidates:
            if len(candidate.content.strip()) < config.min_visual_text_chars:
                continue
            if candidate.raw.get("index") in emitted_vlm_indices:
                metrics.visual_duplicate_skipped_count += 1
                continue
            duplicate, coverage, similarity = _is_duplicate_visual_candidate(
                candidate,
                locked_native_spans,
                config,
            )
            if duplicate:
                metrics.visual_duplicate_skipped_count += 1
                continue
            fused_blocks.append(
                build_supplement_block(
                    candidate,
                    context.page_width,
                    context.page_height,
                    next_index,
                    {
                        "decision": "visual_supplement",
                        "overlap_with_locked_native": coverage,
                        "native_similarity": similarity,
                    } if config.debug else None,
                )
            )
            next_index += 1
            metrics.visual_supplement_count += 1

    next_index = _next_index(fused_blocks)
    structural_bboxes = [
        block_pdf_bbox(block, context.page_width, context.page_height)
        for block in fused_blocks
        if is_structural_type(block.get("type", ""))
    ]
    for span in context.native_spans:
        if span.consumed or not span.content.strip():
            continue
        if coverage_by_boxes(span.bbox, structural_bboxes) >= 0.5:
            span.consumed = True
            continue
        fused_blocks.append(
            {
                "type": "text",
                "bbox": [
                    round(span.bbox[0] / context.page_width, 6),
                    round(span.bbox[1] / context.page_height, 6),
                    round(span.bbox[2] / context.page_width, 6),
                    round(span.bbox[3] / context.page_height, 6),
                ],
                "angle": 0,
                "content": span.content,
                "index": next_index,
                "source": "pdf_native_recovered",
            }
        )
        span.consumed = True
        next_index += 1
        metrics.native_recovered_count += 1

    _sort_fused_blocks(fused_blocks, context.page_width, context.page_height)
    for index, block in enumerate(fused_blocks):
        block["index"] = index + 1
    return fused_blocks, metrics
