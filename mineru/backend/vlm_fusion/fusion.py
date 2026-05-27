# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_fusion.bbox import coverage_by_boxes, reading_order_key
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
from mineru.backend.vlm_fusion.native_text import build_native_match, join_native_spans, text_similarity
from mineru.backend.vlm_fusion.schemas import FusionMetrics, PageFusionContext


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
                        native_match.content,
                        {
                            "decision": "native_locked",
                            "native_quality": native_match.quality,
                            "vlm_text": vlm_content,
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

    fused_blocks.sort(key=reading_order_key)
    for index, block in enumerate(fused_blocks):
        block["index"] = index + 1
    return fused_blocks, metrics
