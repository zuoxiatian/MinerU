# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_fusion.block_builder import is_textual_type
from mineru.backend.vlm_fusion.schemas import VisualTextCandidate


def collect_visual_text_candidates(
    vlm_blocks: list[dict],
    page_width: int,
    page_height: int,
) -> list[VisualTextCandidate]:
    candidates: list[VisualTextCandidate] = []
    for block in vlm_blocks:
        block_type = block.get("type", "text")
        content = block.get("content")
        bbox = block.get("bbox")
        if not is_textual_type(block_type):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        candidates.append(
            VisualTextCandidate(
                bbox=[
                    float(bbox[0]) * page_width,
                    float(bbox[1]) * page_height,
                    float(bbox[2]) * page_width,
                    float(bbox[3]) * page_height,
                ],
                content=content,
                block_type=block_type,
                source="vlm",
                raw=dict(block),
            )
        )
    return candidates

