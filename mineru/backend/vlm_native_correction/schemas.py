# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field

from mineru.backend.vlm_fusion.schemas import NativeSpan, VisualTextCandidate


@dataclass
class PageCorrectionContext:
    page_index: int
    page_width: int
    page_height: int
    layout_blocks: list[dict]
    vlm_blocks: list[dict]
    native_spans: list[NativeSpan]
    visual_candidates: list[VisualTextCandidate] = field(default_factory=list)


@dataclass
class CorrectionMetrics:
    page_idx: int
    native_span_count: int = 0
    vlm_content_block_count: int = 0
    corrected_block_count: int = 0
    corrected_char_count: int = 0
    native_inserted_char_count: int = 0
    skipped_conflict_count: int = 0
    vlm_fallback_count: int = 0

    def to_dict(self) -> dict:
        return {
            "page_idx": self.page_idx,
            "native_span_count": self.native_span_count,
            "vlm_content_block_count": self.vlm_content_block_count,
            "corrected_block_count": self.corrected_block_count,
            "corrected_char_count": self.corrected_char_count,
            "native_inserted_char_count": self.native_inserted_char_count,
            "skipped_conflict_count": self.skipped_conflict_count,
            "vlm_fallback_count": self.vlm_fallback_count,
        }

