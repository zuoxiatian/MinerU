# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = list[float]


@dataclass
class NativeSpan:
    """PDF native text span in page coordinates."""
    bbox: BBox
    content: str
    uid: int = -1
    consumed: bool = False


@dataclass
class NativeMatch:
    """Native text matched to one model block."""
    spans: list[NativeSpan]
    content: str
    quality: float
    reliable: bool


@dataclass
class VisualTextCandidate:
    """VLM text candidate that may supplement native text."""
    bbox: BBox
    content: str
    block_type: str
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeGap:
    """Suspicious gap between adjacent PDF native spans."""
    bbox: BBox
    block_index: int
    left_span_uid: int
    right_span_uid: int
    max_chars: int = 0
    content: str = ""
    source: str = "vlm_gap"


@dataclass
class PageCorrectionContext:
    page_index: int
    page_width: int
    page_height: int
    layout_blocks: list[dict]
    vlm_blocks: list[dict]
    native_spans: list[NativeSpan]
    visual_candidates: list[VisualTextCandidate] = field(default_factory=list)
    native_gaps: list[NativeGap] = field(default_factory=list)


@dataclass
class CorrectionMetrics:
    page_idx: int
    native_span_count: int = 0
    vlm_content_block_count: int = 0
    corrected_block_count: int = 0
    corrected_char_count: int = 0
    native_inserted_char_count: int = 0
    native_gap_count: int = 0
    native_gap_filled_count: int = 0
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
            "native_gap_count": self.native_gap_count,
            "native_gap_filled_count": self.native_gap_filled_count,
            "skipped_conflict_count": self.skipped_conflict_count,
            "vlm_fallback_count": self.vlm_fallback_count,
        }
