# Copyright (c) Opendatalab. All rights reserved.
from dataclasses import dataclass, field
from typing import Any


BBox = list[float]


@dataclass
class NativeSpan:
    """PDF 原生文本片段，bbox 使用 PDF 页面坐标。"""
    bbox: BBox
    content: str
    uid: int = -1
    consumed: bool = False


@dataclass
class NativeMatch:
    """一个 VLM block 匹配到的 PDF 原生文本结果。"""
    spans: list[NativeSpan]
    content: str
    quality: float
    reliable: bool


@dataclass
class VisualTextCandidate:
    """VLM 识别到、可能需要作为补充输出的文本候选。"""
    bbox: BBox
    content: str
    block_type: str
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeGap:
    """PDF 原生 span 之间的可疑缺口。"""
    bbox: BBox
    block_index: int
    left_span_uid: int
    right_span_uid: int
    content: str = ""
    source: str = "vlm_gap"


@dataclass
class PageFusionContext:
    """单页融合所需的完整输入。"""
    page_index: int
    page_width: int
    page_height: int
    layout_blocks: list[dict]
    vlm_blocks: list[dict]
    native_spans: list[NativeSpan]
    visual_candidates: list[VisualTextCandidate]
    native_gaps: list[NativeGap] = field(default_factory=list)


@dataclass
class FusionMetrics:
    """单页融合过程中的统计指标，用于调试和质量观测。"""
    page_idx: int
    layout_block_count: int = 0
    native_span_count: int = 0
    vlm_content_block_count: int = 0
    visual_text_candidate_count: int = 0
    native_consumed_count: int = 0
    native_recovered_count: int = 0
    native_locked_count: int = 0
    native_gap_count: int = 0
    native_gap_filled_count: int = 0
    visual_supplement_count: int = 0
    visual_duplicate_skipped_count: int = 0
    source_conflict_count: int = 0
    native_vlm_order_applied_count: int = 0
    native_vlm_missing_filled_count: int = 0

    def to_dict(self) -> dict:
        return {
            "page_idx": self.page_idx,
            "layout_block_count": self.layout_block_count,
            "native_span_count": self.native_span_count,
            "vlm_content_block_count": self.vlm_content_block_count,
            "visual_text_candidate_count": self.visual_text_candidate_count,
            "native_consumed_count": self.native_consumed_count,
            "native_recovered_count": self.native_recovered_count,
            "native_locked_count": self.native_locked_count,
            "native_gap_count": self.native_gap_count,
            "native_gap_filled_count": self.native_gap_filled_count,
            "visual_supplement_count": self.visual_supplement_count,
            "visual_duplicate_skipped_count": self.visual_duplicate_skipped_count,
            "source_conflict_count": self.source_conflict_count,
            "native_vlm_order_applied_count": self.native_vlm_order_applied_count,
            "native_vlm_missing_filled_count": self.native_vlm_missing_filled_count,
        }
