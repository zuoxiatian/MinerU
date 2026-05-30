# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class NativeCorrectionConfig:
    """Configuration for VLM-primary native correction.

    - debug: write correction details into model_output/fused blocks.
    - native_correction_enable: enable per-bbox native correction.
    - min_anchor_coverage: minimum effective-token coverage required before native can
      correct VLM characters.
    - max_conflict_ratio: conflict-ratio cap for long native-only fills.
    - allow_missing_fill: allow native-only ordinary text to be inserted into VLM.
      This is conservative and does not affect VLM whitespace/line breaks.
    - missing_max_run: maximum consecutive native-only effective characters to insert.
    - missing_max_total_ratio: maximum total inserted characters relative to VLM
      effective-token count.
    - allow_large_missing_fill: allow longer native-only text runs when the
      alignment has low conflict and stable anchors on both sides.
    - large_missing_max_conflict_ratio: maximum conflict ratio allowed for long
      native-only fills.
    - native_gap_enable: detect image-based Chinese text gaps between native spans.
    - native_gap_width_ratio: gap threshold in character-width multiples.
    - native_gap_min_width: absolute minimum gap width.
    - native_gap_crop_padding_ratio: crop padding by line height.
    - native_gap_max_chars: maximum recognized Chinese chars for one gap.
    """
    debug: bool = False
    native_correction_enable: bool = True
    min_anchor_coverage: float = 0.65
    max_conflict_ratio: float = 0.25
    allow_missing_fill: bool = True
    missing_max_run: int = 2
    missing_max_total_ratio: float = 0.05
    allow_large_missing_fill: bool = True
    large_missing_max_conflict_ratio: float = 0.05
    native_gap_enable: bool = True
    native_gap_width_ratio: float = 1.0
    native_gap_min_width: float = 8.0
    native_gap_crop_padding_ratio: float = 0.8
    native_gap_max_chars: int = 8


def get_correction_config() -> NativeCorrectionConfig:
    return NativeCorrectionConfig(
        debug=_env_bool("MINERU_VLM_NATIVE_CORRECTION_DEBUG", False),
        native_correction_enable=_env_bool("MINERU_VLM_NATIVE_CORRECTION_ENABLE", True),
        min_anchor_coverage=_env_float("MINERU_VLM_NATIVE_CORRECTION_ANCHOR_COVERAGE", 0.65),
        max_conflict_ratio=_env_float("MINERU_VLM_NATIVE_CORRECTION_MAX_CONFLICT_RATIO", 0.25),
        allow_missing_fill=_env_bool("MINERU_VLM_NATIVE_CORRECTION_ALLOW_MISSING_FILL", True),
        missing_max_run=max(0, int(_env_float("MINERU_VLM_NATIVE_CORRECTION_MISSING_MAX_RUN", 2))),
        missing_max_total_ratio=_env_float("MINERU_VLM_NATIVE_CORRECTION_MISSING_MAX_TOTAL_RATIO", 0.05),
        allow_large_missing_fill=_env_bool("MINERU_VLM_NATIVE_CORRECTION_ALLOW_LARGE_MISSING_FILL", True),
        large_missing_max_conflict_ratio=_env_float(
            "MINERU_VLM_NATIVE_CORRECTION_LARGE_MISSING_MAX_CONFLICT_RATIO",
            0.05,
        ),
        native_gap_enable=_env_bool("MINERU_VLM_NATIVE_CORRECTION_GAP_ENABLE", True),
        native_gap_width_ratio=_env_float("MINERU_VLM_NATIVE_CORRECTION_GAP_WIDTH_RATIO", 1.0),
        native_gap_min_width=_env_float("MINERU_VLM_NATIVE_CORRECTION_GAP_MIN_WIDTH", 8.0),
        native_gap_crop_padding_ratio=_env_float("MINERU_VLM_NATIVE_CORRECTION_GAP_CROP_PADDING_RATIO", 0.8),
        native_gap_max_chars=max(1, int(_env_float("MINERU_VLM_NATIVE_CORRECTION_GAP_MAX_CHARS", 8))),
    )
