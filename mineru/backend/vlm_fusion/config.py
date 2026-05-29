# Copyright (c) Opendatalab. All rights reserved.
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量，支持 1/true/yes/on。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；非法值回退默认值。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class FusionConfig:
    """VLM fusion 的运行配置。

    大多数阈值通过环境变量覆盖，便于在线上调试不同 PDF 类型时快速试验。
    """
    debug: bool = False
    visual_supplement: bool = True
    supplement_in_visual_region: bool = False
    duplicate_threshold: float = 0.65
    native_overlap_threshold: float = 0.45
    min_visual_text_chars: int = 1
    native_gap_enable: bool = True
    native_gap_width_ratio: float = 2.5
    native_gap_min_width: float = 18.0
    native_gap_crop_padding_ratio: float = 0.8
    native_gap_max_chars: int = 20


def get_fusion_config() -> FusionConfig:
    """从环境变量构造融合配置。"""
    return FusionConfig(
        debug=_env_bool("MINERU_VLM_FUSION_DEBUG", False),
        visual_supplement=_env_bool("MINERU_VLM_VISUAL_SUPPLEMENT", True),
        supplement_in_visual_region=_env_bool("MINERU_VLM_SUPPLEMENT_IN_IMAGE", False),
        duplicate_threshold=_env_float("MINERU_VLM_DUPLICATE_THRESHOLD", 0.65),
        native_overlap_threshold=_env_float("MINERU_VLM_NATIVE_OVERLAP_THRESHOLD", 0.45),
        min_visual_text_chars=max(1, int(_env_float("MINERU_VLM_MIN_VISUAL_TEXT_CHARS", 1))),
        native_gap_enable=_env_bool("MINERU_VLM_NATIVE_GAP_ENABLE", True),
        native_gap_width_ratio=_env_float("MINERU_VLM_NATIVE_GAP_WIDTH_RATIO", 2.5),
        native_gap_min_width=_env_float("MINERU_VLM_NATIVE_GAP_MIN_WIDTH", 18.0),
        native_gap_crop_padding_ratio=_env_float("MINERU_VLM_NATIVE_GAP_CROP_PADDING_RATIO", 0.8),
        native_gap_max_chars=max(1, int(_env_float("MINERU_VLM_NATIVE_GAP_MAX_CHARS", 20))),
    )
