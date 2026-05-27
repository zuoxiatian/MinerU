# Copyright (c) Opendatalab. All rights reserved.
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
class FusionConfig:
    debug: bool = False
    visual_supplement: bool = True
    supplement_in_visual_region: bool = False
    duplicate_threshold: float = 0.65
    native_overlap_threshold: float = 0.45
    min_visual_text_chars: int = 1


def get_fusion_config() -> FusionConfig:
    return FusionConfig(
        debug=_env_bool("MINERU_VLM_FUSION_DEBUG", False),
        visual_supplement=_env_bool("MINERU_VLM_VISUAL_SUPPLEMENT", True),
        supplement_in_visual_region=_env_bool("MINERU_VLM_SUPPLEMENT_IN_IMAGE", False),
        duplicate_threshold=_env_float("MINERU_VLM_DUPLICATE_THRESHOLD", 0.65),
        native_overlap_threshold=_env_float("MINERU_VLM_NATIVE_OVERLAP_THRESHOLD", 0.45),
        min_visual_text_chars=max(1, int(_env_float("MINERU_VLM_MIN_VISUAL_TEXT_CHARS", 1))),
    )

