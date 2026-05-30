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

    参数说明：
    - debug: 是否把融合决策细节写入 block 的 ``_fusion`` 字段。
    - visual_supplement: 是否把 VLM 看到但 native 未覆盖的文本补成额外 block。
    - supplement_in_visual_region: 是否允许在图片/视觉区域内补充 VLM 文本。
    - duplicate_threshold: visual_supplement 去重时的 bbox 覆盖率阈值。
    - native_overlap_threshold: native span 匹配到 VLM block bbox 的最小覆盖率。
    - min_visual_text_chars: visual_supplement 候选文本的最小长度。
    - native_gap_enable: 是否检测 native span 中间可能漏掉的图片字/特殊字。
    - native_gap_width_ratio: gap 阈值中的字符宽度倍数，越小越容易裁 gap。
    - native_gap_min_width: gap 检测的最小宽度，避免普通字间距被误判。
    - native_gap_crop_padding_ratio: 裁 gap 小图时按行高扩展的上下边距比例。
    - native_gap_max_chars: gap VLM 识别结果允许的最大字符数，防止小图读成整行。
    - native_vlm_alignment_enable: 是否启用 native/VLM 字符级对齐和保守补漏。
    - native_vlm_order_enable: 是否允许参考 VLM 文本顺序重排 native spans。
    - native_vlm_anchor_coverage: 重排候选必须达到的有效 token 锚点覆盖率。
    - native_vlm_score_gain: VLM-guided native 顺序相比几何顺序的最小分数提升。
    - native_vlm_max_conflict_ratio: native 与 VLM 字符冲突比例上限，超过则保守用 native。
    - native_vlm_missing_max_run: 连续从 VLM 补入 native 漏字的最大有效字符数。
    - native_vlm_missing_max_total_ratio: 单个 bbox 内补字总量占 native 有效字符数的上限。
    - native_vlm_line_break_enable: 是否用 VLM 连续文本来软化 native 的排版换行。
    - fusion_block_order_reference: 最终 block 顺序来源；默认 ``vlm``，可设为 ``geometry`` 回退旧排序。
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
    native_vlm_alignment_enable: bool = True
    native_vlm_order_enable: bool = True
    native_vlm_anchor_coverage: float = 0.70
    native_vlm_score_gain: float = 0.12
    native_vlm_max_conflict_ratio: float = 0.25
    native_vlm_missing_max_run: int = 2
    native_vlm_missing_max_total_ratio: float = 0.05
    native_vlm_line_break_enable: bool = True
    fusion_block_order_reference: str = "vlm"


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
        native_vlm_alignment_enable=_env_bool("MINERU_VLM_NATIVE_ALIGNMENT_ENABLE", True),
        native_vlm_order_enable=_env_bool("MINERU_VLM_NATIVE_ORDER_ENABLE", True),
        native_vlm_anchor_coverage=_env_float("MINERU_VLM_NATIVE_ANCHOR_COVERAGE", 0.70),
        native_vlm_score_gain=_env_float("MINERU_VLM_NATIVE_SCORE_GAIN", 0.12),
        native_vlm_max_conflict_ratio=_env_float("MINERU_VLM_NATIVE_MAX_CONFLICT_RATIO", 0.25),
        native_vlm_missing_max_run=max(0, int(_env_float("MINERU_VLM_NATIVE_MISSING_MAX_RUN", 2))),
        native_vlm_missing_max_total_ratio=_env_float("MINERU_VLM_NATIVE_MISSING_MAX_TOTAL_RATIO", 0.05),
        native_vlm_line_break_enable=_env_bool("MINERU_VLM_NATIVE_LINE_BREAK_ENABLE", True),
        fusion_block_order_reference=os.getenv("MINERU_VLM_FUSION_BLOCK_ORDER", "vlm").lower(),
    )
