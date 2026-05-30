# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re

from PIL import Image

from mineru.backend.vlm_fusion.block_builder import block_pdf_bbox, is_textual_type
from mineru.backend.vlm_fusion.config import FusionConfig
from mineru.backend.vlm_fusion.native_text import detect_native_gaps_for_block
from mineru.backend.vlm_fusion.schemas import NativeGap, NativeSpan


_DROP_GAP_VALUES = {
    "",
    "[Non-Text]",
    "Non-Text",
    "\u65e0",
    "\u65e0\u6587\u672c",
    "\u6ca1\u6709\u6587\u5b57",
    "\u6ca1\u6709\u6587\u672c",
}
# gap 识别只负责补 span 中间的小缺口。模型常会把裁图边缘的标点或括号也读出来，
# 所以这里只清理 gap 结果的首尾标点，不影响普通文本块。
_EDGE_PUNCTUATION = (
    " \t\r\n,"
    "\uff0c.\u3002!\uff01?\uff1f:\uff1a;\uff1b\u3001"
    "'\"\u201c\u201d\u2018\u2019()\uff08\uff09[]\u3010\u3011{}"
    "-\u2013\u2014\u2190\u2192\u2194"
)


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )


def _is_safe_gap_text(text: str) -> bool:
    """判断 gap 识别结果是否适合插回 native 文本。

    gap 裁图很小，VLM 容易把目录连接线、箭头、页码或边缘符号误读成内容。
    因此这里只允许较安全的缺字结果：
    - 中文等 CJK 字符直接允许，例如“公羊”。
    - 纯数字不允许，避免页码/编号被误补。
    - 单个 ASCII 字母不允许，避免装饰线或边缘噪声误成 L/T/I。
    - 纯符号不允许，例如“→”“-”“—”。
    """
    if not text:
        return False
    if _contains_cjk(text):
        return True
    if text.isdigit():
        return False
    if text.isascii() and len(text) <= 1:
        return False
    return any(char.isalpha() for char in text)


def detect_native_gaps(
    vlm_blocks: list[dict],
    native_spans: list[NativeSpan],
    page_width: int,
    page_height: int,
    config: FusionConfig,
) -> list[NativeGap]:
    """检测一页内所有文本 block 的 PDF 原生文本缺口。"""
    if not config.native_gap_enable:
        return []
    gaps: list[NativeGap] = []
    for block_index, block in enumerate(vlm_blocks):
        block_type = block.get("type", "text")
        if not is_textual_type(block_type):
            continue
        effective_index = int(block.get("index", block_index + 1))
        block_bbox = block_pdf_bbox(block, page_width, page_height)
        gaps.extend(
            detect_native_gaps_for_block(
                effective_index,
                block_bbox,
                native_spans,
                config.native_overlap_threshold,
                width_ratio=config.native_gap_width_ratio,
                min_width=config.native_gap_min_width,
            )
        )
    return gaps


def crop_gap_image(
    page_image: Image.Image,
    gap: NativeGap,
    scale: float,
    config: FusionConfig,
) -> Image.Image | None:
    """按 PDF 坐标中的 gap bbox 裁切页面图片。

    gap.bbox 使用 PDF 页面坐标，page_image 使用渲染后的像素坐标，因此需要乘以 scale。
    上下 padding 用于保留完整字形，左右 padding 较小，避免把相邻 span 一起读进来。
    """
    x0, y0, x1, y1 = gap.bbox
    line_height = max(1.0, y1 - y0)
    padding = line_height * config.native_gap_crop_padding_ratio
    crop_box = (
        max(0, int((x0 - padding * 0.35) * scale)),
        max(0, int((y0 - padding) * scale)),
        min(page_image.width, int((x1 + padding * 0.35) * scale)),
        min(page_image.height, int((y1 + padding) * scale)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None
    return page_image.crop(crop_box)


def normalize_gap_text(text: str | None, max_chars: int) -> str:
    """清理 gap 裁图的 VLM 识别结果。

    注意：这不是全局文本归一化。它只用于缺口补字：
    - 删除所有空白，避免把一个缺口补成多段。
    - 去掉首尾标点，降低裁图边缘误识别影响。
    - 过滤模型表示“无文本”的常见回答。
    - 过长内容直接丢弃，避免小缺口识别成整行文本。
    """
    if text is None:
        return ""
    normalized = re.sub(r"\s+", "", str(text))
    normalized = normalized.strip(_EDGE_PUNCTUATION)
    if normalized in _DROP_GAP_VALUES:
        return ""
    if len(normalized) > max_chars:
        return ""
    visible_count = sum(1 for char in normalized if not char.isspace())
    if visible_count == 0:
        return ""
    if not _is_safe_gap_text(normalized):
        return ""
    return normalized


def recognize_native_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: FusionConfig,
) -> list[NativeGap]:
    """同步识别 gap 内容，并把结果写回 NativeGap.content。"""
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = crop_gap_image(page_image, gap, scale, config)
        if crop is None:
            continue
        gap_images.append(crop)
        gap_refs.append(gap)
    if not gap_images:
        return gaps

    try:
        outputs = predictor.batch_content_extract(gap_images, types=["text"] * len(gap_images))
    finally:
        for image in gap_images:
            image.close()

    for gap, output in zip(gap_refs, outputs):
        gap.content = normalize_gap_text(
            None if output is None else str(output),
            config.native_gap_max_chars,
        )
    return gaps


async def aio_recognize_native_gaps(
    predictor,
    page_image: Image.Image,
    scale: float,
    gaps: list[NativeGap],
    config: FusionConfig,
) -> list[NativeGap]:
    """异步识别 gap 内容，并把结果写回 NativeGap.content。"""
    if not gaps:
        return gaps
    gap_images = []
    gap_refs = []
    for gap in gaps:
        crop = crop_gap_image(page_image, gap, scale, config)
        if crop is None:
            continue
        gap_images.append(crop)
        gap_refs.append(gap)
    if not gap_images:
        return gaps

    try:
        outputs = await predictor.aio_batch_content_extract(
            gap_images,
            types=["text"] * len(gap_images),
        )
    finally:
        for image in gap_images:
            image.close()

    for gap, output in zip(gap_refs, outputs):
        gap.content = normalize_gap_text(
            None if output is None else str(output),
            config.native_gap_max_chars,
        )
    return gaps
