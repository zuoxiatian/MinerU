# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from loguru import logger

from mineru.backend.vlm_ocr_native.bbox import center_in_bbox, intersection_area, reading_order_key
from mineru.backend.vlm_ocr_native.schemas import BBox, NativeGap, NativeMatch, NativeSpan
from mineru.utils.pdf_text_tool import get_lines_from_chars, get_page_chars


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _coerce_bbox(value) -> BBox | None:
    """把不同来源的 bbox 表示统一成合法的 [x0, y0, x1, y1] 浮点列表。"""
    if value is None:
        return None
    if hasattr(value, "bbox"):
        value = value.bbox
    if isinstance(value, dict):
        value = value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _line_text(line: dict) -> str:
    """从 line 结构中取文本；优先使用 line.text，否则拼接内部 spans。"""
    text = line.get("text")
    if isinstance(text, str):
        return text
    spans = line.get("spans") or []
    parts = []
    for span in spans:
        span_text = span.get("text") or span.get("content")
        if isinstance(span_text, str):
            parts.append(span_text)
    return "".join(parts)


def extract_native_spans(pdf_page) -> list[NativeSpan]:
    """从 PDF 页面抽取原生文本 span。

    这里不做 OCR，只读取 PDF 内置的文本层。每个 span 会保留 bbox、content 和 uid，
    后续融合阶段通过 bbox 把这些 span 匹配到 VLM 文本 block。
    """
    try:
        page_chars = get_page_chars(pdf_page)
        lines = get_lines_from_chars(page_chars["chars"])
    except Exception as exc:
        logger.debug(f"Failed to extract native PDF text: {exc}")
        return []

    native_spans: list[NativeSpan] = []
    next_uid = 0
    for line in lines:
        line_bbox = _coerce_bbox(line.get("bbox"))
        spans = line.get("spans") or []
        added_span = False
        for span in spans:
            # span bbox 缺失时使用 line bbox 兜底，保证文本仍有几何位置可用于匹配。
            bbox = _coerce_bbox(span.get("bbox")) or line_bbox
            content = span.get("text") or span.get("content") or ""
            if bbox and str(content).strip():
                native_spans.append(NativeSpan(bbox=bbox, content=str(content), uid=next_uid))
                next_uid += 1
                added_span = True
        if not added_span and line_bbox:
            content = _line_text(line)
            if content.strip():
                native_spans.append(NativeSpan(bbox=line_bbox, content=content, uid=next_uid))
                next_uid += 1
    return native_spans


def _span_overlap_ratio(span_bbox: BBox, block_bbox: BBox) -> float:
    """计算原生 span 被目标 block 覆盖的比例。"""
    span_area = max(1.0, (span_bbox[2] - span_bbox[0]) * (span_bbox[3] - span_bbox[1]))
    return intersection_area(span_bbox, block_bbox) / span_area


def match_native_to_bbox(
    block_bbox: BBox,
    native_spans: list[NativeSpan],
    overlap_threshold: float,
) -> list[NativeSpan]:
    """把未消费的 PDF 原生 span 匹配到一个 VLM block 的 bbox。

    匹配条件是 span 与 block 有足够重叠，或者 span 中心点落在 block 内。
    已消费的 span 不会再次参与匹配，避免同一段文本被重复输出。
    """
    matched = []
    for span in native_spans:
        if span.consumed:
            continue
        if (
            _span_overlap_ratio(span.bbox, block_bbox) >= overlap_threshold
            or center_in_bbox(span.bbox, block_bbox)
        ):
            matched.append(span)
    matched.sort(key=lambda span: reading_order_key({"bbox": span.bbox}))
    return matched


def join_native_spans(spans: list[NativeSpan]) -> str:
    """按阅读顺序拼接原生 spans。

    同一行内直接拼接，不额外插入空格；换行之间用 ``\n`` 分隔。
    这样可以最大限度保留 PDF 原生文本层的字符和标点。
    """
    if not spans:
        return ""
    lines: list[list[NativeSpan]] = []
    for span in spans:
        if not lines:
            lines.append([span])
            continue
        prev = lines[-1][-1]
        prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
        if abs(span.bbox[1] - prev.bbox[1]) <= prev_h * 0.6:
            lines[-1].append(span)
        else:
            lines.append([span])

    line_texts = []
    for line in lines:
        line.sort(key=lambda span: span.bbox[0])
        line_texts.append("".join(span.content for span in line).strip())
    return "\n".join(text for text in line_texts if text)


def group_native_spans_to_lines(spans: list[NativeSpan]) -> list[list[NativeSpan]]:
    """把原生 spans 按 y 坐标聚合成行，并在行内按 x 坐标排序。"""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda span: reading_order_key({"bbox": span.bbox}))
    lines: list[list[NativeSpan]] = []
    for span in sorted_spans:
        if not lines:
            lines.append([span])
            continue
        prev = lines[-1][-1]
        prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
        if abs(span.bbox[1] - prev.bbox[1]) <= prev_h * 0.6:
            lines[-1].append(span)
        else:
            lines.append([span])
    for line in lines:
        line.sort(key=lambda span: span.bbox[0])
    return lines


def join_native_spans_with_gaps(spans: list[NativeSpan], gaps: list[NativeGap]) -> str:
    """拼接原生 spans，并把已识别的 gap 内容插回左侧 span 之后。"""
    if not spans:
        return ""
    gaps_by_left_uid: dict[int, list[NativeGap]] = {}
    for gap in gaps:
        if not gap.content.strip():
            continue
        gaps_by_left_uid.setdefault(gap.left_span_uid, []).append(gap)
    for gap_list in gaps_by_left_uid.values():
        gap_list.sort(key=lambda gap: gap.bbox[0])

    line_texts = []
    for line in group_native_spans_to_lines(spans):
        parts = []
        for span in line:
            parts.append(span.content)
            for gap in gaps_by_left_uid.get(span.uid, []):
                parts.append(gap.content)
        line_text = "".join(parts).strip()
        if line_text:
            line_texts.append(line_text)
    return "\n".join(line_texts)


def score_native_text_quality(text: str) -> float:
    """评估 PDF 原生文本是否可用。

    当前主要惩罚控制字符、Unicode replacement character 和可疑文字系统字符。分数越高，说明
    文本层越干净；融合阶段默认 0.85 以上才会优先锁定原生文本。
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    control_count = len(_CONTROL_RE.findall(stripped))
    replacement_count = stripped.count("\ufffd")
    suspicious_count = sum(1 for char in stripped if _is_suspicious_text_layer_char(char))
    visible_count = sum(1 for char in stripped if not char.isspace())
    if visible_count == 0:
        return 0.0
    bad_ratio = (control_count + replacement_count + suspicious_count) / max(1, len(stripped))
    quality = 1.0 - bad_ratio
    if visible_count <= 1:
        quality *= 0.8
    return max(0.0, min(1.0, quality))


def _is_suspicious_text_layer_char(char: str) -> bool:
    if not char or char.isspace() or char == "\ufffd":
        return False
    return char.isalpha() and not _is_common_text_letter(char)


def _is_common_text_letter(char: str) -> bool:
    return _is_cjk(char) or _is_latin(char) or _is_hiragana_or_katakana(char) or _is_hangul(char)


def _is_latin(char: str) -> bool:
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def _is_cjk(char: str) -> bool:
    return (
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )


def _is_hiragana_or_katakana(char: str) -> bool:
    return (
        "\u3040" <= char <= "\u309f"
        or "\u30a0" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        or "\uff66" <= char <= "\uff9f"
    )


def _is_hangul(char: str) -> bool:
    return "\uac00" <= char <= "\ud7af" or "\u1100" <= char <= "\u11ff"


def build_native_match(
    block_bbox: BBox,
    native_spans: list[NativeSpan],
    overlap_threshold: float,
) -> NativeMatch:
    """构造一个 VLM block 对应的原生文本匹配结果。"""
    spans = match_native_to_bbox(block_bbox, native_spans, overlap_threshold)
    content = join_native_spans(spans)
    quality = score_native_text_quality(content)
    return NativeMatch(
        spans=spans,
        content=content,
        quality=quality,
        reliable=quality >= 0.85 and bool(content.strip()),
    )


def detect_native_gaps_for_block(
    block_index: int,
    block_bbox: BBox,
    native_spans: list[NativeSpan],
    overlap_threshold: float,
    *,
    width_ratio: float,
    min_width: float,
) -> list[NativeGap]:
    """检测单个文本 block 内原生 span 之间的可疑空隙。

    参数：
    - block_index: VLM block 序号，用于后续把 gap 识别结果插回对应 block。
    - block_bbox: VLM block 的 PDF 坐标 bbox。
    - native_spans: 当前页所有 PDF 原生文本 span。
    - overlap_threshold: span 与 block 的最小覆盖率，低于该值不参与本 block gap 检测。
    - width_ratio: 字符宽度倍数阈值，越小越容易认为两个 span 之间存在 gap。
    - min_width: gap 的绝对最小宽度，避免普通字间距被误判。

    检测结果只记录 bbox 和左右 span uid，真正的文字内容稍后由 VLM 裁图识别。

    阈值策略：
    - 基础阈值为 ``max(min_width, lower_median_char_width * width_ratio)``。
    - 使用 lower median 是为了避免“短 span + 长 span”时，长 span 的平均字宽把阈值抬太高。
    - 如果相邻 span 中有短 span，则再用局部短 span 阈值放宽一次，捕捉漫画里
      “这头 [图片字] 用...” 这类短 span 旁边的图片化文字。
    """
    matched = match_native_to_bbox(block_bbox, native_spans, overlap_threshold)
    gaps: list[NativeGap] = []
    for line in group_native_spans_to_lines(matched):
        if len(line) < 2:
            continue
        char_widths = [
            (span.bbox[2] - span.bbox[0]) / max(1, len(span.content.strip()))
            for span in line
            if span.content.strip()
        ]
        if not char_widths:
            continue
        char_widths.sort()
        # 使用 lower median，避免两个 span 时右侧长 span 的平均字宽把阈值抬高，
        # 从而漏掉短 span 后面的图片字 gap。
        median_char_width = char_widths[(len(char_widths) - 1) // 2]
        gap_threshold = max(min_width, median_char_width * width_ratio)

        for left, right in zip(line, line[1:]):
            gap_width = right.bbox[0] - left.bbox[2]
            left_text = left.content.strip()
            right_text = right.content.strip()
            left_char_width = (left.bbox[2] - left.bbox[0]) / max(1, len(left_text))
            right_char_width = (right.bbox[2] - right.bbox[0]) / max(1, len(right_text))
            local_char_width = max(1.0, min(left_char_width, right_char_width, median_char_width))
            pair_gap_threshold = gap_threshold
            if min(len(left_text), len(right_text)) <= 2:
                # 短 span 旁边常出现图片化字或装饰字。这里用左右 span 中更小的
                # 字符宽度估算局部阈值，降低漏检概率，同时仍保留 min_width 下限。
                left_char_width = (left.bbox[2] - left.bbox[0]) / max(1, len(left_text))
                short_span_threshold = max(min_width, min(left_char_width, right_char_width) * 2.0)
                pair_gap_threshold = min(pair_gap_threshold, short_span_threshold)
            if gap_width < pair_gap_threshold:
                continue
            y0 = max(min(left.bbox[1], right.bbox[1]), block_bbox[1])
            y1 = min(max(left.bbox[3], right.bbox[3]), block_bbox[3])
            if y1 <= y0:
                continue
            gaps.append(
                NativeGap(
                    bbox=[
                        max(left.bbox[2], block_bbox[0]),
                        y0,
                        min(right.bbox[0], block_bbox[2]),
                        y1,
                    ],
                    block_index=block_index,
                    left_span_uid=left.uid,
                    right_span_uid=right.uid,
                    max_chars=max(1, round(gap_width / local_char_width)),
                )
            )
    return gaps


def text_similarity(left: str, right: str) -> float:
    """比较两段文本相似度。

    这里只去除空白，不删除标点；因此标点差异会影响相似度。
    """
    left_norm = re.sub(r"\s+", "", left or "")
    right_norm = re.sub(r"\s+", "", right or "")
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()
