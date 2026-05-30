# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from difflib import SequenceMatcher

from mineru.backend.vlm_fusion.char_alignment import (
    CharAlignConfig,
    align_native_with_vlm,
    merge_native_with_vlm,
    normalize_char,
)
from mineru.backend.vlm_fusion.bbox import coverage_by_boxes
from mineru.backend.vlm_fusion.block_builder import (
    block_pdf_bbox,
    build_native_text_block,
    build_supplement_block,
    build_vlm_fallback_block,
    copy_model_block,
    is_structural_type,
    is_textual_type,
)
from mineru.backend.vlm_fusion.config import FusionConfig
from mineru.backend.vlm_fusion.native_text import (
    build_native_match,
    join_native_spans,
    join_native_spans_with_gaps,
    text_similarity,
)
from mineru.backend.vlm_fusion.schemas import FusionMetrics, PageFusionContext


def _row_order_key(pdf_bbox):
    """按行阅读顺序排序；y 坐标做 12pt 粒度归并，降低同一行内轻微抖动的影响。"""
    return (round(float(pdf_bbox[1]) / 12) * 12, float(pdf_bbox[0]))


def _detect_reading_columns(blocks: list[dict], page_width: int, page_height: int):
    """粗略判断页面是否存在左右分栏。

    这里只用于最终 fused block 排序，不改变 block 内容。算法刻意保持简单：
    先排除很宽的跨栏块，再按块中心点横向距离寻找最大间隔。
    """
    items = []
    for block in blocks:
        bbox = block_pdf_bbox(block, page_width, page_height)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0 or height <= 0:
            continue
        if width >= page_width * 0.7:
            continue
        items.append((block, bbox, (bbox[0] + bbox[2]) / 2))

    if len(items) < 4:
        return []

    items.sort(key=lambda item: item[2])
    gaps = [
        (items[idx + 1][2] - items[idx][2], idx)
        for idx in range(len(items) - 1)
    ]
    if not gaps:
        return []

    largest_gap, split_idx = max(gaps, key=lambda item: item[0])
    if largest_gap < page_width * 0.1:
        return []

    left_items = items[: split_idx + 1]
    right_items = items[split_idx + 1 :]
    if len(left_items) < 2 or len(right_items) < 2:
        return []

    return [
        {
            "x0": min(item[1][0] for item in group),
            "x1": max(item[1][2] for item in group),
            "center": sum(item[2] for item in group) / len(group),
        }
        for group in (left_items, right_items)
    ]


def _sort_fused_blocks(blocks: list[dict], page_width: int, page_height: int):
    """对融合后的 blocks 做阅读顺序排序。

    普通页面按行排序；疑似双栏页面先按栏，再按栏内行序排序。
    """
    columns = _detect_reading_columns(blocks, page_width, page_height)
    if not columns:
        blocks.sort(key=lambda block: _row_order_key(block_pdf_bbox(block, page_width, page_height)))
        return

    columns.sort(key=lambda column: column["center"])

    def column_order_key(block):
        bbox = block_pdf_bbox(block, page_width, page_height)
        center_x = (bbox[0] + bbox[2]) / 2
        block_width = bbox[2] - bbox[0]
        if block_width >= page_width * 0.7 and bbox[1] < page_height * 0.2:
            column_index = -1
        else:
            column_index = min(
                range(len(columns)),
                key=lambda idx: abs(center_x - columns[idx]["center"]),
            )
        row_y, row_x = _row_order_key(bbox)
        return (column_index, row_y, row_x)

    blocks.sort(key=column_order_key)


def _next_index(blocks: list[dict]) -> int:
    """返回新增 block 可使用的下一个 index。"""
    indexes = [int(block.get("index", idx + 1)) for idx, block in enumerate(blocks)]
    return (max(indexes) + 1) if indexes else 1


def _is_duplicate_visual_candidate(candidate, locked_spans, config: FusionConfig):
    """判断 VLM 视觉候选文本是否已经被可靠原生文本覆盖。

    同时看 bbox 覆盖率和文本相似度，避免把同一段文本重复补成 visual_supplement。
    """
    coverage = coverage_by_boxes(candidate.bbox, [span.bbox for span in locked_spans])
    native_text = join_native_spans(locked_spans)
    similarity = text_similarity(candidate.content, native_text)
    return coverage >= config.duplicate_threshold and similarity >= 0.55, coverage, similarity


def _effective_normalized_text(text: str) -> str:
    """生成用于查找 VLM 顺序锚点的归一化文本。

    这里会去掉空白并归一化标点/大小写，用于判断 native span 在 VLM 文本中的大致位置。
    该字符串不用于最终输出，最终输出仍来自 native 或保守补入后的文本。
    """
    return "".join(
        normalize_char(char)
        for char in text or ""
        if normalize_char(char).strip()
    )


def _join_spans_in_sequence(spans) -> str:
    """按给定 span 顺序拼接 native 文本。

    参数：
    - spans: 已经排好顺序的 NativeSpan 列表。

    返回：
    - 拼接后的文本。若相邻 span 的 y 差超过上一 span 行高的 0.6 倍，先保留一个
      ``\n``。后续 ``_soften_native_line_breaks_by_vlm`` 会根据 VLM 是否连续来软化它。
    """
    if not spans:
        return ""
    parts = []
    prev = None
    for span in spans:
        if prev is not None:
            prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
            if abs(span.bbox[1] - prev.bbox[1]) > prev_h * 0.6:
                parts.append("\n")
        parts.append(span.content)
        prev = span
    return "".join(parts).strip()


def _is_cjk_char(char: str) -> bool:
    """判断字符是否属于 CJK，用于决定软换行处是否需要补空格。"""
    return any(
        "\u3400" <= item <= "\u4dbf"
        or "\u4e00" <= item <= "\u9fff"
        or "\uf900" <= item <= "\ufaff"
        for item in char
    )


def _line_break_replacement(prev_char: str, next_char: str) -> str:
    """决定 native 换行被软化后替换成什么。

    中文/中文之间通常不需要空格；英文、数字或混合文本之间需要空格，避免单词粘连。
    标点附近尽量不加空格，避免生成 ``你好 ，`` 这类文本。
    """
    if not prev_char or not next_char:
        return ""
    if _is_cjk_char(prev_char) and _is_cjk_char(next_char):
        return ""
    if next_char in "，。！？；：、,.!?;:)]}）】」』":
        return ""
    if prev_char in "([{（【「『":
        return ""
    return " "


def _soften_native_line_breaks_by_vlm(text: str, vlm_text: str, config: FusionConfig) -> tuple[str, bool]:
    """用 VLM 的连续文本判断 native 换行是否只是排版折行。

    参数：
    - text: 当前 bbox 内已经合并好的文本，通常来自 native。
    - vlm_text: 同一 bbox 的 VLM 文本。
    - config: 融合配置，``native_vlm_line_break_enable`` 可关闭该逻辑。

    返回：
    - (softened_text, changed)。如果 VLM 文本没有换行而 native 有换行，则把 native
      的换行当作 soft break；如果 VLM 也有换行，则认为 VLM 也看到了结构性换行，不处理。
    """
    if not config.native_vlm_line_break_enable:
        return text, False
    if "\n" not in text or "\n" in (vlm_text or ""):
        return text, False

    chars = list(text)
    parts = []
    changed = False
    for index, char in enumerate(chars):
        if char != "\n":
            parts.append(char)
            continue
        prev_char = next((item for item in reversed(chars[:index]) if not item.isspace()), "")
        next_char = next((item for item in chars[index + 1:] if not item.isspace()), "")
        parts.append(_line_break_replacement(prev_char, next_char))
        changed = True
    normalized = "".join(parts)
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized.strip(), changed


def _span_vlm_position(span, vlm_norm: str, fallback_index: int) -> tuple[float, int]:
    """估计 native span 在 VLM 文本中的位置。

    参数：
    - span: 当前 native span。
    - vlm_norm: 去空白、归一化后的 VLM 文本。
    - fallback_index: 找不到锚点时使用的原始顺序，保证排序稳定。

    返回：
    - (position, fallback_index)。position 越小越靠前；找不到时为 inf。

    逻辑：
    1. 先做精确子串匹配。
    2. 单字符 span 做单字符查找。
    3. 长 span 做滑动窗口相似度匹配，处理少量 VLM 错字/标点差异。
    """
    span_norm = _effective_normalized_text(span.content)
    if not span_norm or not vlm_norm:
        return (float("inf"), fallback_index)
    exact = vlm_norm.find(span_norm)
    if exact >= 0:
        return (float(exact), fallback_index)
    if len(span_norm) == 1:
        single = vlm_norm.find(span_norm)
        if single >= 0:
            return (float(single), fallback_index)

    best_pos = float("inf")
    best_score = 0.0
    window = max(1, len(span_norm))
    for pos in range(0, max(1, len(vlm_norm) - window + 1)):
        ratio = SequenceMatcher(None, span_norm, vlm_norm[pos: pos + window]).ratio()
        if ratio > best_score:
            best_score = ratio
            best_pos = float(pos)
    if best_score >= 0.75:
        return (best_pos, fallback_index)
    return (float("inf"), fallback_index)


def _build_native_vlm_config(config: FusionConfig) -> CharAlignConfig:
    """把页面级 FusionConfig 中的字符对齐参数转换成 CharAlignConfig。"""
    return CharAlignConfig(
        native_missing_max_run=config.native_vlm_missing_max_run,
        native_missing_max_total_ratio=config.native_vlm_missing_max_total_ratio,
        max_conflict_ratio=config.native_vlm_max_conflict_ratio,
    )


def _build_vlm_ordered_native_text(spans, native_text: str, vlm_text: str, config: FusionConfig):
    """尝试按 VLM 阅读顺序重排 native spans。

    参数：
    - spans: 当前 bbox 匹配到的 native spans。
    - native_text: 按几何顺序拼接的 native 文本。
    - vlm_text: 同一 bbox 的 VLM 文本，作为阅读顺序参考。
    - config: 控制是否启用、锚点覆盖率阈值、分数提升阈值等。

    返回：
    - (selected_text, debug, applied)。

    只有当重排候选满足以下条件才应用：
    - anchor_coverage >= native_vlm_anchor_coverage。
    - conflict_ratio <= native_vlm_max_conflict_ratio。
    - score_gain >= native_vlm_score_gain。

    这保证 VLM 只指导顺序，不会在证据不足时带偏 native。
    """
    if not config.native_vlm_order_enable or len(spans) < 2 or not vlm_text.strip():
        return native_text, None, False

    vlm_norm = _effective_normalized_text(vlm_text)
    ordered_spans = sorted(
        enumerate(spans),
        key=lambda item: _span_vlm_position(item[1], vlm_norm, item[0]),
    )
    reordered = [span for _, span in ordered_spans]
    if [span.uid for span in reordered] == [span.uid for span in spans]:
        return native_text, None, False

    candidate_text = _join_spans_in_sequence(reordered)
    base_alignment = align_native_with_vlm(native_text, vlm_text)
    candidate_alignment = align_native_with_vlm(candidate_text, vlm_text)
    score_gain = candidate_alignment.score - base_alignment.score

    applied = (
        candidate_alignment.anchor_coverage >= config.native_vlm_anchor_coverage
        and candidate_alignment.conflict_ratio <= config.native_vlm_max_conflict_ratio
        and score_gain >= config.native_vlm_score_gain
    )
    debug = {
        "base_alignment": base_alignment.to_debug(),
        "candidate_alignment": candidate_alignment.to_debug(),
        "score_gain": round(score_gain, 6),
        "candidate_text": candidate_text,
    }
    if applied:
        return candidate_text, debug, True
    return native_text, debug, False


def _fuse_native_text_for_block(native_text: str, vlm_text: str, spans, config: FusionConfig):
    """在单个 VLM bbox 内融合 native 文本和 VLM 文本。

    参数：
    - native_text: 当前 bbox 的 native 候选文本，可能已经包含 gap 识别补字。
    - vlm_text: 当前 bbox 的 VLM 文本。
    - spans: 当前 bbox 匹配到的 native spans，用于 VLM-guided span 重排。
    - config: 融合阈值和开关。

    返回：
    - (final_text, debug, missing_filled_count, order_applied)。

    处理顺序：
    1. 用 VLM 文本尝试重排 native spans。
    2. 做字符级对齐，只保守补 native 漏掉的普通文字。
    3. 如果 VLM 认为该 bbox 是连续句子，则软化 native 的排版换行。
    """
    if not config.native_vlm_alignment_enable or not vlm_text.strip():
        return native_text, {}, 0, False

    ordered_text, order_debug, order_applied = _build_vlm_ordered_native_text(
        spans,
        native_text,
        vlm_text,
        config,
    )
    merge_result = merge_native_with_vlm(
        ordered_text,
        vlm_text,
        _build_native_vlm_config(config),
    )
    merged_text, line_break_softened = _soften_native_line_breaks_by_vlm(
        merge_result.text,
        vlm_text,
        config,
    )
    debug = {
        "selected_native_text": ordered_text,
        "merged_text": merged_text,
        "merge_decision": merge_result.decision,
        "line_break_softened": line_break_softened,
        "alignment": merge_result.alignment.to_debug(),
    }
    if order_debug is not None:
        debug["order"] = order_debug
    return merged_text, debug, merge_result.filled_missing_count, order_applied


def fuse_page(context: PageFusionContext, config: FusionConfig) -> tuple[list[dict], FusionMetrics, list[dict]]:
    """融合单页的 VLM 结果和 PDF 原生文本。

    决策优先级：
    1. 文本类 block 如果能匹配到高质量原生文本，使用 PDF 原生文本锁定 content。
    2. 原生文本不可靠时，保留 VLM content 作为 fallback。
    3. 结构类 block 直接保留 VLM 结果。
    4. 未被原生文本覆盖的 VLM 文本可作为 visual_supplement 补充。
    5. 仍未消费的 PDF 原生 span 可作为 pdf_native_recovered 补回。
    """
    metrics = FusionMetrics(
        page_idx=context.page_index,
        layout_block_count=len(context.layout_blocks),
        native_span_count=len(context.native_spans),
        vlm_content_block_count=len(context.vlm_blocks),
        visual_text_candidate_count=len(context.visual_candidates),
        native_gap_count=len(context.native_gaps),
        native_gap_filled_count=sum(1 for gap in context.native_gaps if gap.content.strip()),
    )

    fused_blocks: list[dict] = []
    compare_records: list[dict] = []
    locked_native_spans = []
    emitted_vlm_indices = set()

    for index, block in enumerate(context.vlm_blocks):
        # 以 VLM block 为主轴遍历页面内容；文本类 block 会尝试寻找同位置的 PDF 原生文本。
        block.setdefault("index", index + 1)
        block_type = block.get("type", "text")
        if is_textual_type(block_type):
            block_bbox = block_pdf_bbox(block, context.page_width, context.page_height)
            native_match = build_native_match(
                block_bbox,
                context.native_spans,
                config.native_overlap_threshold,
            )
            vlm_content = block.get("content") or ""
            if native_match.reliable:
                # 原生文本可靠时优先使用原生文本，减少 VLM/OCR 对普通文字的误读。
                # 如果前面识别到了同一个 block 内的 gap，则把 gap 内容插回 span 之间。
                block_gaps = [
                    gap
                    for gap in context.native_gaps
                    if gap.block_index == block.get("index", index + 1) and gap.content.strip()
                ]
                native_content = (
                    join_native_spans_with_gaps(native_match.spans, block_gaps)
                    if block_gaps
                    else native_match.content
                )
                alignment_debug = {}
                missing_filled_count = 0
                order_applied = False
                native_content, alignment_debug, missing_filled_count, order_applied = _fuse_native_text_for_block(
                    native_content,
                    vlm_content,
                    native_match.spans,
                    config,
                )
                for span in native_match.spans:
                    span.consumed = True
                locked_native_spans.extend(native_match.spans)
                emitted_vlm_indices.add(block.get("index", index + 1))
                metrics.native_locked_count += 1
                metrics.native_vlm_missing_filled_count += missing_filled_count
                if order_applied:
                    metrics.native_vlm_order_applied_count += 1
                if vlm_content and text_similarity(native_match.content, vlm_content) < 0.55:
                    metrics.source_conflict_count += 1
                fused_blocks.append(
                    build_native_text_block(
                        block,
                        native_content,
                        {
                            "decision": "native_locked",
                            "native_quality": native_match.quality,
                            "vlm_text": vlm_content,
                            "native_gap_count": len(block_gaps),
                            "native_vlm_alignment": alignment_debug,
                        } if config.debug else None,
                    )
                )
                compare_records.append(
                    {
                        "page_index": context.page_index,
                        "bbox_index": block.get("index", index + 1),
                        "native_text": native_match.content,
                        "vlm_text": vlm_content,
                        "final_text": native_content,
                    }
                )
            else:
                # 原生文本质量差或没有匹配内容时，回退到 VLM 识别文本。
                # 如果 VLM 有内容，也把已匹配 span 标记为 consumed，避免后面重复补出。
                if vlm_content:
                    for span in native_match.spans:
                        span.consumed = True
                emitted_vlm_indices.add(block.get("index", index + 1))
                fused_blocks.append(
                    build_vlm_fallback_block(
                        block,
                        {
                            "decision": "vlm_fallback",
                            "native_quality": native_match.quality,
                            "native_text": native_match.content,
                        } if config.debug else None,
                    )
                )
                compare_records.append(
                    {
                        "page_index": context.page_index,
                        "bbox_index": block.get("index", index + 1),
                        "native_text": native_match.content,
                        "vlm_text": vlm_content,
                        "final_text": vlm_content,
                    }
                )
        elif is_structural_type(block_type):
            # 表格、图片、公式、代码等结构类区域不做原生文本替换。
            fused_blocks.append(copy_model_block(block))
        else:
            fused_blocks.append(copy_model_block(block))

    metrics.native_consumed_count = sum(1 for span in context.native_spans if span.consumed)

    if config.visual_supplement:
        next_index = _next_index(fused_blocks)
        for candidate in context.visual_candidates:
            # visual_supplement 用于补 VLM 看见但原生文本没有覆盖的文字，
            # 例如图片中的文字、页眉页脚或特殊渲染文本。
            if len(candidate.content.strip()) < config.min_visual_text_chars:
                continue
            if candidate.raw.get("index") in emitted_vlm_indices:
                metrics.visual_duplicate_skipped_count += 1
                continue
            duplicate, coverage, similarity = _is_duplicate_visual_candidate(
                candidate,
                locked_native_spans,
                config,
            )
            if duplicate:
                metrics.visual_duplicate_skipped_count += 1
                continue
            fused_blocks.append(
                build_supplement_block(
                    candidate,
                    context.page_width,
                    context.page_height,
                    next_index,
                    {
                        "decision": "visual_supplement",
                        "overlap_with_locked_native": coverage,
                        "native_similarity": similarity,
                    } if config.debug else None,
                )
            )
            next_index += 1
            metrics.visual_supplement_count += 1

    next_index = _next_index(fused_blocks)
    structural_bboxes = [
        block_pdf_bbox(block, context.page_width, context.page_height)
        for block in fused_blocks
        if is_structural_type(block.get("type", ""))
    ]
    for span in context.native_spans:
        if span.consumed or not span.content.strip():
            continue
        if coverage_by_boxes(span.bbox, structural_bboxes) >= 0.5:
            # 落在表格/图片等结构区域里的残留原生文本不补，避免破坏结构块。
            span.consumed = True
            continue
        # 没有被任何 VLM 文本块消费到的原生文本，作为恢复块补回。
        fused_blocks.append(
            {
                "type": "text",
                "bbox": [
                    round(span.bbox[0] / context.page_width, 6),
                    round(span.bbox[1] / context.page_height, 6),
                    round(span.bbox[2] / context.page_width, 6),
                    round(span.bbox[3] / context.page_height, 6),
                ],
                "angle": 0,
                "content": span.content,
                "index": next_index,
                "source": "pdf_native_recovered",
            }
        )
        span.consumed = True
        next_index += 1
        metrics.native_recovered_count += 1

    if config.fusion_block_order_reference != "vlm":
        _sort_fused_blocks(fused_blocks, context.page_width, context.page_height)
    for index, block in enumerate(fused_blocks):
        block["index"] = index + 1
    return fused_blocks, metrics, compare_records
