# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from collections import Counter
import unicodedata

from mineru.backend.vlm_ocr_native.block_builder import (
    block_pdf_bbox,
    copy_model_block,
    is_structural_type,
    is_textual_type,
)
from mineru.backend.vlm_ocr_native.char_alignment import (
    CONFLICT,
    CONTROL_OR_UNKNOWN,
    DIGIT,
    EQUIVALENT,
    LETTER_OR_CJK,
    MATCH,
    NATIVE_EXTRA,
    NATIVE_MISSING,
    PUNCT,
    SPACE,
    align_native_with_vlm,
    effective_token_count,
)
from mineru.backend.vlm_ocr_native.native_text import build_native_match
from mineru.backend.vlm_ocr_native.config import NativeCorrectionConfig
from mineru.backend.vlm_ocr_native.schemas import CorrectionMetrics, PageCorrectionContext


def correct_page(
    context: PageCorrectionContext,
    config: NativeCorrectionConfig,
) -> tuple[list[dict], CorrectionMetrics, list[dict]]:
    """Correct one page using VLM as the primary source.

    VLM owns block order, content formatting, whitespace, line breaks, and structural
    output. Native PDF text is matched per bbox and used only as a character-level
    correction source for textual blocks.
    """
    metrics = CorrectionMetrics(
        page_idx=context.page_index,
        native_span_count=len(context.native_spans),
        vlm_content_block_count=len(context.vlm_blocks),
    )
    corrected_blocks: list[dict] = []
    compare_records: list[dict] = []

    for index, block in enumerate(context.vlm_blocks):
        block.setdefault("index", index + 1)
        block_type = block.get("type", "text")
        block_bbox = block_pdf_bbox(block, context.page_width, context.page_height)
        native_match = build_native_match(block_bbox, context.native_spans, overlap_threshold=0.45)
        native_text = native_match.content
        vlm_text = block.get("content") or ""
        correctable = _is_correctable_text_block(block_type, block)

        if not correctable:
            corrected = copy_model_block(block)
            corrected_blocks.append(corrected)
            final_text = corrected.get("content") or vlm_text
            debug = None
        elif not config.native_correction_enable or not vlm_text.strip():
            final_text = vlm_text or native_text
            if not vlm_text.strip() and native_text.strip():
                metrics.vlm_fallback_count += 1
            debug = None
        else:
            result = correct_vlm_text_with_native(
                vlm_text,
                native_text,
                config,
            )
            final_text = result["text"]
            metrics.corrected_char_count += result["corrected_char_count"]
            metrics.native_inserted_char_count += result["inserted_char_count"]
            if result["corrected_char_count"] or result["inserted_char_count"]:
                metrics.corrected_block_count += 1
            if result["decision"] in {"keep_vlm_conflict", "keep_vlm_order_conflict"}:
                metrics.skipped_conflict_count += 1
            debug = result if config.debug else None

        if correctable:
            corrected = copy_model_block(block)
            corrected["content"] = final_text
            corrected["source"] = "vlm_native_corrected" if final_text != vlm_text else block.get("source", "vlm")
            if debug is not None:
                corrected["_correction"] = debug
            corrected_blocks.append(corrected)
        text_source = block.get("source", "vlm")
        final_source = "vlm_native_corrected" if correctable and final_text != vlm_text else text_source
        if native_text or vlm_text or final_text:
            compare_records.append(
                {
                    "page_index": context.page_index,
                    "bbox_index": block.get("index", index + 1),
                    "block_type": block_type,
                    "text_source": text_source,
                    "final_source": final_source,
                    "native_text": native_text,
                    "vlm_text": vlm_text,
                    "final_text": final_text,
                }
            )

    return corrected_blocks, metrics, compare_records


def _is_correctable_text_block(block_type: str, block: dict) -> bool:
    if is_textual_type(block_type) and not is_structural_type(block_type):
        return True
    return block_type == "image" and block.get("sub_type") == "text_image" and bool((block.get("content") or "").strip())


def correct_vlm_text_with_native(
    vlm_text: str,
    native_text: str,
    config: NativeCorrectionConfig,
) -> dict:
    """Use native text to correct VLM characters while keeping VLM formatting.

    The output is built from the VLM character stream. Native can only:
    - replace a VLM character when the aligned native character is safer;
    - insert ordinary native text runs when VLM likely missed text.

    Native never contributes its own whitespace or line breaks in this mode.
    """
    if not native_text.strip():
        return _result(vlm_text, "vlm_no_native")

    alignment = align_native_with_vlm(native_text, vlm_text)
    if _has_same_content_different_order(native_text, vlm_text):
        return _result(vlm_text, "keep_vlm_order_conflict", alignment=alignment)

    content_stats = _content_alignment_stats(alignment)
    if content_stats["anchor_coverage"] < config.min_anchor_coverage and not (
        _can_correct_low_anchor_conflicts(content_stats) or _can_fill_large_native_missing(alignment, content_stats, config)
    ):
        return _result(vlm_text, "keep_vlm_conflict", alignment=alignment)

    ops = alignment.operations
    parts: list[str] = []
    corrected = 0
    inserted = 0
    vlm_effective = max(1, effective_token_count(vlm_text))
    # Short captions/speech bubbles often have only a few effective chars. A pure
    # ratio cap would make the total insert budget smaller than missing_max_run,
    # so keep the total budget at least as large as one allowed missing run.
    max_insert_total = max(config.missing_max_run, int(vlm_effective * config.missing_max_total_ratio))

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        if op.op_type in {MATCH, EQUIVALENT}:
            parts.append(op.vlm or op.native)
            idx += 1
            continue
        if op.op_type == CONFLICT:
            replacement = _choose_native_replacement(op.native, op.vlm)
            if replacement is not None:
                parts.append(replacement)
                corrected += 1
            else:
                parts.append(op.vlm)
            idx += 1
            continue
        if op.op_type == NATIVE_MISSING:
            # VLM-only characters stay in output because VLM is the base stream.
            parts.append(op.vlm)
            idx += 1
            continue
        if op.op_type == NATIVE_EXTRA:
            run_start = idx
            while idx < len(ops) and ops[idx].op_type == NATIVE_EXTRA:
                idx += 1
            run = ops[run_start:idx]
            run_text = "".join(item.native for item in run if _can_insert_native_extra(item))
            if _native_extra_exists_as_vlm_missing(ops, run_text, run_start, idx):
                continue
            if _can_insert_native_run(
                run_text,
                inserted,
                max_insert_total,
                ops,
                run_start,
                idx,
                content_stats,
                config,
            ):
                parts.append(run_text)
                inserted += len(run_text)
            continue
        idx += 1

    decision = "native_corrected" if corrected or inserted else "vlm"
    return _result(
        "".join(parts),
        decision,
        alignment=alignment,
        corrected_char_count=corrected,
        inserted_char_count=inserted,
    )


def _choose_native_replacement(native_char: str, vlm_char: str) -> str | None:
    """Return native char when it is safer than the aligned VLM char.

    This is intentionally conservative. We replace only ordinary text, digits, and
    common punctuation. Whitespace and structural symbols stay from VLM.
    """
    if not native_char or not vlm_char:
        return None
    native_type = _simple_char_type(native_char)
    vlm_type = _simple_char_type(vlm_char)
    if native_type == SPACE or vlm_type == SPACE:
        return None
    if native_type == PUNCT and vlm_type == PUNCT:
        return native_char
    if native_type in {LETTER_OR_CJK, DIGIT} and _char_script_group(native_char) == _char_script_group(vlm_char):
        return native_char
    return None


def _content_alignment_stats(alignment) -> dict:
    native_content_count = 0
    anchor_count = 0
    conflict_count = 0
    native_missing_count = 0
    native_extra_count = 0

    for op in alignment.operations:
        native_is_content = _is_text_content_char(op.native)
        vlm_is_content = _is_text_content_char(op.vlm)
        if native_is_content:
            native_content_count += 1
        if op.op_type in {MATCH, EQUIVALENT} and native_is_content:
            anchor_count += 1
        elif op.op_type == CONFLICT and (native_is_content or vlm_is_content):
            conflict_count += 1
        elif op.op_type == NATIVE_MISSING and vlm_is_content:
            native_missing_count += 1
        elif op.op_type == NATIVE_EXTRA and native_is_content:
            native_extra_count += 1

    denominator = max(1, native_content_count)
    return {
        "anchor_coverage": anchor_count / denominator,
        "conflict_ratio": conflict_count / denominator,
        "native_missing_count": native_missing_count,
        "native_extra_count": native_extra_count,
    }


def _can_correct_low_anchor_conflicts(content_stats: dict) -> bool:
    """Allow character correction for short same-shape text with few anchors.

    Comic speech bubbles and sound effects often repeat a small set of characters.
    A phrase like "吧唧，吧唧" vs "吧啷，吧啷" has many aligned conflicts and only
    one repeated stable anchor, so the global anchor ratio is low even though the
    alignment shape is reliable. In this narrow case, native may still correct
    ordinary aligned characters. Any effective insertion/deletion keeps the old
    conservative behavior.
    """
    return (
        content_stats["anchor_coverage"] > 0
        and content_stats["conflict_ratio"] > 0
        and content_stats["native_missing_count"] == 0
        and content_stats["native_extra_count"] == 0
    )


def _can_fill_large_native_missing(alignment, content_stats: dict, config: NativeCorrectionConfig) -> bool:
    return (
        config.allow_missing_fill
        and config.allow_large_missing_fill
        and content_stats["native_extra_count"] > config.missing_max_run
        and _allows_large_missing_fill(content_stats, config)
        and _has_large_fillable_native_run(alignment.operations, config.missing_max_run)
    )


def _has_large_fillable_native_run(ops, short_run_limit: int) -> bool:
    idx = 0
    while idx < len(ops):
        if ops[idx].op_type != NATIVE_EXTRA:
            idx += 1
            continue
        run_start = idx
        while idx < len(ops) and ops[idx].op_type == NATIVE_EXTRA:
            idx += 1
        run_text = "".join(item.native for item in ops[run_start:idx] if _can_insert_native_extra(item))
        if (
            len(run_text) > short_run_limit
            and _has_vlm_anchor_before(ops, run_start)
            and _has_vlm_anchor_after(ops, idx)
        ):
            return True
    return False


def _can_insert_native_run(
    run_text: str,
    inserted: int,
    max_insert_total: int,
    ops,
    run_start: int,
    run_end: int,
    content_stats: dict,
    config: NativeCorrectionConfig,
) -> bool:
    if not config.allow_missing_fill or not run_text:
        return False
    if not _has_vlm_anchor_before(ops, run_start) or not _has_vlm_anchor_after(ops, run_end):
        return False
    if len(run_text) <= config.missing_max_run and inserted + len(run_text) <= max_insert_total:
        return True
    return (
        config.allow_large_missing_fill
        and _allows_large_missing_fill(content_stats, config)
    )


def _allows_large_missing_fill(content_stats: dict, config: NativeCorrectionConfig) -> bool:
    return content_stats["conflict_ratio"] <= min(
        config.max_conflict_ratio,
        config.large_missing_max_conflict_ratio,
    )


def _can_insert_native_extra(op) -> bool:
    return _simple_char_type(op.native) == LETTER_OR_CJK


def _has_same_content_different_order(native_text: str, vlm_text: str) -> bool:
    native_content = _content_text(native_text)
    vlm_content = _content_text(vlm_text)
    return bool(
        native_content
        and vlm_content
        and native_content != vlm_content
        and Counter(native_content) == Counter(vlm_content)
    )


def _native_extra_exists_as_vlm_missing(ops, run_text: str, run_start: int, run_end: int) -> bool:
    if not run_text:
        return False
    idx = 0
    while idx < len(ops):
        if run_start <= idx < run_end or ops[idx].op_type != NATIVE_MISSING:
            idx += 1
            continue
        missing_start = idx
        while idx < len(ops) and ops[idx].op_type == NATIVE_MISSING:
            idx += 1
        if missing_start == run_start:
            continue
        missing_text = "".join(item.vlm for item in ops[missing_start:idx] if _is_text_content_char(item.vlm))
        if missing_text == run_text:
            return True
    return False


def _content_text(text: str) -> str:
    return "".join(char for char in text or "" if _is_text_content_char(char))


def _simple_char_type(char: str) -> str:
    if not char or char.isspace():
        return SPACE
    if char.isdigit():
        return DIGIT
    if _is_common_text_letter(char):
        return LETTER_OR_CJK
    if char.isalpha():
        return CONTROL_OR_UNKNOWN
    if len(char) == 1 and not char.isalnum():
        return PUNCT
    return CONTROL_OR_UNKNOWN


def _is_text_content_char(char: str) -> bool:
    return _simple_char_type(char) in {LETTER_OR_CJK, DIGIT}


def _char_script_group(char: str) -> str:
    if not char:
        return CONTROL_OR_UNKNOWN
    if char.isdigit():
        return DIGIT
    if _is_cjk(char):
        return "CJK"
    if _is_latin(char):
        return "LATIN"
    if _is_hiragana_or_katakana(char):
        return "KANA"
    if _is_hangul(char):
        return "HANGUL"
    if len(char) == 1 and not char.isalnum():
        return PUNCT
    return CONTROL_OR_UNKNOWN


def _is_common_text_letter(char: str) -> bool:
    return _is_cjk(char) or _is_latin(char) or _is_hiragana_or_katakana(char) or _is_hangul(char)


def _is_latin(char: str) -> bool:
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def _is_hiragana_or_katakana(char: str) -> bool:
    return (
        "\u3040" <= char <= "\u309f"
        or "\u30a0" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        or "\uff66" <= char <= "\uff9f"
    )


def _is_hangul(char: str) -> bool:
    return "\uac00" <= char <= "\ud7af" or "\u1100" <= char <= "\u11ff"


def _has_vlm_anchor_before(ops, index: int) -> bool:
    for op in reversed(ops[:index]):
        if op.op_type in {MATCH, EQUIVALENT} and _is_text_content_char(op.vlm):
            return True
    return False


def _has_vlm_anchor_after(ops, index: int) -> bool:
    for op in ops[index:]:
        if op.op_type in {MATCH, EQUIVALENT} and _is_text_content_char(op.vlm):
            return True
    return False


def _is_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )


def _result(
    text: str,
    decision: str,
    *,
    alignment=None,
    corrected_char_count: int = 0,
    inserted_char_count: int = 0,
) -> dict:
    debug = alignment.to_debug() if alignment is not None else {}
    return {
        "text": text,
        "decision": decision,
        "alignment": debug,
        "corrected_char_count": corrected_char_count,
        "inserted_char_count": inserted_char_count,
    }
