# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_ocr_native.bbox import pdf_to_unit_bbox
from mineru.backend.vlm_ocr_native.schemas import BBox, VisualTextCandidate


TEXTUAL_TYPES = {
    "text",
    "title",
    "ref_text",
    "phonetic",
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "list",
    "list_item",
    "image_caption",
    "table_caption",
    "code_caption",
    "image_footnote",
    "table_footnote",
}

STRUCTURAL_TYPES = {
    "table",
    "image",
    "image_block",
    "chart",
    "equation",
    "code",
    "algorithm",
}


def normalize_block_type(block_type: str) -> str:
    """把模型输出中的细分类 block type 归一到 middle_json 期望的类型。"""
    if block_type == "list_item":
        return "list"
    return block_type


def is_textual_type(block_type: str) -> bool:
    """判断 block 是否属于可被 PDF 原生文本替换/补充的文本类区域。"""
    return normalize_block_type(block_type) in TEXTUAL_TYPES


def is_structural_type(block_type: str) -> bool:
    """判断 block 是否属于表格、图片、公式、代码等结构类区域。"""
    return normalize_block_type(block_type) in STRUCTURAL_TYPES


def copy_model_block(block: dict) -> dict:
    """复制模型 block，并移除融合输出不需要的临时字段。"""
    copied = {
        key: value
        for key, value in dict(block).items()
        if key not in {"scored"}
    }
    copied["type"] = normalize_block_type(copied.get("type", "text"))
    return copied


def build_native_text_block(source_block: dict, content: str, debug: dict | None = None) -> dict:
    """基于 VLM block 的几何信息构造 PDF 原生文本锁定块。"""
    block = copy_model_block(source_block)
    block["content"] = content
    block["source"] = "pdf_native"
    block["_content_locked"] = True
    if debug is not None:
        block["_correction"] = debug
    return block


def build_vlm_fallback_block(source_block: dict, debug: dict | None = None) -> dict:
    """构造 VLM fallback 块，用于原生文本不可靠的场景。"""
    block = copy_model_block(source_block)
    block["source"] = "vlm"
    if debug is not None:
        block["_correction"] = debug
    return block


def build_supplement_block(
    candidate: VisualTextCandidate,
    page_width: int,
    page_height: int,
    next_index: int,
    debug: dict | None = None,
) -> dict:
    """把未被原生文本覆盖的 VLM 文本候选构造成补充 block。"""
    block = {
        "type": normalize_block_type(candidate.block_type) if candidate.block_type in {"title", "text"} else "text",
        "bbox": pdf_to_unit_bbox(candidate.bbox, page_width, page_height),
        "angle": 0,
        "content": candidate.content,
        "index": next_index,
        "source": "visual_supplement",
    }
    if debug is not None:
        block["_correction"] = debug
    return block

def block_pdf_bbox(block: dict, page_width: int, page_height: int) -> BBox:
    """把 block 的归一化 bbox 转成 PDF 页面坐标。"""
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return [
        float(bbox[0]) * page_width,
        float(bbox[1]) * page_height,
        float(bbox[2]) * page_width,
        float(bbox[3]) * page_height,
    ]
