# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_fusion.bbox import pdf_to_unit_bbox
from mineru.backend.vlm_fusion.schemas import BBox, VisualTextCandidate


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
    if block_type == "list_item":
        return "list"
    return block_type


def is_textual_type(block_type: str) -> bool:
    return normalize_block_type(block_type) in TEXTUAL_TYPES


def is_structural_type(block_type: str) -> bool:
    return normalize_block_type(block_type) in STRUCTURAL_TYPES


def copy_model_block(block: dict) -> dict:
    copied = {
        key: value
        for key, value in dict(block).items()
        if key not in {"scored"}
    }
    copied["type"] = normalize_block_type(copied.get("type", "text"))
    return copied


def build_native_text_block(source_block: dict, content: str, debug: dict | None = None) -> dict:
    block = copy_model_block(source_block)
    block["content"] = content
    block["source"] = "pdf_native"
    block["_content_locked"] = True
    if debug is not None:
        block["_fusion"] = debug
    return block


def build_vlm_fallback_block(source_block: dict, debug: dict | None = None) -> dict:
    block = copy_model_block(source_block)
    block["source"] = "vlm"
    if debug is not None:
        block["_fusion"] = debug
    return block


def build_supplement_block(
    candidate: VisualTextCandidate,
    page_width: int,
    page_height: int,
    next_index: int,
    debug: dict | None = None,
) -> dict:
    block = {
        "type": normalize_block_type(candidate.block_type) if candidate.block_type in {"title", "text"} else "text",
        "bbox": pdf_to_unit_bbox(candidate.bbox, page_width, page_height),
        "angle": 0,
        "content": candidate.content,
        "index": next_index,
        "source": "visual_supplement",
    }
    if debug is not None:
        block["_fusion"] = debug
    return block

def block_pdf_bbox(block: dict, page_width: int, page_height: int) -> BBox:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return [
        float(bbox[0]) * page_width,
        float(bbox[1]) * page_height,
        float(bbox[2]) * page_width,
        float(bbox[3]) * page_height,
    ]

