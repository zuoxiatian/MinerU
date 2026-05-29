# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm.model_output_to_middle_json import (
    blocks_to_page_info,
    finalize_middle_json as vlm_finalize_middle_json,
)
from mineru.utils.pdfium_guard import pdfium_guard
from mineru.version import __version__


def init_middle_json():
    """初始化 fusion 后端的 middle_json 顶层结构。"""
    return {
        "pdf_info": [],
        "_backend": "vlm_fusion",
        "_version_name": __version__,
        "_fusion_metrics": [],
    }


def append_page_fusion_to_middle_json(
    middle_json,
    fused_blocks_list,
    metrics_list,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    progress_bar=None,
):
    """把一批 fused pages 追加到 middle_json。

    fused_blocks 已经完成内容融合；这里复用 VLM 后端的 blocks_to_page_info，
    继续生成 MinerU 下游消费的页面结构、图片资源和统计信息。
    """
    for offset, (page_blocks, metrics, image_dict) in enumerate(
        zip(fused_blocks_list, metrics_list, images_list)
    ):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index]
        page_info = blocks_to_page_info(page_blocks, image_dict, page, image_writer, page_index)
        middle_json["pdf_info"].append(page_info)
        middle_json["_fusion_metrics"].append(metrics.to_dict())
        if progress_bar is not None:
            progress_bar.update(1)


def finalize_middle_json(pdf_info_list):
    """复用 VLM 后端的最终收尾逻辑，例如 span/block 后处理和页面级字段补齐。"""
    vlm_finalize_middle_json(pdf_info_list)
