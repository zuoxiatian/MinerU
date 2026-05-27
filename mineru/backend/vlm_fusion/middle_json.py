# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm.model_output_to_middle_json import (
    blocks_to_page_info,
    finalize_middle_json as vlm_finalize_middle_json,
)
from mineru.utils.pdfium_guard import pdfium_guard
from mineru.version import __version__


def init_middle_json():
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
    vlm_finalize_middle_json(pdf_info_list)

