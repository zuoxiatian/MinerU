# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import time

import pypdfium2 as pdfium
from loguru import logger
from tqdm import tqdm

from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from mineru.backend.vlm.vlm_analyze import (
    ModelSingleton,
    _maybe_enable_serial_execution,
    predictor_execution_guard,
)
from mineru.backend.vlm_fusion.analyze import (
    _close_images,
    _extract_window_native_spans,
    _page_size,
)
from mineru.backend.vlm_fusion.visual_candidates import collect_visual_text_candidates
from mineru.backend.vlm_fusion.vlm_runner import (
    aio_batch_content_extract_from_layouts,
    aio_batch_layout_detect,
    batch_content_extract_from_layouts,
    batch_layout_detect,
)
from mineru.backend.vlm_native_correction.config import get_correction_config
from mineru.backend.vlm_native_correction.correction import correct_page
from mineru.backend.vlm_native_correction.middle_json import (
    append_page_correction_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.vlm_native_correction.schemas import PageCorrectionContext
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.config_reader import get_processing_window_size
from mineru.utils.enum_class import ImageType
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)


def _correct_window_pages(
    pdf_doc,
    window_start: int,
    layout_results,
    vlm_results,
    native_spans_list,
):
    config = get_correction_config()
    corrected_blocks_list = []
    metrics_list = []
    compare_records_list = []
    for offset, (layout_blocks, vlm_blocks, native_spans) in enumerate(
        zip(layout_results, vlm_results, native_spans_list)
    ):
        page_index = window_start + offset
        width, height = _page_size(pdf_doc, page_index)
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        layout_block_dicts = [dict(block) for block in layout_blocks]
        for block_index, block in enumerate(layout_block_dicts):
            block.setdefault("index", block_index + 1)

        context = PageCorrectionContext(
            page_index=page_index,
            page_width=width,
            page_height=height,
            layout_blocks=layout_block_dicts,
            vlm_blocks=vlm_block_dicts,
            native_spans=native_spans,
            visual_candidates=collect_visual_text_candidates(vlm_block_dicts, width, height),
        )
        corrected_blocks, metrics, compare_records = correct_page(context, config)
        corrected_blocks_list.append(corrected_blocks)
        metrics_list.append(metrics)
        compare_records_list.append(compare_records)
    return corrected_blocks_list, metrics_list, compare_records_list


def _extend_model_output(
    model_output,
    layout_results,
    vlm_results,
    corrected_blocks_list,
    compare_records_list,
    metrics_list,
):
    model_output.extend(
        {
            "layout": [dict(block) for block in layout],
            "vlm": [dict(block) for block in vlm],
            "corrected": corrected,
            "bbox_text_compare": compare_records,
            "metrics": metrics.to_dict(),
        }
        for layout, vlm, corrected, compare_records, metrics in zip(
            layout_results,
            vlm_results,
            corrected_blocks_list,
            compare_records_list,
            metrics_list,
        )
    )


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor=None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    """Synchronous entry for VLM-primary native correction."""
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    model_output = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        logger.info(
            f"VLM native correction processing-window run. page_count={page_count}, "
            f"window_size={effective_window_size or 0}"
        )
        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_start in range(0, page_count, effective_window_size):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = load_images_from_pdf_doc(
                    pdf_doc,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                    pdf_bytes=pdf_bytes,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    with predictor_execution_guard(predictor):
                        layout_results = batch_layout_detect(predictor, images_pil_list)
                        vlm_results = batch_content_extract_from_layouts(
                            predictor,
                            images_pil_list,
                            layout_results,
                            image_analysis=image_analysis,
                        )
                    native_spans_list = _extract_window_native_spans(
                        pdf_doc,
                        window_start,
                        len(images_pil_list),
                    )
                    corrected_blocks_list, metrics_list, compare_records_list = _correct_window_pages(
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                    )
                    _extend_model_output(
                        model_output,
                        layout_results,
                        vlm_results,
                        corrected_blocks_list,
                        compare_records_list,
                        metrics_list,
                    )
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(progress_bar, last_append_end_time, now=time.time())
                    append_page_correction_to_middle_json(
                        middle_json,
                        corrected_blocks_list,
                        metrics_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"vlm native correction infer finished, cost: {infer_time}, "
                f"speed: {round(page_count / infer_time, 3)} page/s"
            )
        finalize_middle_json(middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, model_output
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor=None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    """Async entry for VLM-primary native correction."""
    if predictor is None:
        from mineru.backend.vlm.vlm_analyze import _get_model_async

        predictor = await _get_model_async(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    model_output = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_start in range(0, page_count, effective_window_size):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    layout_results = await aio_batch_layout_detect(predictor, images_pil_list)
                    vlm_results = await aio_batch_content_extract_from_layouts(
                        predictor,
                        images_pil_list,
                        layout_results,
                        image_analysis=image_analysis,
                    )
                    native_spans_list = await asyncio.to_thread(
                        _extract_window_native_spans,
                        pdf_doc,
                        window_start,
                        len(images_pil_list),
                    )
                    corrected_blocks_list, metrics_list, compare_records_list = await asyncio.to_thread(
                        _correct_window_pages,
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                    )
                    _extend_model_output(
                        model_output,
                        layout_results,
                        vlm_results,
                        corrected_blocks_list,
                        compare_records_list,
                        metrics_list,
                    )
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(progress_bar, last_append_end_time, now=time.time())
                    append_page_correction_to_middle_json(
                        middle_json,
                        corrected_blocks_list,
                        metrics_list,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"vlm native correction async infer finished, cost: {infer_time}, "
                f"speed: {round(page_count / infer_time, 3)} page/s"
            )
        await asyncio.to_thread(finalize_middle_json, middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, model_output
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
