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
    _get_model_async,
    _maybe_enable_serial_execution,
    aio_predictor_execution_guard,
    predictor_execution_guard,
)
from mineru.backend.vlm_fusion.config import get_fusion_config
from mineru.backend.vlm_fusion.fusion import fuse_page
from mineru.backend.vlm_fusion.middle_json import (
    append_page_fusion_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.vlm_fusion.native_text import extract_native_spans
from mineru.backend.vlm_fusion.schemas import PageFusionContext
from mineru.backend.vlm_fusion.visual_candidates import collect_visual_text_candidates
from mineru.backend.vlm_fusion.vlm_runner import (
    aio_batch_content_extract_from_layouts,
    aio_batch_layout_detect,
    batch_content_extract_from_layouts,
    batch_layout_detect,
)
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
    pdfium_guard,
)


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _page_size(pdf_doc, page_index: int) -> tuple[int, int]:
    with pdfium_guard():
        page = pdf_doc[page_index]
        return tuple(map(int, page.get_size()))


def _extract_window_native_spans(pdf_doc, page_start: int, count: int):
    native_spans_list = []
    for offset in range(count):
        with pdfium_guard():
            page = pdf_doc[page_start + offset]
        native_spans_list.append(extract_native_spans(page))
    return native_spans_list


def _fuse_window_pages(
    pdf_doc,
    window_start: int,
    layout_results,
    vlm_results,
    native_spans_list,
):
    config = get_fusion_config()
    fused_blocks_list = []
    metrics_list = []
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
        visual_candidates = collect_visual_text_candidates(vlm_block_dicts, width, height)
        context = PageFusionContext(
            page_index=page_index,
            page_width=width,
            page_height=height,
            layout_blocks=layout_block_dicts,
            vlm_blocks=vlm_block_dicts,
            native_spans=native_spans,
            visual_candidates=visual_candidates,
        )
        fused_blocks, metrics = fuse_page(context, config)
        fused_blocks_list.append(fused_blocks)
        metrics_list.append(metrics)
    return fused_blocks_list, metrics_list


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
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f"VLM fusion processing-window run. page_count={page_count}, "
            f"window_size={configured_window_size}, total_windows={total_windows}"
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
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
                    logger.info(
                        f"VLM fusion processing window {window_index + 1}/{total_windows}: "
                        f"pages {window_start + 1}-{window_end + 1}/{page_count} "
                        f"({len(images_pil_list)} pages)"
                    )
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
                    fused_blocks_list, metrics_list = _fuse_window_pages(
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                    )
                    model_output.extend(
                        {
                            "layout": [dict(block) for block in layout],
                            "vlm": [dict(block) for block in vlm],
                            "fused": fused,
                            "metrics": metrics.to_dict(),
                        }
                        for layout, vlm, fused, metrics in zip(
                            layout_results,
                            vlm_results,
                            fused_blocks_list,
                            metrics_list,
                        )
                    )
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(progress_bar, last_append_end_time, now=time.time())
                    append_page_fusion_to_middle_json(
                        middle_json,
                        fused_blocks_list,
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
                f"vlm fusion infer finished, cost: {infer_time}, "
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
    if predictor is None:
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
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f"VLM fusion async processing-window run. page_count={page_count}, "
            f"window_size={configured_window_size}, total_windows={total_windows}"
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f"VLM fusion async processing window {window_index + 1}/{total_windows}: "
                        f"pages {window_start + 1}-{window_end + 1}/{page_count} "
                        f"({len(images_pil_list)} pages)"
                    )
                    async with aio_predictor_execution_guard(predictor):
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
                    fused_blocks_list, metrics_list = await asyncio.to_thread(
                        _fuse_window_pages,
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                    )
                    model_output.extend(
                        {
                            "layout": [dict(block) for block in layout],
                            "vlm": [dict(block) for block in vlm],
                            "fused": fused,
                            "metrics": metrics.to_dict(),
                        }
                        for layout, vlm, fused, metrics in zip(
                            layout_results,
                            vlm_results,
                            fused_blocks_list,
                            metrics_list,
                        )
                    )
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(progress_bar, last_append_end_time, now=time.time())
                    append_page_fusion_to_middle_json(
                        middle_json,
                        fused_blocks_list,
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
                f"vlm fusion async infer finished, cost: {infer_time}, "
                f"speed: {round(page_count / infer_time, 3)} page/s"
            )
        await asyncio.to_thread(finalize_middle_json, middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, model_output
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
