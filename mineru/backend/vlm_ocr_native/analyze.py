# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
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
from mineru.backend.vlm_ocr_native.native_text import extract_native_spans
from mineru.backend.vlm_ocr_native.visual_candidates import collect_visual_text_candidates
from mineru.backend.vlm_ocr_native.vlm_runner import (
    aio_batch_content_extract_from_layouts,
    aio_batch_layout_detect,
    batch_content_extract_from_layouts,
    batch_layout_detect,
)
from mineru.backend.vlm_ocr_native.config import get_correction_config
from mineru.backend.vlm_ocr_native.correction import correct_page
from mineru.backend.vlm_ocr_native.gap_recognizer import (
    aio_recognize_native_correction_gaps,
    detect_native_correction_gaps,
    recognize_native_correction_gaps,
)
from mineru.backend.vlm_ocr_native.middle_json import (
    append_page_correction_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.vlm_ocr_native.ocr_text_regions import extract_ocr_text_blocks
from mineru.backend.vlm_ocr_native.page_classifier import analyze_layout_flow, classify_layout_blocks
from mineru.backend.vlm_ocr_native.schemas import PageCorrectionContext
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


TYPE_REGION_BLOCK_TYPES = {"header", "footer", "page_number"}


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is None:
            continue
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


def _correct_window_pages(
    pdf_doc,
    window_start: int,
    layout_results,
    vlm_results,
    native_spans_list,
    native_gaps_list=None,
):
    config = get_correction_config()
    native_gaps_list = native_gaps_list or [[] for _ in native_spans_list]
    corrected_blocks_list = []
    metrics_list = []
    compare_records_list = []
    for offset, (layout_blocks, vlm_blocks, native_spans, native_gaps) in enumerate(
        zip(layout_results, vlm_results, native_spans_list, native_gaps_list)
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
            native_gaps=native_gaps,
        )
        corrected_blocks, metrics, compare_records = correct_page(context, config)
        corrected_blocks_list.append(corrected_blocks)
        metrics_list.append(metrics)
        compare_records_list.append(compare_records)
    return corrected_blocks_list, metrics_list, compare_records_list


def _native_gap_to_dict(gap):
    return {
        "bbox": [round(value, 3) for value in gap.bbox],
        "block_index": gap.block_index,
        "left_span_uid": gap.left_span_uid,
        "right_span_uid": gap.right_span_uid,
        "max_chars": gap.max_chars,
        "content": gap.content,
        "source": gap.source,
    }


def _recognize_window_native_gaps(
    predictor,
    images_list,
    vlm_results,
    native_spans_list,
    page_modes=None,
):
    config = get_correction_config()
    page_modes = page_modes or ["vlm"] * len(vlm_results)
    native_gaps_list = []
    for image_dict, vlm_blocks, native_spans, page_mode in zip(images_list, vlm_results, native_spans_list, page_modes):
        if page_mode == "ocr":
            native_gaps_list.append([])
            continue
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        gaps = detect_native_correction_gaps(
            vlm_block_dicts,
            native_spans,
            pdf_width,
            pdf_height,
            config,
        )
        native_gaps_list.append(
            recognize_native_correction_gaps(predictor, page_image, scale, gaps, config)
        )
    return native_gaps_list


async def _aio_recognize_window_native_gaps(
    predictor,
    images_list,
    vlm_results,
    native_spans_list,
    page_modes=None,
):
    config = get_correction_config()
    page_modes = page_modes or ["vlm"] * len(vlm_results)
    native_gaps_list = []
    for image_dict, vlm_blocks, native_spans, page_mode in zip(images_list, vlm_results, native_spans_list, page_modes):
        if page_mode == "ocr":
            native_gaps_list.append([])
            continue
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        gaps = detect_native_correction_gaps(
            vlm_block_dicts,
            native_spans,
            pdf_width,
            pdf_height,
            config,
        )
        native_gaps_list.append(
            await aio_recognize_native_correction_gaps(predictor, page_image, scale, gaps, config)
        )
    return native_gaps_list


def _extend_model_output(
    model_output,
    layout_results,
    vlm_results,
    corrected_blocks_list,
    compare_records_list,
    metrics_list,
    native_gaps_list=None,
    page_classifications=None,
    page_modes=None,
    ocr_debug_list=None,
    page_layouts=None,
):
    native_gaps_list = native_gaps_list or [[] for _ in corrected_blocks_list]
    page_classifications = page_classifications or [None] * len(corrected_blocks_list)
    page_modes = page_modes or ["vlm"] * len(corrected_blocks_list)
    ocr_debug_list = ocr_debug_list or [None] * len(corrected_blocks_list)
    page_layouts = page_layouts or [None] * len(corrected_blocks_list)
    for layout, vlm, corrected, compare_records, native_gaps, metrics, classification, page_mode, ocr_debug, page_layout in zip(
        layout_results,
        vlm_results,
        corrected_blocks_list,
        compare_records_list,
        native_gaps_list,
        metrics_list,
        page_classifications,
        page_modes,
        ocr_debug_list,
        page_layouts,
    ):
        item = {
            "model_output_version": "vlm_ocr_native_layout_v1",
            "layout": [dict(block) for block in layout],
            "vlm": [dict(block) for block in vlm],
            "corrected": corrected,
            "bbox_text_compare": compare_records,
            "native_gaps": [_native_gap_to_dict(gap) for gap in native_gaps],
            "metrics": metrics.to_dict(),
        }
        item["page_mode"] = page_mode
        if classification is not None:
            item["page_classification"] = classification.__dict__
        if page_layout is not None:
            item["page_layout"] = page_layout.__dict__
        if ocr_debug is not None:
            item["ocr"] = ocr_debug
        model_output.append(item)


def _collect_vlm_type_regions(
    layout_blocks,
    page_width: int,
    page_height: int,
) -> list[dict]:
    regions = []
    for block in layout_blocks or []:
        block_dict = dict(block)
        block_type = str(block_dict.get("type") or "").lower()
        if block_type not in TYPE_REGION_BLOCK_TYPES:
            continue
        bbox = block_dict.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            pdf_bbox = [
                x0 * page_width,
                y0 * page_height,
                x1 * page_width,
                y1 * page_height,
            ]
        else:
            pdf_bbox = [x0, y0, x1, y1]
        regions.append(
            {
                "type": block_type,
                "bbox": pdf_bbox,
                "source": "vlm_layout",
            }
        )
    return regions


def _build_window_vlm_or_ocr_results(
    predictor,
    images_list,
    layout_results,
    image_analysis: bool,
    ocr_language: str | None,
):
    vlm_results = [None] * len(images_list)
    page_modes = ["vlm"] * len(images_list)
    page_classifications = []
    page_layouts = []
    ocr_debug_list = [None] * len(images_list)
    normal_indices = []
    reading_order = os.getenv("MINERU_VLM_OCR_NATIVE_READING_ORDER", "left_to_right")

    for index, (image_dict, layout_blocks) in enumerate(zip(images_list, layout_results)):
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        classification = classify_layout_blocks(
            layout_blocks,
            page_width=pdf_width,
            page_height=pdf_height,
        )
        page_layout = analyze_layout_flow(
            layout_blocks,
            page_width=pdf_width,
            page_height=pdf_height,
            reading_order=reading_order,
        )
        page_classifications.append(classification)
        page_layouts.append(page_layout)
        if classification.is_comic_like:
            page_modes[index] = "ocr"
            type_regions = _collect_vlm_type_regions(layout_blocks, pdf_width, pdf_height)
            blocks, debug = extract_ocr_text_blocks(
                page_image,
                scale=scale,
                page_width=pdf_width,
                page_height=pdf_height,
                language=ocr_language,
                layout_hint=page_layout.__dict__,
                type_regions=type_regions,
            )
            vlm_results[index] = blocks
            ocr_debug_list[index] = debug
        else:
            normal_indices.append(index)

    if normal_indices:
        normal_images = [images_list[index]["img_pil"] for index in normal_indices]
        normal_layouts = [layout_results[index] for index in normal_indices]
        normal_results = batch_content_extract_from_layouts(
            predictor,
            normal_images,
            normal_layouts,
            image_analysis=image_analysis,
        )
        for index, result in zip(normal_indices, normal_results):
            vlm_results[index] = result

    return vlm_results, page_modes, page_classifications, ocr_debug_list, page_layouts


async def _aio_build_window_vlm_or_ocr_results(
    predictor,
    images_list,
    layout_results,
    image_analysis: bool,
    ocr_language: str | None,
):
    vlm_results = [None] * len(images_list)
    page_modes = ["vlm"] * len(images_list)
    page_classifications = []
    page_layouts = []
    ocr_debug_list = [None] * len(images_list)
    normal_indices = []
    reading_order = os.getenv("MINERU_VLM_OCR_NATIVE_READING_ORDER", "left_to_right")

    for index, (image_dict, layout_blocks) in enumerate(zip(images_list, layout_results)):
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        classification = classify_layout_blocks(
            layout_blocks,
            page_width=pdf_width,
            page_height=pdf_height,
        )
        page_layout = analyze_layout_flow(
            layout_blocks,
            page_width=pdf_width,
            page_height=pdf_height,
            reading_order=reading_order,
        )
        page_classifications.append(classification)
        page_layouts.append(page_layout)
        if classification.is_comic_like:
            page_modes[index] = "ocr"
            type_regions = _collect_vlm_type_regions(layout_blocks, pdf_width, pdf_height)
            blocks, debug = await asyncio.to_thread(
                extract_ocr_text_blocks,
                page_image,
                scale=scale,
                page_width=pdf_width,
                page_height=pdf_height,
                language=ocr_language,
                layout_hint=page_layout.__dict__,
                type_regions=type_regions,
            )
            vlm_results[index] = blocks
            ocr_debug_list[index] = debug
        else:
            normal_indices.append(index)

    if normal_indices:
        normal_images = [images_list[index]["img_pil"] for index in normal_indices]
        normal_layouts = [layout_results[index] for index in normal_indices]
        normal_results = await aio_batch_content_extract_from_layouts(
            predictor,
            normal_images,
            normal_layouts,
            image_analysis=image_analysis,
        )
        for index, result in zip(normal_indices, normal_results):
            vlm_results[index] = result

    return vlm_results, page_modes, page_classifications, ocr_debug_list, page_layouts


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
    """Synchronous entry for VLM layout + OCR comic-page native correction."""
    ocr_language = kwargs.pop("ocr_language", None)
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
            f"VLM OCR native processing-window run. page_count={page_count}, "
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
                        (
                            vlm_results,
                            page_modes,
                            page_classifications,
                            ocr_debug_list,
                            page_layouts,
                        ) = _build_window_vlm_or_ocr_results(
                            predictor,
                            images_list,
                            layout_results,
                            image_analysis,
                            ocr_language,
                        )
                    native_spans_list = _extract_window_native_spans(
                        pdf_doc,
                        window_start,
                        len(images_pil_list),
                    )
                    with predictor_execution_guard(predictor):
                        native_gaps_list = _recognize_window_native_gaps(
                            predictor,
                            images_list,
                            vlm_results,
                            native_spans_list,
                            page_modes,
                        )
                    corrected_blocks_list, metrics_list, compare_records_list = _correct_window_pages(
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                        native_gaps_list,
                    )
                    _extend_model_output(
                        model_output,
                        layout_results,
                        vlm_results,
                        corrected_blocks_list,
                        compare_records_list,
                        metrics_list,
                        native_gaps_list,
                        page_classifications,
                        page_modes,
                        ocr_debug_list,
                        page_layouts,
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
                f"vlm ocr native infer finished, cost: {infer_time}, "
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
    """Async entry for VLM layout + OCR comic-page native correction."""
    ocr_language = kwargs.pop("ocr_language", None)
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
                    (
                        vlm_results,
                        page_modes,
                        page_classifications,
                        ocr_debug_list,
                        page_layouts,
                    ) = await _aio_build_window_vlm_or_ocr_results(
                        predictor,
                        images_list,
                        layout_results,
                        image_analysis,
                        ocr_language,
                    )
                    native_spans_list = await asyncio.to_thread(
                        _extract_window_native_spans,
                        pdf_doc,
                        window_start,
                        len(images_pil_list),
                    )
                    native_gaps_list = await _aio_recognize_window_native_gaps(
                        predictor,
                        images_list,
                        vlm_results,
                        native_spans_list,
                        page_modes,
                    )
                    corrected_blocks_list, metrics_list, compare_records_list = await asyncio.to_thread(
                        _correct_window_pages,
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                        native_gaps_list,
                    )
                    _extend_model_output(
                        model_output,
                        layout_results,
                        vlm_results,
                        corrected_blocks_list,
                        compare_records_list,
                        metrics_list,
                        native_gaps_list,
                        page_classifications,
                        page_modes,
                        ocr_debug_list,
                        page_layouts,
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
                f"vlm ocr native async infer finished, cost: {infer_time}, "
                f"speed: {round(page_count / infer_time, 3)} page/s"
            )
        await asyncio.to_thread(finalize_middle_json, middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, model_output
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
