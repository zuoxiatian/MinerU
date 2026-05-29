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
from mineru.backend.vlm_fusion.gap_recognizer import (
    aio_recognize_native_gaps,
    detect_native_gaps,
    recognize_native_gaps,
)
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
    """释放窗口内渲染出来的 PIL 图像，避免长文档处理时占用过多内存。"""
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _page_size(pdf_doc, page_index: int) -> tuple[int, int]:
    """读取 PDF 页面的原始坐标尺寸，后续用于 VLM 归一化 bbox 与 PDF bbox 互转。"""
    with pdfium_guard():
        page = pdf_doc[page_index]
        return tuple(map(int, page.get_size()))


def _extract_window_native_spans(pdf_doc, page_start: int, count: int):
    """抽取一个窗口内每页的 PDF 原生文本 span。

    返回值与窗口页序一一对应：第 N 个元素就是窗口内第 N 页的原生文本列表。
    原生文本后续会作为普通文本块的优先来源。
    """
    native_spans_list = []
    for offset in range(count):
        with pdfium_guard():
            page = pdf_doc[page_start + offset]
        native_spans_list.append(extract_native_spans(page))
    return native_spans_list


def _native_gap_to_dict(gap):
    """把 NativeGap 转成可序列化字典，放入调试用的 model_output。"""
    return {
        "bbox": [round(value, 3) for value in gap.bbox],
        "block_index": gap.block_index,
        "left_span_uid": gap.left_span_uid,
        "right_span_uid": gap.right_span_uid,
        "content": gap.content,
        "source": gap.source,
    }


def _fuse_window_pages(
    pdf_doc,
    window_start: int,
    layout_results,
    vlm_results,
    native_spans_list,
    native_gaps_list=None,
):
    """融合一个处理窗口内的所有页面。

    这里把 VLM block、PDF 原生文本、视觉补充候选和 gap 识别结果组装成
    PageFusionContext，然后交给 fusion.py::fuse_page 做页面级决策。
    """
    config = get_fusion_config()
    native_gaps_list = native_gaps_list or [[] for _ in native_spans_list]
    fused_blocks_list = []
    metrics_list = []
    for offset, (layout_blocks, vlm_blocks, native_spans, native_gaps) in enumerate(
        zip(layout_results, vlm_results, native_spans_list, native_gaps_list)
    ):
        page_index = window_start + offset
        width, height = _page_size(pdf_doc, page_index)

        # VLM 输出对象可能是模型自定义 block，这里统一复制成 dict，
        # 并补齐 index，保证后续 gap 匹配和调试输出有稳定块编号。
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        layout_block_dicts = [dict(block) for block in layout_blocks]
        for block_index, block in enumerate(layout_block_dicts):
            block.setdefault("index", block_index + 1)
        visual_candidates = collect_visual_text_candidates(vlm_block_dicts, width, height)

        # PageFusionContext 是页面融合的完整输入快照：
        # layout blocks 目前主要用于统计和调试，融合决策以 vlm_blocks 为主。
        context = PageFusionContext(
            page_index=page_index,
            page_width=width,
            page_height=height,
            layout_blocks=layout_block_dicts,
            vlm_blocks=vlm_block_dicts,
            native_spans=native_spans,
            visual_candidates=visual_candidates,
            native_gaps=native_gaps,
        )
        fused_blocks, metrics = fuse_page(context, config)
        fused_blocks_list.append(fused_blocks)
        metrics_list.append(metrics)
    return fused_blocks_list, metrics_list


def _recognize_window_native_gaps(
    predictor,
    images_list,
    vlm_results,
    native_spans_list,
):
    """同步识别窗口内每页的原生文本缺口。

    原生 PDF 文本有时会把可见字符拆成多个 span，中间的图片化字符或异常字符
    可能没有出现在原生文本中。这里先按几何距离检测可疑空隙，再裁图交给 VLM
    识别，识别结果会在融合时插回相邻 span 之间。
    """
    config = get_fusion_config()
    native_gaps_list = []
    for image_dict, vlm_blocks, native_spans in zip(images_list, vlm_results, native_spans_list):
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        gaps = detect_native_gaps(
            vlm_block_dicts,
            native_spans,
            pdf_width,
            pdf_height,
            config,
        )
        native_gaps_list.append(
            recognize_native_gaps(predictor, page_image, scale, gaps, config)
        )
    return native_gaps_list


async def _aio_recognize_window_native_gaps(
    predictor,
    images_list,
    vlm_results,
    native_spans_list,
):
    """异步版本的窗口原生文本缺口识别，逻辑与同步函数保持一致。"""
    config = get_fusion_config()
    native_gaps_list = []
    for image_dict, vlm_blocks, native_spans in zip(images_list, vlm_results, native_spans_list):
        page_image = image_dict["img_pil"]
        width, height = page_image.size
        scale = image_dict["scale"]
        pdf_width = int(width / scale)
        pdf_height = int(height / scale)
        vlm_block_dicts = [dict(block) for block in vlm_blocks]
        for block_index, block in enumerate(vlm_block_dicts):
            block.setdefault("index", block_index + 1)
        gaps = detect_native_gaps(
            vlm_block_dicts,
            native_spans,
            pdf_width,
            pdf_height,
            config,
        )
        native_gaps_list.append(
            await aio_recognize_native_gaps(predictor, page_image, scale, gaps, config)
        )
    return native_gaps_list


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
    """VLM fusion 同步入口。

    完整流程：
    1. 初始化模型和 PDF 文档。
    2. 按窗口渲染页面，减少一次性加载大文档造成的内存压力。
    3. 先跑 VLM layout，再按 layout 做内容识别。
    4. 抽取 PDF 原生文本，并识别原生文本之间的可疑缺口。
    5. 调用 fuse_page 产出 fused blocks。
    6. 转成 middle_json，同时返回便于排查的 model_output。
    """
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

                # 每个窗口独立渲染、推理、融合、释放图片，避免长文档处理时
                # PIL 图像和模型中间结果长期堆积。
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
                        # layout_results 是页面区域结构；vlm_results 是同一批区域
                        # 经过内容识别和后处理后的 block 列表。
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
                    with predictor_execution_guard(predictor):
                        # gap 识别依赖 VLM block 的 bbox 和 PDF 原生 span 的 bbox，
                        # 识别结果只用于补齐原生文本中的小缺口。
                        native_gaps_list = _recognize_window_native_gaps(
                            predictor,
                            images_list,
                            vlm_results,
                            native_spans_list,
                        )
                    fused_blocks_list, metrics_list = _fuse_window_pages(
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                        native_gaps_list,
                    )
                    # model_output 保留 layout/vlm/fused/native_gaps/metrics，
                    # 主要用于调试融合决策，不是最终 middle_json 的必要字段。
                    model_output.extend(
                        {
                            "layout": [dict(block) for block in layout],
                            "vlm": [dict(block) for block in vlm],
                            "fused": fused,
                            "native_gaps": [_native_gap_to_dict(gap) for gap in native_gaps],
                            "metrics": metrics.to_dict(),
                        }
                        for layout, vlm, fused, native_gaps, metrics in zip(
                            layout_results,
                            vlm_results,
                            fused_blocks_list,
                            native_gaps_list,
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
    """VLM fusion 异步入口。

    与 doc_analyze 的业务流程相同；差异是模型调用使用 async API，
    CPU/同步 PDF 操作通过 asyncio.to_thread 放到线程中执行，避免阻塞事件循环。
    """
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

                # 异步路径从 pdf_bytes 渲染窗口页面，避免在事件循环里直接执行重 CPU/IO 工作。
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
                        # 与同步入口一致：先 layout，后内容识别，再做 helper post process。
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
                    async with aio_predictor_execution_guard(predictor):
                        native_gaps_list = await _aio_recognize_window_native_gaps(
                            predictor,
                            images_list,
                            vlm_results,
                            native_spans_list,
                        )
                    fused_blocks_list, metrics_list = await asyncio.to_thread(
                        _fuse_window_pages,
                        pdf_doc,
                        window_start,
                        layout_results,
                        vlm_results,
                        native_spans_list,
                        native_gaps_list,
                    )
                    model_output.extend(
                        {
                            "layout": [dict(block) for block in layout],
                            "vlm": [dict(block) for block in vlm],
                            "fused": fused,
                            "native_gaps": [_native_gap_to_dict(gap) for gap in native_gaps],
                            "metrics": metrics.to_dict(),
                        }
                        for layout, vlm, fused, native_gaps, metrics in zip(
                            layout_results,
                            vlm_results,
                            fused_blocks_list,
                            native_gaps_list,
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
