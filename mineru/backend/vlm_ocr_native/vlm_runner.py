# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio


def _flatten_prepared_inputs(prepared_inputs):
    """把每页 prepare_for_extract 的结果展平，方便一次性批量调用模型。

    返回的 all_indices 记录每个裁图结果属于哪一页、哪一个 block，
    模型输出回来后再按这个索引写回 layout_results。
    """
    all_images = []
    all_prompts = []
    all_params = []
    all_indices = []
    for img_idx, prepared in enumerate(prepared_inputs):
        block_images, prompts, params, indices = prepared
        all_images.extend(block_images)
        all_prompts.extend(prompts)
        all_params.extend(params)
        all_indices.extend((img_idx, idx) for idx in indices)
    return all_images, all_prompts, all_params, all_indices


def batch_layout_detect(predictor, images):
    """同步批量版面检测入口，直接委托给 predictor。"""
    return predictor.batch_layout_detect(images)


def batch_content_extract_from_layouts(
    predictor,
    images,
    layout_results,
    image_analysis: bool = True,
):
    """同步批量内容识别。

    helper 先根据 layout_results 准备 block 裁图、prompt 和推理参数；
    predictor 批量生成文本后写回原 layout block；最后由 helper 做统一后处理，
    输出融合层使用的标准 block 列表。
    """
    prepared_inputs = predictor.helper.batch_prepare_for_extract(
        predictor.executor,
        images,
        layout_results,
        None,
        image_analysis,
    )
    all_images, all_prompts, all_params, all_indices = _flatten_prepared_inputs(prepared_inputs)
    if all_images:
        outputs = predictor._batch_predict(all_images, all_prompts, all_params, None, None)
        for (img_idx, idx), output in zip(all_indices, outputs):
            layout_results[img_idx][idx].content = output.text
            layout_results[img_idx][idx].scored = output.scored
    processed_list = predictor.helper.batch_post_process(predictor.executor, layout_results)
    return processed_list


async def aio_batch_layout_detect(predictor, images):
    """异步批量版面检测入口。"""
    return await predictor.aio_batch_layout_detect(images)


async def aio_batch_content_extract_from_layouts(
    predictor,
    images,
    layout_results,
    image_analysis: bool = True,
):
    """异步批量内容识别，流程与同步函数保持一致。"""
    prepared_inputs = await asyncio.gather(
        *[
            predictor.helper.aio_prepare_for_extract(
                predictor.executor,
                image,
                layout_result,
                None,
                image_analysis,
            )
            for image, layout_result in zip(images, layout_results)
        ]
    )
    all_images, all_prompts, all_params, all_indices = _flatten_prepared_inputs(prepared_inputs)
    if all_images:
        outputs = await predictor._aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            None,
            asyncio.Semaphore(predictor.max_concurrency),
            None,
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            layout_results[img_idx][idx].content = output.text
            layout_results[img_idx][idx].scored = output.scored
    return await asyncio.gather(
        *[
            predictor.helper.aio_post_process(predictor.executor, layout_result)
            for layout_result in layout_results
        ]
    )
