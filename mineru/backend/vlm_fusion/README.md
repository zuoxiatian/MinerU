# VLM Fusion Backend 执行流程

`vlm_fusion` 是一个“版面检测 + VLM 内容识别 + PDF 原生文本融合”的后端实现。
它的目标不是完全相信 VLM 输出，也不是完全相信 PDF 原生文本，而是在同一个页面上按块融合两类结果：

- VLM 负责识别页面布局、图片化文字、表格/图片/公式等结构化区域。
- PDF 原生文本负责提供可复制文本，优先用于普通文本块，减少 OCR/VLM 误读。
- gap 识别负责补 PDF 原生文本 span 之间疑似漏掉的小片段。
- visual supplement 负责把未被原生文本覆盖、但 VLM 识别到的文本补回来。

## 总入口

同步入口是 `analyze.py::doc_analyze`，异步入口是 `analyze.py::aio_doc_analyze`。
两个入口的处理步骤一致，只是模型调用和 PDF 渲染部分分别使用同步/异步实现。

## 每个处理窗口的主流程

1. 打开 PDF，按 `get_processing_window_size(default=64)` 把页面分成窗口。
2. 将窗口内页面渲染成 PIL 图像。
3. 调用 VLM 做 layout detection，得到每页的 layout blocks。
4. 根据 layout blocks 裁切页面区域，调用 VLM 做内容识别。
5. 对 VLM 输出做 helper post process，得到标准 block 字典列表。
6. 从 PDF 页面抽取原生文本 span，保留 bbox、content、uid。
7. 检测原生文本 span 之间是否存在较大空隙，并裁切 gap 图片交给 VLM 识别。
8. 构造 `PageFusionContext`，调用 `fusion.py::fuse_page` 做页面级融合。
9. 将 fused blocks 转成 middle_json 页面结构，并记录 `_fusion_metrics`。

## 页面融合决策

`fusion.py::fuse_page` 是核心决策函数，主要逻辑如下：

1. 遍历 VLM blocks。
2. 对文本类 block，按 bbox 匹配 PDF 原生 spans。
3. 如果原生文本质量可靠，使用原生文本锁定该 block 的 `content`。
4. 如果该 block 有已识别的 native gap，把 gap 内容插回相邻 span 中间。
5. 如果原生文本不可靠，但 VLM 有内容，则保留 VLM 内容作为 fallback。
6. 对表格、图片、公式、代码等结构类 block，直接保留模型 block。
7. 对 VLM 识别到但未被锁定原生文本覆盖的文本，按配置补充为 `visual_supplement`。
8. 对尚未消费的 PDF 原生 span，如果不落在结构区域中，补充为 `pdf_native_recovered`。
9. 最后按阅读顺序重新排序，并重写连续的 `index`。

## 关键模块

- `analyze.py`：编排完整执行流程，负责 PDF 窗口、模型调用、融合和 middle_json 输出。
- `vlm_runner.py`：封装 VLM layout/content 调用，屏蔽同步/异步预测细节。
- `native_text.py`：抽取、匹配、拼接 PDF 原生文本，并计算文本质量和相似度。
- `gap_recognizer.py`：检测原生文本 span 中间的大空隙，裁图并识别缺失文字。
- `fusion.py`：按页面做核心融合决策。
- `block_builder.py`：构造不同来源的 fused block。
- `visual_candidates.py`：从 VLM 文本 blocks 中收集可作为补充的候选文本。
- `middle_json.py`：把 fused blocks 转成 MinerU 的 middle_json。
- `bbox.py`：坐标转换、面积覆盖率、阅读顺序等几何工具。
- `schemas.py`：定义融合流程中传递的数据结构。
- `config.py`：读取融合开关和阈值配置。

## 文本来源标记

融合后的 block 通常会带 `source` 字段：

- `pdf_native`：该文本块内容来自 PDF 原生文本，通常最可信。
- `vlm`：原生文本不可靠时，保留 VLM 内容。
- `visual_supplement`：VLM 识别到的额外文本补充块。
- `pdf_native_recovered`：没有被 VLM block 消费到的原生文本补回块。
- `vlm_gap`：gap 识别结果，通常插入到 `pdf_native` 文本内部，不一定单独成块。

## 标点和空白处理

主体文本不会统一转换或删除标点。原生文本和 VLM 文本通常按来源保留。
唯一明确的标点清理在 `gap_recognizer.py::normalize_gap_text`：

- 删除 gap 识别结果里的所有空白。
- 去掉首尾中英文标点、引号和括号。
- 过滤 `[Non-Text]`、`无文本` 等非文本响应。

这个处理只用于原生文本缺口补全，不影响普通文本块。
