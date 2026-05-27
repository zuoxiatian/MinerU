# VLM 布局优先融合方案

## 目标

重新实现一套 VLM 解析逻辑：

1. VLM 先做页面布局检测，获取当前页所有 block bbox。
2. 同一页同时支持两种内容来源：
   - PDF 原生文本读取。
   - VLM/OCR 视觉内容读取。
3. 将当前页识别到的所有内容按 bbox 进行匹配、补全、去重和融合。
4. PDF 原生可可靠提取的内容必须保持准确，不能被 VLM/OCR 覆盖改坏。
5. VLM/OCR 看到但 PDF 原生文本层没有覆盖的视觉文字，例如艺术字、浮层字、图片中文字、矢量化装饰字，需要作为补充内容插入。
6. 最终仍输出兼容现有 `middle_json`、Markdown、content list 的结构。

核心原则：

```text
PDF native 负责准确，不被改坏。
VLM/OCR 负责补洞，补充 native 看不到的视觉文字和结构化内容。
```

不能把“native 优先”理解成“只要 native 存在就丢弃其他视觉文字”。正确策略是：

```text
native reliable:
    锁定 native 内容
    但继续保留 visual text candidates

for each visual text candidate:
    if overlaps locked native text enough:
        skip as duplicate
    else:
        insert as supplemental text block
```

## 当前可复用能力

现有代码中已经有不少基础能力可以复用：

- `mineru_vl_utils.MinerUClient.batch_layout_detect`
  - 可用于 VLM 只做布局检测。
- `mineru_vl_utils.MinerUClient.batch_content_extract`
  - 可基于 layout block 对指定区域做 VLM 内容识别。
- `mineru.utils.pdf_text_tool.get_page_chars`
  - 可通过 PDFium 获取 PDF 原生字符及 bbox。
- `mineru.utils.pdf_text_tool.get_lines_from_chars`
  - 可将字符组织为 line/span。
- `mineru.utils.span_pre_proc.txt_spans_extract`
  - 现有 PDF 原生文本回填逻辑。
- `mineru.utils.span_pre_proc.SpanBlockMatcher`
  - 可按 block bbox 匹配 span。
- `mineru.utils.span_block_fix.fix_text_block`
  - 可将文本 span 整理为 line。
- `mineru.backend.hybrid.hybrid_magic_model.MagicModel`
  - 已经包含部分“layout + OCR/PDF-like span 回填”的思路，可作为参考。

## 推荐整体流程

```text
PDF page image
    -> VLM layout_detect
    -> PDF native text extract
    -> VLM/OCR visual text candidates
    -> VLM content_extract for visual/structured blocks
    -> page-level fusion
    -> middle_json page_info
    -> finalize_middle_json
    -> markdown/content_list
```

融合逻辑建议放在 `model_output -> middle_json` 之前。原因是 `middle_json` 后续还会进入段落合并、图片裁剪、表格跨页合并等流程，越晚融合越难保证结构一致。

## 页面级处理步骤

### 1. VLM 布局检测

对每页图片调用：

```python
layout_blocks = predictor.batch_layout_detect(images)
```

得到每页 layout block：

```json
{
  "type": "text",
  "bbox": [0.1, 0.2, 0.8, 0.3],
  "index": 1,
  "angle": 0
}
```

注意：

- VLM 返回 bbox 通常是归一化坐标。
- 进入融合前，需要统一转成 PDF page 坐标。
- 所有后续内容源都必须使用同一坐标系。

### 2. PDF 原生文本读取

通过 PDFium 获取当前页原生字符：

```python
page_chars = get_page_chars(pdf_page)
native_lines = get_lines_from_chars(page_chars["chars"])
```

也可以复用现有 `txt_spans_extract` 流程，将 PDF 字符直接回填到 layout block 中。

### 3. VLM/OCR 视觉文字候选

VLM/OCR 识别出的文字不要直接丢弃，也不要立刻覆盖 native 文本。先作为 `visual_text_candidate` 暂存：

```python
{
    "bbox": [x0, y0, x1, y1],
    "type": "text",
    "content": "...",
    "source": "vlm_or_ocr",
}
```

这些候选用于补充 PDF 原生文本层没有覆盖到的内容，例如：

- 艺术字。
- 页面中间插入的浮层文字。
- 图片中文字。
- 矢量化装饰文字。
- PDF text layer 没有覆盖的小标题、标注、印章文字。

### 4. VLM 内容识别

对需要视觉理解的 block 调用 VLM 内容识别：

```python
vlm_blocks = predictor.batch_content_extract(images, layout_blocks)
```

建议第一版只让 VLM 识别这些类型：

- `table`
- `image`
- `chart`
- `equation`
- `code`
- PDF 原生文本为空或质量较差的 `text/title/ref_text/list`
- PDF 原生文本没有覆盖到的视觉文字候选区域

普通文本优先走 PDF 原生读取，可以显著减少 VLM 内容识别开销，并提高可复制 PDF 的文字准确率。

但这不是说忽略 VLM/OCR 看到的文字。VLM/OCR 识别出的文字需要作为 `visual_text_candidate` 暂存，后续用 bbox 与可靠 native 文本判重。如果没有被 native 覆盖，则作为补充文本插入页面。

### 5. 建立页面 bbox registry

每页维护一个统一的 registry：

```python
page_registry = {
    "layout_blocks": [],
    "native_spans": [],
    "native_lines": [],
    "locked_native_blocks": [],
    "visual_text_candidates": [],
    "vlm_blocks": [],
}
```

所有元素都必须包含：

```python
{
    "bbox": [x0, y0, x1, y1],
    "type": "...",
    "content": "...",
    "source": "pdf_native|vlm|ocr|visual_supplement",
}
```

融合阶段建议临时保留审计信息：

```python
{
    "content": "...",
    "source": "pdf_native",
    "_fusion": {
        "native_text": "...",
        "vlm_text": "...",
        "decision": "native_locked|visual_supplement|vlm_fallback",
        "native_quality": 0.96,
        "overlap_with_locked_native": 0.12,
    },
}
```

最终导出前可以清理 `_fusion`，但调试阶段建议保留。

### 6. 按 layout block 融合

以 VLM layout block 为主骨架，对每个 block 执行：

1. 查找落入该 block bbox 的 PDF native spans。
2. 查找对应的 VLM content block。
3. 对 native text 做质量判断。
4. native 可靠时锁定内容，标记为 `native_locked`，后续 VLM/OCR 不能覆盖。
5. native 不可靠或为空时，允许 VLM/OCR fallback。
6. 生成最终 `preproc_block`。
7. 标记已消费的 native spans。

未被任何 layout block 消费的 PDF native line，可以作为 recovered text block 插回页面。

同时还需要处理 VLM/OCR 视觉文字候选：

1. 收集所有 `visual_text_candidate`。
2. 与 locked native blocks/native spans 做 bbox overlap。
3. overlap 足够高时，认为是重复识别，跳过。
4. overlap 很低时，认为是 PDF 原生文本层漏掉的视觉文字，作为 supplemental text block 插入页面。
5. 插入后按阅读顺序重新排序。

## 融合策略

### 文本类 block

适用类型：

- `text`
- `title`
- `ref_text`
- `phonetic`
- `list`
- `header`
- `footer`
- `page_number`
- `aside_text`
- `page_footnote`

策略：

```text
native_text 有效:
    使用 native_text
    标记 native_locked
    禁止 VLM/OCR 覆盖该 block 内容
native_text 为空/乱码/过短:
    使用 vlm_text
native_text 和 vlm_text 都有效但差异较大:
    默认使用 native_text
    记录 source_conflict 供调试
```

文本有效性可先用简单规则：

- 非空。
- 可见字符比例足够高。
- 控制字符、乱码字符比例低。
- 字符 bbox 与 layout block overlap 足够高。
- 文本长度和 bbox 面积没有明显异常。
- 字符顺序没有严重跳变。

native text 一旦通过质量检查，建议写入：

```python
block["source"] = "pdf_native"
block["_content_locked"] = True
```

后续流程可以调整段落结构，但不应修改已锁定 span 的文本内容。

### 视觉文字补洞

适用场景：

- 艺术字。
- 页面中间插入的浮层文字。
- 图片中文字。
- 矢量化装饰文字。
- PDF text layer 没有覆盖的小标题、标注、印章文字。

策略：

```text
visual_text_candidate 非空:
    计算它与 locked native spans/blocks 的 overlap
overlap >= duplicate_threshold:
    认为 native 已覆盖，跳过
overlap < duplicate_threshold:
    作为 supplemental text block 插入
```

建议第一版阈值：

```text
duplicate_threshold = 0.65
```

如果候选文字 bbox 与 native bbox 只部分重叠，需要进一步判断：

```text
candidate_coverage_by_native = native_overlap_area / candidate_area
```

只有 `candidate_coverage_by_native` 足够高时才判定重复。这样可以避免中间插入的艺术字因为挨着正文而被误删。

补充 block 建议结构：

```python
{
    "type": "text",
    "bbox": candidate_bbox,
    "source": "visual_supplement",
    "lines": [
        {
            "bbox": candidate_bbox,
            "spans": [
                {
                    "bbox": candidate_bbox,
                    "type": "text",
                    "content": candidate_text,
                    "source": "vlm_or_ocr",
                }
            ],
        }
    ],
}
```

如果 VLM layout 已经将该候选标成 `title`，可以保留 `title` 类型。

### 表格 block

策略：

```text
VLM html 有效:
    使用 VLM html
VLM html 为空，但 native lines 存在:
    降级为文本块，或作为 table_text fallback
VLM html 和 native_text 都存在:
    保留 VLM html
    native_text 可后续用于空单元格补全
```

第一版建议不要做复杂表格修复，只保证不丢内容。

### 图片和图表 block

策略：

```text
保留 VLM image/chart block
使用现有 cut_image_and_table 生成图片资源
如果 VLM 有图像描述，则写入 content
PDF native text 只用于 caption/footnote，不直接覆盖 image/chart
```

图片或图表内部的可见文字可以进入 `visual_text_candidate`，再根据配置决定：

```text
默认:
    不作为独立正文输出，避免图片内容重复
需要尽量不漏文字:
    可作为 visual_supplement 插入，或写入 image/chart 的描述字段
```

### 公式 block

策略：

```text
VLM latex 有效:
    使用 VLM latex
否则:
    fallback 到 native_text
```

行内公式可以后续继续复用 hybrid/pipeline 里的公式检测逻辑。

### 代码 block

策略：

```text
VLM code 有效:
    使用 VLM code
native_text 有效且 VLM 为空:
    使用 native_text
```

代码语言识别继续复用：

```python
guess_language_by_text
```

### 未匹配内容

如果 PDF 原生文本没有被任何 layout block 消费：

```text
native line 不在 image/table/equation 区域内:
    生成 recovered text block
native line 落入 discarded 区域:
    放入 discarded_blocks
```

这样可以减少 VLM 漏框导致的文本丢失。

如果 VLM/OCR 视觉文字候选没有被 native 覆盖：

```text
visual candidate 不在 image/table/equation 主体内部:
    生成 supplemental text/title block
visual candidate 位于 image/chart 内部:
    根据配置决定是否输出为 image 内描述或独立文本
visual candidate 与 locked native 高重叠:
    跳过，避免重复
```

对于艺术字和浮层字，建议默认作为独立文本 block 输出，因为它们通常是用户希望看到的页面内容。

## 建议新增模块

建议不要直接大改现有 `vlm_analyze.py` 和 `vlm_magic_model.py`，先新建一套实验链路：

```text
mineru/backend/vlm/fusion_analyze.py
mineru/backend/vlm/fusion_model_output_to_middle_json.py
mineru/backend/vlm/fusion_magic_model.py
mineru/backend/vlm/page_fusion.py
```

职责建议：

- `fusion_analyze.py`
  - 负责窗口分页、加载图片、调用 VLM layout/content、调用融合转换。
- `page_fusion.py`
  - 负责 page-level bbox registry、匹配、去重、融合。
- `fusion_magic_model.py`
  - 负责将融合后的 page blocks 转成 MinerU block 结构。
- `fusion_model_output_to_middle_json.py`
  - 负责 append/finalize middle_json。

## 配置入口

第一版可以先使用环境变量控制，避免 CLI 大改：

```powershell
$env:MINERU_VLM_TEXT_SOURCE="fusion"
```

建议支持：

```text
vlm
    旧 VLM 逻辑，完全使用 VLM two_step_extract。

pdf
    VLM 只做 layout，文本全部使用 PDF 原生读取。

fusion
    PDF 原生读取 + VLM/OCR 视觉补洞 + VLM 内容读取多源融合。
```

后续稳定后，再将其暴露成 CLI/API 参数。

## 第一版 MVP 范围

建议第一版只实现下面内容：

1. VLM layout 检测。
2. PDF 原生文本按 layout bbox 回填文本类 block。
3. 对可靠 PDF native 文本启用 `_content_locked`，禁止 VLM/OCR 覆盖。
4. VLM/OCR 生成视觉文字候选，用 bbox 与 locked native 判重。
5. 未被 native 覆盖的视觉文字候选作为 supplemental text block 插入页面。
6. VLM 识别 `table/image/chart/equation/code`。
7. 未命中的 PDF native text 插回页面。
8. 输出兼容现有 `middle_json`。
9. 继续复用现有 Markdown/content list 生成逻辑。

暂不做：

- VLM/PDF 文本差异智能纠错。
- 表格 html 和 native text 的精细单元格融合。
- 复杂跨页 block 纠正。
- 新的 UI/API 参数。

## 风险点

### 坐标系统一

这是最关键的风险点。

必须明确每个阶段 bbox 的坐标系：

- VLM layout bbox：归一化坐标。
- PDFium text bbox：PDF page 坐标。
- PIL image crop bbox：渲染图像坐标，受 scale 影响。

建议所有融合逻辑统一使用 PDF page 坐标。

### PDF 原生文本质量

部分 PDF 原生文本可能存在：

- 乱码。
- 字符顺序错误。
- ligature 问题。
- 不可见文本层。
- 扫描 PDF 无文本层。

因此 PDF 原生文本只能作为优先源，不应作为唯一源。

同时需要注意：即使 PDF 原生文本质量很好，也不代表页面所有可见文字都在 text layer 中。艺术字、浮层字、图中文字、矢量化文字可能完全没有 native text。因此 native 可靠只能用于锁定已有内容，不能用于否决所有 VLM/OCR 视觉文字候选。

### VLM 漏框和误框

VLM layout 可能漏掉小字、页眉页脚、脚注、公式编号等。

需要通过“未消费 native text 回收”降低漏内容概率。

另一个相反问题是：PDF native 可能漏掉视觉文字。需要通过“visual text candidate 补洞”降低漏艺术字、浮层字、图片中文字的概率。

### 重复内容

PDF native 和 VLM/OCR 同时识别文本时，容易重复输出。

必须通过 span consumed 标记、locked native 标记和 overlap 阈值去重。

去重时不能只看 block 级 bbox 粗重叠。对于插入在正文中间的艺术字，候选 bbox 可能与正文 layout block 有交集，但它并没有被 native spans 覆盖。推荐以候选 bbox 被 locked native spans 覆盖的比例为主，而不是只看与 layout block 的重叠。

### 表格融合复杂度

表格是最复杂部分。

第一版建议只保留 VLM table html，native text 仅作为 fallback 或 debug 信息。

## 推荐实施顺序

1. 新增 `page_fusion.py`，实现 bbox 匹配、span consumed、native locked、visual text candidate 补洞。
2. 新增 `fusion_magic_model.py`，将融合后的 page blocks 转成 `preproc_blocks`。
3. 新增 `fusion_model_output_to_middle_json.py`，复用现有 finalize 逻辑。
4. 新增 `fusion_analyze.py`，串起 layout detect、PDF text extract、VLM/OCR visual candidates、VLM content extract。
5. 在 `mineru/cli/common.py` 中用环境变量切换新旧 VLM 逻辑。
6. 用小 PDF 验证：
   - 可复制 PDF。
   - 扫描 PDF。
   - 混合图文 PDF。
   - 表格 PDF。
   - 页眉页脚/脚注多的 PDF。
   - 页面中间插入艺术字/浮层字的 PDF。

## 验证指标

建议每页记录 debug 信息：

```json
{
  "page_idx": 0,
  "layout_block_count": 12,
  "native_span_count": 90,
  "vlm_content_block_count": 5,
  "visual_text_candidate_count": 4,
  "native_consumed_count": 84,
  "native_recovered_count": 3,
  "native_locked_count": 10,
  "visual_supplement_count": 2,
  "visual_duplicate_skipped_count": 2,
  "source_conflict_count": 1
}
```

输出对比：

- Markdown 是否少内容。
- 是否重复段落。
- 中间插入的艺术字、浮层字是否没有漏。
- 表格是否仍能渲染。
- 图片、图表、公式是否保留。
- 可复制 PDF 的正文是否没有被 VLM/OCR 改错。
- `*_layout.pdf` 是否与融合后的 block 对齐。

