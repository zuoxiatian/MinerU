# vlm_ocr_native 说明

`vlm_ocr_native` 是一个实验后端，用来处理普通 PDF 与漫画/图像主导页面混合的场景。

目标不是替代原 VLM 流程，而是在 VLM layout 判断出页面更像漫画时，改用 OCR 识别整页文字区域，再继续执行 native text correction 和现有 middle_json/Markdown 输出流程。

## 总体流程

1. PDF 页面渲染成图片。
2. 运行 VLM layout 检测：
   ```python
   predictor.batch_layout_detect(images)
   ```
3. 基于 VLM layout 判断页面类型：
   - 普通页：走 VLM layout block 裁图 + VLM 内容识别。
   - 漫画/图像主导页：走 OCR 检测和识别。
4. 两条分支都会进入复制到本目录内的 native correction 逻辑。
5. 输出保持现有 middle_json、Markdown、content_list、model_output 兼容。

本目录要求自包含，不依赖 `mineru.backend.vlm_native_correction` 或 `mineru.backend.vlm_fusion`。旧实验模式可以删除，本目录仍应能独立工作。

## 普通页处理

普通页沿用 VLM 主流程：

```text
VLM layout
  -> batch_content_extract_from_layouts
  -> native text extraction
  -> native gap recognition
  -> correct_page
  -> middle_json
```

普通页的文字来源通常是：

```json
"text_source": "vlm"
```

如果 native correction 修改了文字，调试输出中会出现：

```json
"final_source": "vlm_native_corrected"
```

## 漫画页处理

漫画页不再调用 VLM 内容识别，而是：

```text
VLM layout
  -> 判断 comic-like
  -> OCR det+rec
  -> 合并上下相邻 OCR bbox
  -> 根据 VLM layout hint 排序
  -> native correction
  -> middle_json
```

OCR 文字 block 会带：

```json
"source": "ocr"
```

对应的 `bbox_text_compare.json` 里会看到：

```json
"text_source": "ocr"
```

## 漫画页判定规则

判定只使用 VLM layout，不先跑 OCR，避免 OCR 耗时。

当前规则在 `page_classifier.py`：

```text
is_comic_like =
    largest_image_block_area_ratio >= 0.75
    or image_block_count > text_block_count
    or total_image_block_area_ratio >= 0.65
    or max_column_image_block_area_ratio >= 0.60
```

### 1. 最大图片块覆盖率

用于处理 VLM 把整页漫画识别成一张大图的情况：

```text
largest_image_block_area_ratio >= 0.75
```

触发原因：

```json
"reason": "largest_image_block"
```

### 2. 图片块数量多于文字块数量

用于处理页面明显由多个图片块组成的情况：

```text
image_block_count > text_block_count
```

触发原因：

```json
"reason": "image_block_count"
```

### 3. 全页图片块总面积

用于处理一页被拆成多个漫画分镜的情况。单个 image block 不大，但总图片面积很大：

```text
total_image_block_area_ratio >= 0.65
```

触发原因：

```json
"reason": "total_image_block_area"
```

典型情况：

```json
{
  "largest_image_block_area_ratio": 0.343168,
  "total_image_block_area_ratio": 0.729278,
  "reason": "total_image_block_area"
}
```

### 4. 双栏/双页的列级图片占比

有些扫描页是双页或双栏混合：左侧像漫画，右侧像普通图文。整页图片总面积会被右侧文字稀释，因此新增列级判定。

当 VLM layout 判断出：

```json
"page_layout": {
  "flow": "double_column"
}
```

会按每列单独计算图片面积：

```text
任一列 image_area / column_area >= 0.60
```

触发原因：

```json
"reason": "column_image_block_area"
```

调试字段：

```json
"max_column_image_block_area_ratio": 0.632884,
"column_image_block_area_ratios": [
  {
    "column_index": 1,
    "image_block_count": 5,
    "image_block_area_ratio": 0.632884
  },
  {
    "column_index": 2,
    "image_block_count": 3,
    "image_block_area_ratio": 0.306967
  }
]
```

当前实现是页面级分流：任一列命中后，整页走 OCR。后续如果需要更精细，可以做“漫画列 OCR，普通列 VLM，再合并”。

## 双栏判断与排序

双栏判断只使用 VLM layout block 的几何分布，不使用 OCR。

位置：`page_classifier.py::analyze_layout_flow`

大致逻辑：

1. 收集 VLM layout 中可用于判断列结构的 block。
2. 按页面中心线分成左右两组。
3. 左右两组数量都足够，并且中心距离、纵向重叠满足条件时，标记为：
   ```json
   "flow": "double_column"
   ```
4. 否则：
   ```json
   "flow": "single_column"
   ```

漫画页 OCR 分支会接收这个 `page_layout` 作为 `layout_hint`。如果是双栏，OCR 合并后的 block 会按列排序：

```text
左列从上到下 -> 右列从上到下
```

阅读方向可通过环境变量配置：

```powershell
$env:MINERU_VLM_OCR_NATIVE_READING_ORDER="left_to_right"
```

可选值当前约定：

```text
left_to_right
right_to_left
```

## OCR bbox 合并

位置：`ocr_text_regions.py`

OCR 检测到的原始文字框通常是一行或一小段。漫画对白可能上下紧挨，因此会合并上下接近的框。

核心条件：

```text
vertical_gap <= max_vertical_gap_px
and (
    x_overlap_ratio >= min_x_overlap_ratio
    or center_x close enough
)
```

默认参数：

```text
MINERU_VLM_OCR_NATIVE_MERGE_GAP_PX = 12
MINERU_VLM_OCR_NATIVE_MIN_X_OVERLAP = 0.25
```

可通过环境变量调整。

## 页眉、页脚和页码

OCR 本身不知道页面结构，默认只会识别文字。因此漫画 OCR 分支会借用 VLM layout 的结构区域。

流程：

1. 从 VLM layout 收集：
   ```text
   header
   footer
   page_number
   ```
2. OCR 合并后的 bbox 转为 PDF 坐标。
3. 如果 OCR bbox 中心点落入这些 VLM 区域，就把 OCR block 的 `type` 改成对应类型。

这样这些内容不会作为普通正文 `text` 输出，而是进入现有 `MagicModel` 的 discarded/page number 处理逻辑。

调试字段：

```json
"ocr": {
  "type_regions": [
    {
      "type": "footer",
      "bbox": [...],
      "source": "vlm_layout"
    }
  ]
}
```

被命中的 OCR block 会带：

```json
"_ocr_type_region": {
  "type": "footer",
  "bbox": [...],
  "source": "vlm_layout"
}
```

## 调试输出字段

`*_model.json` 每页会包含：

```json
"model_output_version": "vlm_ocr_native_layout_v1"
```

如果没有这个字段，通常说明：

1. API/Gradio 服务没有重启。
2. 端口上还有旧的 Python 子进程。
3. 当前输出是旧结果。

可用脚本重启：

```powershell
.\run\stop_mineru.bat
.\run\start_mineru.bat
```

当前启动脚本会清理 `7860/8000` 端口上的旧监听进程，避免旧 API 残留。

### page_classification

示例：

```json
"page_classification": {
  "is_comic_like": true,
  "largest_image_block_area_ratio": 0.068442,
  "total_image_block_area_ratio": 0.363082,
  "max_column_image_block_area_ratio": 0.632884,
  "column_image_block_area_ratios": [...],
  "image_block_count": 8,
  "text_block_count": 16,
  "reason": "column_image_block_area"
}
```

### page_layout

示例：

```json
"page_layout": {
  "flow": "double_column",
  "source": "vlm_layout",
  "reading_order": "left_to_right",
  "columns": [...],
  "reason": "vlm_blocks_split_by_page_center"
}
```

### bbox_text_compare

`*_bbox_text_compare.json` 会保留文字来源：

```json
"text_source": "ocr",
"final_source": "ocr"
```

普通 VLM：

```json
"text_source": "vlm",
"final_source": "vlm"
```

被 native correction 修改：

```json
"final_source": "vlm_native_corrected"
```

## 已知取舍

### 整页分流而不是局部分流

当前只做页面级分流：

```text
普通页 -> 整页 VLM
漫画页 -> 整页 OCR
```

对于双页/双栏混合页面，如果只有一列像漫画，当前也会整页 OCR。这是为了先保证漫画文字可识别，降低实现复杂度。

后续可优化为：

```text
漫画列 OCR
普通列 VLM
按 column/order 合并
```

### 只用 VLM 判断漫画页

漫画页判定当前只用 VLM layout，不用 OCR bbox。这样能避免先跑 OCR 带来的额外耗时。

如果后续 VLM layout 仍有漏判，可以加 OCR 二次判断：

```text
OCR merged bbox 分布像气泡/漫画文字
或 OCR 文字区域主要集中在图像块内
```

### VLM layout 可能误判页眉页脚

页眉页脚归类依赖 VLM layout。如果 VLM 把装饰线、非文字误判成 header/footer，OCR block 中心落入时可能被错误归类。

目前处理策略是保守地只使用 VLM 的 `header/footer/page_number` 区域，不额外用关键词强删正文。

## 相关文件

- `analyze.py`：主流程、普通/VLM 与漫画/OCR 分流、debug 输出。
- `page_classifier.py`：漫画页判定、双栏判断、列级图片占比。
- `ocr_text_regions.py`：OCR 检测识别、bbox 合并、按 VLM layout hint 排序、页眉页脚归类。
- `correction.py`：native text correction 与 `bbox_text_compare` 记录。
- `middle_json.py`：把修正后的 block 转成 middle_json。
- `vlm_runner.py`：VLM layout 和 VLM content extraction 封装。
