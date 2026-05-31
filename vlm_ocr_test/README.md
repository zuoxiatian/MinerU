# MinerU VLM/OCR 单模型测试

这个目录是一个单文件 Gradio 测试页，直接调用本地 VLM/OCR 模型，不调用 `mineru-api` 的 `/file_parse`。

启动测试页面：

```powershell
$env:CUDA_PATH="C:\Users\LZ-DSJ-01\miniconda3\envs\mineru\Library"
$env:Path="$env:CUDA_PATH\bin;$env:Path"
$env:MINERU_MODEL_SOURCE="local"
conda run -n mineru python .\vlm_ocr_test\app.py
```

打开：

```text
http://127.0.0.1:7868
```

说明：

- VLM 固定使用 `transformers` 推理引擎。
- 左侧是文件、页码、DPI、VLM token、OCR 语言等设置。
- 右侧是 VLM 对话和 OCR 文本输出。
- PDF 会先渲染成整页图片，再送入 VLM 或 OCR。
- 页面不会调用 `/file_parse`，也不会获取 bbox、中间 JSON、模型输出 JSON 或 content list。



DEFAULT_PROMPTS: dict[str, str] = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
    "[cross_page_table_merge]": "",  # prompt is dynamic, built from table content
}