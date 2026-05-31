# Copyright (c) Opendatalab. All rights reserved.
from pathlib import Path


OFFICE_PARSE_DIR_NAME = "office"
VLM_PARSE_DIR_NAME = "vlm"
VLM_FUSION_PARSE_DIR_NAME = "vlm_fusion"
VLM_NATIVE_CORRECTION_PARSE_DIR_NAME = "vlm_native_correction"
VLM_OCR_NATIVE_PARSE_DIR_NAME = "vlm_ocr_native"


def build_parse_dir(
    output_dir: str | Path,
    pdf_name: str,
    backend: str,
    parse_method: str,
    *,
    is_office: bool = False,
) -> Path:
    output_root = Path(output_dir)
    if is_office:
        return output_root / pdf_name / OFFICE_PARSE_DIR_NAME
    if backend.startswith("pipeline"):
        return output_root / pdf_name / parse_method
    if backend.startswith("vlm-ocr-native") or parse_method == VLM_OCR_NATIVE_PARSE_DIR_NAME:
        return output_root / pdf_name / VLM_OCR_NATIVE_PARSE_DIR_NAME
    if backend.startswith("vlm-native-correction") or parse_method == VLM_NATIVE_CORRECTION_PARSE_DIR_NAME:
        return output_root / pdf_name / VLM_NATIVE_CORRECTION_PARSE_DIR_NAME
    if backend.startswith("vlm-fusion") or parse_method == VLM_FUSION_PARSE_DIR_NAME:
        return output_root / pdf_name / VLM_FUSION_PARSE_DIR_NAME
    if backend.startswith("vlm"):
        return output_root / pdf_name / VLM_PARSE_DIR_NAME
    if backend.startswith("hybrid"):
        return output_root / pdf_name / f"hybrid_{parse_method}"
    raise ValueError(f"Unknown backend type: {backend}")


def resolve_parse_dir(
    output_dir: str | Path,
    pdf_name: str,
    backend: str,
    parse_method: str,
    *,
    is_office: bool = False,
    allow_office_fallback: bool = False,
) -> Path:
    parse_dir = build_parse_dir(
        output_dir,
        pdf_name,
        backend,
        parse_method,
        is_office=is_office,
    )
    if is_office:
        return parse_dir

    if allow_office_fallback and not parse_dir.exists():
        office_dir = build_parse_dir(
            output_dir,
            pdf_name,
            backend,
            parse_method,
            is_office=True,
        )
        if office_dir.exists():
            return office_dir
    return parse_dir
