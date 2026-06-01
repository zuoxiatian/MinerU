from mineru.backend.vlm_ocr_native.char_alignment import (
    CONFLICT,
    align_native_with_vlm,
)


def test_alignment_ignores_spacing_between_content_characters():
    native_text = "\u4e8c OO \u4e03\u5e74\u516d\u6708\u4e8c\u5341\u516d\u65e5"
    vlm_text = "\u4e8c00 \u4e03\u5e74\u516d\u6708\u4e8c\u5341\u516d\u65e5"

    result = align_native_with_vlm(native_text, vlm_text)

    assert result.native_missing_count == 0
    assert result.native_extra_count == 0
    assert [
        (op.op_type, op.native, op.vlm)
        for op in result.operations
        if op.native == "O" or op.vlm == "0"
    ] == [
        (CONFLICT, "O", "0"),
        (CONFLICT, "O", "0"),
    ]
