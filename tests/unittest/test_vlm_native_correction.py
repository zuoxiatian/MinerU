from mineru.backend.vlm_native_correction.config import NativeCorrectionConfig
from mineru.backend.vlm_native_correction.correction import correct_vlm_text_with_native


def test_vlm_primary_keeps_vlm_spacing_when_native_differs():
    result = correct_vlm_text_with_native(
        "第 1 章",
        "第1章",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "第 1 章"
    assert result["decision"] == "vlm"


def test_vlm_primary_corrects_character_from_native():
    result = correct_vlm_text_with_native(
        "今天天汽很好",
        "今天天气很好",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "今天天气很好"
    assert result["corrected_char_count"] == 1


def test_vlm_primary_keeps_vlm_line_breaks():
    result = correct_vlm_text_with_native(
        "听上\n去不怎么友好",
        "听上去不怎么友好",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "听上\n去不怎么友好"


def test_vlm_primary_can_insert_short_native_missing_text():
    result = correct_vlm_text_with_native(
        "这头用咩咩声",
        "这头公羊用咩咩声",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "这头公羊用咩咩声"
    assert result["inserted_char_count"] == 2


def test_vlm_primary_can_insert_large_native_missing_text_when_anchored():
    result = correct_vlm_text_with_native(
        "开头结尾",
        "开头中间漏掉了很长一段文字结尾",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "开头中间漏掉了很长一段文字结尾"
    assert result["inserted_char_count"] == 11


def test_vlm_primary_rejects_large_native_missing_text_when_conflicted():
    result = correct_vlm_text_with_native(
        "开头X结尾",
        "开头中间漏掉了很长一段文字Y结尾",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "开头X结尾"
    assert result["decision"] == "keep_vlm_conflict"


def test_vlm_primary_does_not_replace_vlm_with_suspicious_native_script():
    result = correct_vlm_text_with_native(
        "\u7406\u89e3\u63d0\u51fa\u4e86\u8003\u9a8c",
        "\u7406\u89e3\u1a00\u51fa\u4e86\u8003\u9a8c",
        NativeCorrectionConfig(),
    )

    assert result["text"] == "\u7406\u89e3\u63d0\u51fa\u4e86\u8003\u9a8c"
    assert result["corrected_char_count"] == 0


def test_vlm_primary_keeps_vlm_when_native_has_same_content_wrong_order():
    vlm_text = "\u6211\u4eec\u5e2e\u5b83\u53d6\u540d\u5b57\n\u597d\u4e0d\u597d\uff1f\n\u597d\u554a\u3002"
    native_text = "\u6211\u4eec\u5e2e\u5b83\u53d6\u540d\u5b57\u597d\u554a\u3002\n\u597d\u4e0d\u597d\uff1f"

    result = correct_vlm_text_with_native(
        vlm_text,
        native_text,
        NativeCorrectionConfig(),
    )

    assert result["text"] == vlm_text
    assert result["decision"] == "keep_vlm_order_conflict"
    assert result["inserted_char_count"] == 0
