from mineru.backend.vlm_fusion.char_alignment import (
    effective_token_count,
    merge_native_with_vlm,
)
from mineru.backend.vlm_fusion.config import FusionConfig
from mineru.backend.vlm_fusion.fusion import fuse_page
from mineru.backend.vlm_fusion.gap_recognizer import normalize_gap_text
from mineru.backend.vlm_fusion.native_text import detect_native_gaps_for_block
from mineru.backend.vlm_fusion.schemas import NativeSpan, PageFusionContext


def test_merge_keeps_native_on_vlm_wrong_character():
    result = merge_native_with_vlm("今天天汽很好", "今天天气很好")

    assert result.text == "今天天汽很好"
    assert result.filled_missing_count == 0


def test_merge_fills_short_native_missing_text():
    result = merge_native_with_vlm("今天气很好", "今天天气很好")

    assert result.text == "今天天气很好"
    assert result.filled_missing_count == 1


def test_merge_does_not_fill_missing_digit():
    result = merge_native_with_vlm("第页", "第1页")

    assert result.text == "第页"
    assert result.filled_missing_count == 0


def test_effective_token_count_ignores_space_and_punctuation():
    assert effective_token_count("A, B。") == 2


def test_fuse_page_orders_native_spans_by_vlm_text_per_bbox():
    context = PageFusionContext(
        page_index=0,
        page_width=100,
        page_height=100,
        layout_blocks=[],
        vlm_blocks=[
            {
                "type": "text",
                "bbox": [0, 0, 1, 1],
                "angle": 0,
                "content": "ABC",
                "index": 1,
            }
        ],
        native_spans=[
            NativeSpan(bbox=[0, 0, 10, 10], content="A", uid=1),
            NativeSpan(bbox=[10, 0, 20, 10], content="C", uid=2),
            NativeSpan(bbox=[20, 0, 30, 10], content="B", uid=3),
        ],
        visual_candidates=[],
    )

    blocks, metrics, compare_records = fuse_page(context, FusionConfig(debug=True))

    assert blocks[0]["content"] == "ABC"
    assert blocks[0]["source"] == "pdf_native"
    assert metrics.native_vlm_order_applied_count == 1
    assert compare_records == [
        {
            "page_index": 0,
            "bbox_index": 1,
            "native_text": "ACB",
            "vlm_text": "ABC",
            "final_text": "ABC",
        }
    ]


def test_detect_native_gap_between_short_left_and_long_right_span():
    spans = [
        NativeSpan(
            bbox=[133.2279, 165.8719, 161.6059, 179.8719],
            content="这头",
            uid=8,
        ),
        NativeSpan(
            bbox=[193.5529, 165.8799, 394.0609, 179.8799],
            content="用咩咩声回答我的问题，但听上",
            uid=9,
        ),
    ]

    gaps = detect_native_gaps_for_block(
        5,
        [100.912, 161.508, 394.638, 227.238],
        spans,
        0.45,
        width_ratio=2.5,
        min_width=18.0,
    )

    assert len(gaps) == 1
    assert gaps[0].left_span_uid == 8
    assert gaps[0].right_span_uid == 9


def test_fuse_page_softens_native_line_break_when_vlm_is_continuous():
    context = PageFusionContext(
        page_index=0,
        page_width=100,
        page_height=100,
        layout_blocks=[],
        vlm_blocks=[
            {
                "type": "text",
                "bbox": [0, 0, 1, 1],
                "angle": 0,
                "content": "听上去不怎么友好",
                "index": 1,
            }
        ],
        native_spans=[
            NativeSpan(bbox=[0, 0, 40, 10], content="听上", uid=1),
            NativeSpan(bbox=[0, 20, 80, 30], content="去不怎么友好", uid=2),
        ],
        visual_candidates=[],
    )

    blocks, _, compare_records = fuse_page(context, FusionConfig(debug=True))

    assert blocks[0]["content"] == "听上去不怎么友好"
    assert compare_records[0]["native_text"] == "听上\n去不怎么友好"
    assert compare_records[0]["final_text"] == "听上去不怎么友好"


def test_normalize_gap_text_drops_symbol_noise_but_keeps_cjk_text():
    assert normalize_gap_text("→", 20) == ""
    assert normalize_gap_text("—", 20) == ""
    assert normalize_gap_text("1", 20) == ""
    assert normalize_gap_text("L", 20) == ""
    assert normalize_gap_text("公羊", 20) == "公羊"
