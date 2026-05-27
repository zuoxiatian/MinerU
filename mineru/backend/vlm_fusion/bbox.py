# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from mineru.backend.vlm_fusion.schemas import BBox


def clamp_bbox(bbox: BBox, width: float, height: float) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    return [x0, y0, x1, y1]


def unit_to_pdf_bbox(bbox: BBox, width: float, height: float) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return clamp_bbox([x0 * width, y0 * height, x1 * width, y1 * height], width, height)


def pdf_to_unit_bbox(bbox: BBox, width: float, height: float) -> BBox:
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = clamp_bbox(bbox, width, height)
    return [
        round(x0 / width, 6),
        round(y0 / height, 6),
        round(x1 / width, 6),
        round(y1 / height, 6),
    ]


def bbox_area(bbox: BBox) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def intersection_area(bbox1: BBox, bbox2: BBox) -> float:
    x0 = max(float(bbox1[0]), float(bbox2[0]))
    y0 = max(float(bbox1[1]), float(bbox2[1]))
    x1 = min(float(bbox1[2]), float(bbox2[2]))
    y1 = min(float(bbox1[3]), float(bbox2[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def coverage_by_boxes(target_bbox: BBox, boxes: list[BBox]) -> float:
    target_area = bbox_area(target_bbox)
    if target_area <= 0 or not boxes:
        return 0.0

    # Sweep-line union area of intersections. The box count per page is small,
    # so a simple exact rectangle-union implementation is sufficient.
    intersections: list[BBox] = []
    for box in boxes:
        x0 = max(target_bbox[0], box[0])
        y0 = max(target_bbox[1], box[1])
        x1 = min(target_bbox[2], box[2])
        y1 = min(target_bbox[3], box[3])
        if x1 > x0 and y1 > y0:
            intersections.append([x0, y0, x1, y1])
    if not intersections:
        return 0.0

    xs = sorted({x for box in intersections for x in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        y_segments = [
            (box[1], box[3])
            for box in intersections
            if box[0] < right and box[2] > left
        ]
        if not y_segments:
            continue
        y_segments.sort()
        merged_height = 0.0
        cur_y0, cur_y1 = y_segments[0]
        for y0, y1 in y_segments[1:]:
            if y0 <= cur_y1:
                cur_y1 = max(cur_y1, y1)
            else:
                merged_height += cur_y1 - cur_y0
                cur_y0, cur_y1 = y0, y1
        merged_height += cur_y1 - cur_y0
        area += (right - left) * merged_height
    return min(1.0, area / target_area)


def center_in_bbox(inner: BBox, outer: BBox) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def reading_order_key(block: dict):
    bbox = block.get("bbox") or [0, 0, 0, 0]
    return (round(float(bbox[1]) / 12) * 12, float(bbox[0]))

