"""Tests for COCO → YOLO conversion correctness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from convert_coco import coco_to_yolo_bbox, validate_annotation


class TestBboxConversion:
    """Critical: wrong conversion = zero score."""

    def test_simple_box(self):
        """COCO [x, y, w, h] pixels → YOLO [cx, cy, w, h] normalized."""
        # Box at (100, 200) with size (50, 80) on a 1000x800 image
        result = coco_to_yolo_bbox([100, 200, 50, 80], 1000, 800)
        cx, cy, nw, nh = result
        assert abs(cx - 0.125) < 1e-6   # (100 + 25) / 1000
        assert abs(cy - 0.300) < 1e-6   # (200 + 40) / 800
        assert abs(nw - 0.050) < 1e-6   # 50 / 1000
        assert abs(nh - 0.100) < 1e-6   # 80 / 800

    def test_full_image_box(self):
        """Box covering the entire image → center at (0.5, 0.5), size (1.0, 1.0)."""
        result = coco_to_yolo_bbox([0, 0, 2000, 1500], 2000, 1500)
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 0.5) < 1e-6
        assert abs(result[2] - 1.0) < 1e-6
        assert abs(result[3] - 1.0) < 1e-6

    def test_corner_box(self):
        """Box at top-left corner."""
        result = coco_to_yolo_bbox([0, 0, 100, 100], 1000, 1000)
        assert abs(result[0] - 0.05) < 1e-6
        assert abs(result[1] - 0.05) < 1e-6

    def test_bottom_right_box(self):
        """Box at bottom-right corner."""
        result = coco_to_yolo_bbox([900, 900, 100, 100], 1000, 1000)
        assert abs(result[0] - 0.95) < 1e-6
        assert abs(result[1] - 0.95) < 1e-6

    def test_values_clamped(self):
        """Out-of-bounds values should be clamped to [0, 1]."""
        result = coco_to_yolo_bbox([950, 950, 100, 100], 1000, 1000)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestAnnotationValidation:
    def test_valid_annotation(self):
        ann = {"bbox": [100, 100, 50, 50]}
        assert validate_annotation(ann, 1000, 1000) is True

    def test_zero_width(self):
        ann = {"bbox": [100, 100, 0, 50]}
        assert validate_annotation(ann, 1000, 1000) is False

    def test_zero_height(self):
        ann = {"bbox": [100, 100, 50, 0]}
        assert validate_annotation(ann, 1000, 1000) is False

    def test_negative_coords(self):
        ann = {"bbox": [-10, 100, 50, 50]}
        assert validate_annotation(ann, 1000, 1000) is False

    def test_out_of_bounds(self):
        ann = {"bbox": [980, 100, 50, 50]}
        assert validate_annotation(ann, 1000, 1000) is False

    def test_edge_box(self):
        """Box touching the edge should be valid."""
        ann = {"bbox": [950, 950, 50, 50]}
        assert validate_annotation(ann, 1000, 1000) is True
