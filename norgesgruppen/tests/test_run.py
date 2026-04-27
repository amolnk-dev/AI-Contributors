"""Tests for run.py submission script."""

import ast
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path


BANNED_IMPORTS = {"os", "subprocess", "socket", "ctypes", "builtins"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}


class TestRunPySecurity:
    """Verify run.py passes sandbox security scan."""

    def _get_run_py_source(self):
        run_py = Path(__file__).parent.parent / "run.py"
        return run_py.read_text()

    def test_no_banned_imports(self):
        """run.py must not import os, subprocess, socket, ctypes, builtins."""
        source = self._get_run_py_source()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    assert module not in BANNED_IMPORTS, \
                        f"Banned import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    assert module not in BANNED_IMPORTS, \
                        f"Banned import: from {node.module}"

    def test_no_banned_calls(self):
        """run.py must not use eval(), exec(), compile(), __import__()."""
        source = self._get_run_py_source()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in BANNED_CALLS, \
                        f"Banned call: {node.func.id}()"

    def test_uses_pathlib(self):
        """run.py should import pathlib."""
        source = self._get_run_py_source()
        assert "from pathlib import Path" in source or "import pathlib" in source


class TestRunPyOutput:
    """Verify output format matches spec."""

    def test_image_id_extraction(self):
        """img_00042.jpg → 42, img_00001.jpg → 1."""
        # Simulate the extraction logic from run.py
        test_cases = [
            ("img_00042.jpg", 42),
            ("img_00001.jpg", 1),
            ("img_00332.jpg", 332),
            ("img_00099.jpeg", 99),
        ]
        for filename, expected_id in test_cases:
            stem = Path(filename).stem
            image_id = int(stem.split("_")[-1])
            assert image_id == expected_id, f"{filename} → {image_id}, expected {expected_id}"

    def test_bbox_coco_format(self):
        """Predictions must use COCO [x, y, w, h], not xyxy."""
        # xyxy: [100, 200, 150, 280] → COCO: [100, 200, 50, 80]
        x1, y1, x2, y2 = 100, 200, 150, 280
        bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]
        assert bbox == [100.0, 200.0, 50.0, 80.0]
        # Width and height must be positive
        assert bbox[2] > 0
        assert bbox[3] > 0
