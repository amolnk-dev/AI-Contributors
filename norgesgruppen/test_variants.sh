#!/bin/bash
# Test all WBF variants and compare scores.
# Usage: bash test_variants.sh [--model models/best_full_300ep_a2.pt]
set -e
cd "$(dirname "$0")"

MODEL="${1:-models/best_full_300ep_a2.pt}"

echo "=========================================="
echo "Testing WBF variants with model: $MODEL"
echo "=========================================="

echo ""
echo "--- Variant 1: skip_box_thr=0.01 only (iou=0.55, conf_type=avg) ---"
python test_local.py --model "$MODEL" --run-py run_wbf_skip01.py 2>&1 | tail -8

echo ""
echo "--- Variant 2: skip=0.01 + iou=0.65 (conf_type=avg) ---"
python test_local.py --model "$MODEL" --run-py run_wbf_iou65.py 2>&1 | tail -8

echo ""
echo "--- Variant 3: skip=0.01 + iou=0.65 + conf_type=max (MAIN run.py) ---"
python test_local.py --model "$MODEL" --run-py run.py 2>&1 | tail -8

echo ""
echo "--- Variant 4: skip=0.01 + iou=0.65 + conf_type=absent_model_aware_avg ---"
python test_local.py --model "$MODEL" --run-py run_wbf_absent.py 2>&1 | tail -8

echo ""
echo "--- Variant 5: 4-pass inference ---"
python test_local.py --model "$MODEL" --run-py run_4pass.py 2>&1 | tail -8

echo ""
echo "--- Variant 6: NMW fusion ---"
python test_local.py --model "$MODEL" --run-py run_nmw.py 2>&1 | tail -8

echo ""
echo "=========================================="
echo "Done! Compare val_metric scores above."
echo "=========================================="
