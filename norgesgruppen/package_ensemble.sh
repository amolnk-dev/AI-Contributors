#!/bin/bash
# Package 3-model ensemble for submission
# Usage: bash package_ensemble.sh model_a.pt model_b.pt model_c.pt

set -e
cd "$(dirname "$0")"

MODEL_A="${1:-models/best_finetune_cos.pt}"
MODEL_B="${2:-models/best_ftcos_sgd.pt}"
MODEL_C="${3:-models/best_sgd_seed42.pt}"
OUTPUT="submission.zip"

for M in "$MODEL_A" "$MODEL_B" "$MODEL_C"; do
    if [ ! -f "$M" ]; then
        echo "ERROR: Model not found: $M"
        exit 1
    fi
done

# Check total size
TOTAL_MB=0
for M in "$MODEL_A" "$MODEL_B" "$MODEL_C"; do
    SIZE=$(stat -f%z "$M" 2>/dev/null || stat -c%s "$M" 2>/dev/null)
    TOTAL_MB=$((TOTAL_MB + SIZE / 1024 / 1024))
done
echo "Total model size: ${TOTAL_MB} MB"
if [ "$TOTAL_MB" -gt 400 ]; then
    echo "ERROR: Models exceed 420 MB limit (${TOTAL_MB} MB)"
    exit 1
fi

# Remove stale zip to prevent appending
rm -f "$OUTPUT"

# Create staging
STAGING=$(mktemp -d)
cp run_ensemble.py "$STAGING/run.py"
cp "$MODEL_A" "$STAGING/model_a.pt"
cp "$MODEL_B" "$STAGING/model_b.pt"
cp "$MODEL_C" "$STAGING/model_c.pt"

# Create zip
cd "$STAGING"
zip -r "$OLDPWD/$OUTPUT" . -x ".*" "__MACOSX/*"
cd "$OLDPWD"
rm -rf "$STAGING"

SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
SIZE_MB=$((SIZE / 1024 / 1024))

echo ""
echo "=== Ensemble Submission ==="
echo "Output: $OUTPUT (${SIZE_MB} MB)"
echo "Models: $(basename $MODEL_A), $(basename $MODEL_B), $(basename $MODEL_C)"
echo ""
echo "Contents:"
unzip -l "$OUTPUT"

if [ "$SIZE_MB" -gt 420 ]; then
    echo "ERROR: Package exceeds 420 MB limit!"
    exit 1
fi

echo ""
echo "Ready to upload"
