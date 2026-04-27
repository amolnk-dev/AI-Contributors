#!/bin/bash
# Package run.py + model weights for NorgesGruppen submission
#
# Usage: bash package.sh [model_path]
#   Default model: models/best.pt

set -e
cd "$(dirname "$0")"

MODEL="${1:-models/best.pt}"
OUTPUT="submission.zip"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Train first: python train.py"
    exit 1
fi

if [ ! -f "run.py" ]; then
    echo "ERROR: run.py not found"
    exit 1
fi

# Check for banned imports in run.py
BANNED="^import os$|^from os |^import subprocess|^import socket|^import ctypes|^import builtins"
if grep -qE "$BANNED" run.py; then
    echo "ERROR: run.py contains banned imports!"
    grep -nE "$BANNED" run.py
    exit 1
fi

# Create staging directory
STAGING=$(mktemp -d)
cp run.py "$STAGING/"
cp "$MODEL" "$STAGING/best.pt"

# Create zip
cd "$STAGING"
zip -r "$OLDPWD/$OUTPUT" . -x ".*" "__MACOSX/*"
cd "$OLDPWD"
rm -rf "$STAGING"

# Validate
SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
SIZE_MB=$((SIZE / 1024 / 1024))

echo ""
echo "=== Submission Package ==="
echo "Output: $OUTPUT"
echo "Size: ${SIZE_MB} MB"

if [ "$SIZE_MB" -gt 420 ]; then
    echo "ERROR: Package exceeds 420 MB limit!"
    exit 1
fi

echo ""
echo "Contents:"
unzip -l "$OUTPUT"

echo ""
echo "Verify run.py is at root:"
unzip -l "$OUTPUT" | grep "run.py" | head -1

echo ""
echo "Ready to upload at https://app.ainm.no/submit/norgesgruppen-data"
