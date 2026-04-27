#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== NM i AI 2026 — Setup ==="

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install deps
if command -v uv &> /dev/null; then
    echo "Installing with uv..."
    uv pip install -r requirements.txt
    for task in cv ml nlp; do
        uv pip install -r tasks/${task}/requirements.txt 2>/dev/null || true
    done
else
    echo "Installing with pip (consider installing uv for speed)..."
    pip install -r requirements.txt
    for task in cv ml nlp; do
        pip install -r tasks/${task}/requirements.txt 2>/dev/null || true
    done
fi

echo ""
echo "=== Setup complete ==="
echo "Activate: source .venv/bin/activate"
echo "Run task: cd tasks/cv && python api.py"
echo "Test:     pytest tasks/cv/tests/"
echo "Validate: bash scripts/validate.sh"
