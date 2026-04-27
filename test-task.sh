#!/bin/bash
set -e

cd "$(dirname "$0")/.."

CATEGORY=$1
TASK_NAME=$2

if [ -z "$CATEGORY" ] || [ -z "$TASK_NAME" ]; then
    echo "Usage: scripts/import-task.sh <cv|ml|nlp> <task-name>"
    exit 1
fi

echo "=== Importing task: ${TASK_NAME} (${CATEGORY}) ==="

# Update README
cat > "tasks/${CATEGORY}/README.md" << EOF
# Task: ${TASK_NAME}

**Type**: ${CATEGORY^^}
**Port**: $((9049 + $(echo "cv ml nlp" | tr ' ' '\n' | grep -n "^${CATEGORY}$" | cut -d: -f1)))

## Problem Description
[Paste task description here]

## Current Approach
[Description of current model/strategy]

## Score
- Baseline: —
- Current best: —
EOF

# Create task analysis doc
ANALYSIS_FILE="docs/task-analysis/${CATEGORY}-analysis.md"
if [ -f "$ANALYSIS_FILE" ]; then
    sed -i.bak "s/\[Paste task description.*\]/[Paste ${TASK_NAME} description here]/" "$ANALYSIS_FILE"
    rm -f "${ANALYSIS_FILE}.bak"
fi

# Create exec plan
mkdir -p docs/exec-plans/active
cat > "docs/exec-plans/active/EP-${CATEGORY}.md" << EOF
# Execution Plan: ${TASK_NAME}

**Status**: active
**Task**: ${CATEGORY}
**Created**: $(date +%Y-%m-%d)
**Target score**: TBD

## Objective
[What are we trying to achieve?]

## Approach
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Architecture
[Model architecture, data pipeline, key decisions]

## Risks
- [Risk 1]

## Success Criteria
- [ ] API returns valid responses
- [ ] Score > X on validation set
- [ ] Inference within timeout
- [ ] No prohibited API calls
EOF

echo ""
echo "Task ${TASK_NAME} imported to tasks/${CATEGORY}/"
echo ""
echo "Next steps:"
echo "  1. Edit tasks/${CATEGORY}/dtos.py with the actual request/response schema"
echo "  2. Fill docs/task-analysis/${CATEGORY}-analysis.md"
echo "  3. Implement model in tasks/${CATEGORY}/model.py"
