#!/bin/bash
# Collect overnight autoresearch results and best models from all VMs.
#
# Usage:
#   scripts/gcp/collect-overnight.sh              — collect results + models
#   scripts/gcp/collect-overnight.sh --results     — results TSVs only (fast)
#   scripts/gcp/collect-overnight.sh --top N       — show top N configs

cd "$(dirname "$0")/../.."

RESULTS_ONLY=false
TOP_N=20

for arg in "$@"; do
    case $arg in
        --results) RESULTS_ONLY=true ;;
        --top) shift; TOP_N=${2:-20} ;;
    esac
done

DEST_DIR="tasks/norgesgruppen/models/overnight"
mkdir -p "$DEST_DIR"

# Find all running norgesgruppen VMs
VMS=$(gcloud compute instances list \
    --filter="name~^ainm-norgesgruppen AND status=RUNNING" \
    --format="csv[no-heading](name,zone)" 2>/dev/null)

if [ -z "$VMS" ]; then
    echo "No running ainm-norgesgruppen VMs found."
    # Still try to show existing results
    if ls tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | head -1 > /dev/null; then
        echo "Showing existing local results..."
    else
        exit 0
    fi
fi

echo "=== Collecting overnight results ==="
echo ""

# Download results TSVs and best models from each VM
while IFS=',' read -r VM ZONE; do
    [ -z "$VM" ] && continue
    VM_ID=$(echo "$VM" | sed 's/^ainm-norgesgruppen-*//' | sed 's/^ainm-norgesgruppen/main/')
    if [ -z "$VM_ID" ]; then VM_ID="main"; fi

    echo "--- $VM ($VM_ID) ---"

    # Check if autoresearch is still running
    RUNNING=$(gcloud compute ssh "$VM" --zone="$ZONE" --command="pgrep -f autoresearch_overnight.py > /dev/null && echo 'RUNNING' || echo 'STOPPED'" 2>/dev/null)
    echo "  Status: $RUNNING"

    # Download results TSV
    gcloud compute scp --zone="$ZONE" \
        "$VM":~/task/overnight_results_*.tsv \
        tasks/norgesgruppen/ 2>/dev/null && echo "  Results TSV: downloaded" || echo "  Results TSV: not found"

    if [ "$RESULTS_ONLY" = false ]; then
        # Download best model
        gcloud compute scp --zone="$ZONE" \
            "$VM":~/task/models/best_overnight_*.pt \
            "$DEST_DIR/" 2>/dev/null && echo "  Best model: downloaded" || echo "  Best model: not found"
    fi

    # Show last few results
    gcloud compute ssh "$VM" --zone="$ZONE" --command="
        if [ -f ~/task/overnight_results_*.tsv ]; then
            RUNS=\$(tail -n +2 ~/task/overnight_results_*.tsv | wc -l)
            BEST=\$(tail -n +2 ~/task/overnight_results_*.tsv | sort -t'\t' -k3 -rn | head -1 | cut -f3)
            echo \"  Runs: \$RUNS, Best: \$BEST\"
        fi
    " 2>/dev/null

    echo ""
done <<< "$VMS"

echo "=== Aggregated Results ==="
echo ""

# Merge all results and show top configs
if ls tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | head -1 > /dev/null; then
    echo "Top $TOP_N configs by val_metric:"
    echo ""
    # Header
    head -1 tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | head -1
    # All data rows sorted by val_metric (column 3)
    tail -q -n +2 tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | \
        grep -v "^$" | \
        sort -t$'\t' -k3 -rn | \
        head -"$TOP_N"
    echo ""
    echo "Total experiments: $(tail -q -n +2 tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | grep -v "^$" | wc -l | tr -d ' ')"
    echo "Successful: $(tail -q -n +2 tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | grep -E "kept|rejected" | wc -l | tr -d ' ')"
    echo "Failed/timeout: $(tail -q -n +2 tasks/norgesgruppen/overnight_results_*.tsv 2>/dev/null | grep -E "failed|timeout|crash" | wc -l | tr -d ' ')"
else
    echo "No results files found yet."
fi

echo ""
echo "Best models saved to: $DEST_DIR/"
ls -lh "$DEST_DIR"/*.pt 2>/dev/null || echo "(none yet)"
