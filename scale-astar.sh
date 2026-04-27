#!/bin/bash
# Spin up the full GPU fleet for competition.
#
# Usage:
#   scripts/gcp/fleet-up.sh baseline          — 3 VMs (1 per task, for rapid baselines)
#   scripts/gcp/fleet-up.sh overnight         — 7-9 VMs (parallel autoresearch branches)
#   scripts/gcp/fleet-up.sh max               — All A100s, max parallelism
#   scripts/gcp/fleet-up.sh custom cv:heavy ml:light nlp:heavy  — Specify per-task
#
# All modes support --spot flag to use preemptible pricing (70% cheaper).

cd "$(dirname "$0")/../.."

MODE=${1:-baseline}
SPOT_FLAG=""

# Check for --spot anywhere in args
for arg in "$@"; do
    if [ "$arg" = "--spot" ]; then SPOT_FLAG="--spot"; fi
done

case $MODE in
    baseline)
        echo "=== Fleet: Baseline (3 VMs) ==="
        bash scripts/gcp/create-vm.sh cv medium $SPOT_FLAG &
        bash scripts/gcp/create-vm.sh ml light $SPOT_FLAG &
        bash scripts/gcp/create-vm.sh nlp medium $SPOT_FLAG &
        wait
        ;;

    overnight)
        echo "=== Fleet: Overnight Autoresearch (7+ VMs) ==="
        echo "Spinning up multiple VMs per task for parallel experiments..."
        # Primary VMs (A100 for main experiments)
        bash scripts/gcp/create-vm.sh cv heavy $SPOT_FLAG &
        bash scripts/gcp/create-vm.sh ml heavy $SPOT_FLAG &
        bash scripts/gcp/create-vm.sh nlp heavy $SPOT_FLAG &
        wait
        # Secondary VMs (for parallel experiment branches)
        bash scripts/gcp/create-vm.sh cv heavy $SPOT_FLAG --name=b &
        bash scripts/gcp/create-vm.sh ml light $SPOT_FLAG --name=b &
        bash scripts/gcp/create-vm.sh nlp heavy $SPOT_FLAG --name=b &
        wait
        # Optional third CV/NLP if quota allows
        bash scripts/gcp/create-vm.sh cv medium $SPOT_FLAG --name=c 2>/dev/null &
        bash scripts/gcp/create-vm.sh nlp medium $SPOT_FLAG --name=c 2>/dev/null &
        wait
        ;;

    max)
        echo "=== Fleet: Maximum Compute ==="
        # 3x A100 per task + extras
        for TASK in cv ml nlp; do
            bash scripts/gcp/create-vm.sh $TASK heavy $SPOT_FLAG &
            bash scripts/gcp/create-vm.sh $TASK heavy $SPOT_FLAG --name=b &
            bash scripts/gcp/create-vm.sh $TASK heavy $SPOT_FLAG --name=c &
        done
        wait
        ;;

    custom)
        echo "=== Fleet: Custom ==="
        shift  # remove "custom"
        for spec in "$@"; do
            if [ "$spec" = "--spot" ]; then continue; fi
            TASK=$(echo "$spec" | cut -d: -f1)
            TIER=$(echo "$spec" | cut -d: -f2)
            NAME=$(echo "$spec" | cut -d: -f3)
            NAME_FLAG=""
            if [ -n "$NAME" ]; then NAME_FLAG="--name=$NAME"; fi
            bash scripts/gcp/create-vm.sh $TASK $TIER $SPOT_FLAG $NAME_FLAG &
        done
        wait
        ;;

    *)
        echo "Usage: scripts/gcp/fleet-up.sh <baseline|overnight|max|custom> [--spot]"
        echo ""
        echo "Modes:"
        echo "  baseline   3 VMs: cv=L4, ml=T4, nlp=L4"
        echo "  overnight  7-9 VMs: multiple A100s per task for parallel autoresearch"
        echo "  max        9 VMs: 3x A100 per task"
        echo "  custom     Specify per-task: cv:heavy ml:light nlp:heavy:branch2"
        echo ""
        echo "Cost estimates (per hour, all 3 tasks):"
        echo "  baseline:              ~\$1.75/hr  (spot: ~\$0.53/hr)"
        echo "  overnight (7 VMs):     ~\$18/hr    (spot: ~\$5.50/hr)"
        echo "  max (9x A100):         ~\$27/hr    (spot: ~\$8/hr)"
        exit 1
        ;;
esac

echo ""
echo "=== Fleet is up. Running status check... ==="
bash scripts/gcp/status.sh
