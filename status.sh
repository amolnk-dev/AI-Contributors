#!/bin/bash
# Redeploy autoresearch swarm to astar VMs.
# 2-step: SCP tarball+script, SSH run script. ~20s per VM.
#
# Usage:
#   scripts/gcp/redeploy-swarm.sh ainm-astar-e1           # single VM
#   scripts/gcp/redeploy-swarm.sh --all                    # all VMs, 30 parallel
#   scripts/gcp/redeploy-swarm.sh --all --parallel 50      # 50 at a time

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_DIR="${REPO_DIR}/tasks/astar-island"
PROJECT="ai-nm26osl-1823"
TARBALL="/tmp/astar-deploy.tar.gz"
SCRIPT="${REPO_DIR}/scripts/gcp/swarm-start-remote.sh"
PARALLEL=30
LOGDIR="/tmp/swarm-deploy-logs"
mkdir -p "$LOGDIR"

build_tarball() {
    echo "Building deploy tarball..."
    cd "$TASK_DIR"
    tar czf "$TARBALL" \
        autoresearch_swarm.py params.py train.py model.py evaluate.py dtos.py utils.py simulator.py client.py \
        data/gt_r*.npy data/round*_initial.json data/obs_*.jsonl data/calibration.json data/gbt_models.pkl 2>/dev/null
    echo "Tarball: $(du -h "$TARBALL" | cut -f1)"
    cd "$REPO_DIR"
}

deploy_one_vm() {
    local VM=$1
    local ZONE=$2
    local LOG="$LOGDIR/${VM}.log"

    {
        gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
            "$TARBALL" "$SCRIPT" "${VM}:/tmp/" 2>&1 && \
        gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
            --command='bash /tmp/swarm-start-remote.sh' 2>&1
    } > "$LOG" 2>&1

    if grep -q "STARTED" "$LOG"; then
        echo "[OK] $VM"
    else
        echo "[FAIL] $VM — see $LOG"
    fi
}

# ── Main ──
if [ "${1:-}" = "--all" ]; then
    shift
    while [ $# -gt 0 ]; do
        case $1 in --parallel) PARALLEL=$2; shift 2;; *) shift;; esac
    done

    build_tarball

    echo "Discovering VMs..."
    VMS=$(gcloud compute instances list --project="$PROJECT" \
        --filter="name~^ainm-astar AND status:RUNNING" \
        --format="csv[no-heading](name,zone)" 2>/dev/null)
    TOTAL=$(echo "$VMS" | grep -c . || echo 0)
    echo "Found ${TOTAL} VMs. Deploying ${PARALLEL} at a time..."
    echo ""

    COUNT=0
    BATCH=0
    while IFS=, read -r VM ZONE; do
        [ -z "$VM" ] && continue
        deploy_one_vm "$VM" "$ZONE" &
        COUNT=$((COUNT + 1))
        if [ $((COUNT % PARALLEL)) -eq 0 ]; then
            BATCH=$((BATCH + 1))
            wait
            echo "--- Batch $BATCH done ($COUNT/$TOTAL) ---"
        fi
    done <<< "$VMS"
    wait

    OK=$(grep -rl "STARTED" "$LOGDIR"/ 2>/dev/null | wc -l | tr -d ' ')
    FAIL=$((TOTAL - OK))
    echo ""
    echo "=== Done: $OK succeeded, $FAIL failed out of $TOTAL ==="
    if [ "$FAIL" -gt 0 ]; then
        echo "Failed VMs:"
        for f in "$LOGDIR"/*.log; do
            grep -L "STARTED" "$f" 2>/dev/null | sed 's|.*/||;s|\.log||'
        done
    fi
else
    VM="${1:?Usage: redeploy-swarm.sh <vm-name> or --all}"
    ZONE=$(gcloud compute instances list --project="$PROJECT" \
        --filter="name=${VM}" --format="value(zone)" 2>/dev/null)
    [ -z "$ZONE" ] && { echo "ERROR: VM ${VM} not found"; exit 1; }

    build_tarball
    deploy_one_vm "$VM" "$ZONE"
    cat "$LOGDIR/${VM}.log"
fi
