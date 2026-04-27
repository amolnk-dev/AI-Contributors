#!/bin/bash
# FAST redeploy: upload to GCS once, VMs pull from GCS.
# ~3 min for 900+ VMs vs ~30 min with SCP.
#
# Usage:
#   scripts/gcp/fast-redeploy.sh              # all ainm-astar-* VMs
#   scripts/gcp/fast-redeploy.sh --test e1    # just ainm-astar-e1

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_DIR="${REPO_DIR}/tasks/astar-island"
PROJECT="ai-nm26osl-1823"
BUCKET="gs://ainm-astar-data"
PARALLEL=80
LOGDIR="/tmp/swarm-deploy-logs"
rm -rf "$LOGDIR" && mkdir -p "$LOGDIR"

# Step 1: Build tarball
echo "Building tarball..."
cd "$TASK_DIR"
tar czf /tmp/astar-deploy.tar.gz \
    autoresearch_swarm.py params.py train.py model.py evaluate.py dtos.py utils.py simulator.py client.py \
    data/gt_r*.npy data/round*_initial.json data/obs_*.jsonl data/calibration.json data/gbt_models.pkl 2>/dev/null
echo "  $(du -h /tmp/astar-deploy.tar.gz | cut -f1)"

# Step 2: Upload to GCS (once)
echo "Uploading to GCS..."
gsutil -q cp /tmp/astar-deploy.tar.gz "${BUCKET}/deploy.tar.gz"
gsutil -q cp "${REPO_DIR}/scripts/gcp/swarm-start-remote.sh" "${BUCKET}/swarm-start-remote.sh"
echo "  Done"

# Step 3: Deploy to VMs
deploy_vm() {
    local VM=$1
    local ZONE=$2
    local LOG="$LOGDIR/${VM}.log"

    gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
        --command="gsutil -q cp ${BUCKET}/deploy.tar.gz /tmp/astar-deploy.tar.gz && gsutil -q cp ${BUCKET}/swarm-start-remote.sh /tmp/swarm-start-remote.sh && bash /tmp/swarm-start-remote.sh" \
        </dev/null >"$LOG" 2>&1

    if grep -q "STARTED" "$LOG"; then
        echo "[OK] $VM"
    else
        echo "[FAIL] $VM"
    fi
}

if [ "${1:-}" = "--test" ]; then
    VM="ainm-astar-${2:?need VM suffix}"
    ZONE=$(gcloud compute instances list --project="$PROJECT" \
        --filter="name=${VM}" --format="value(zone)" 2>/dev/null)
    [ -z "$ZONE" ] && { echo "VM $VM not found"; exit 1; }
    deploy_vm "$VM" "$ZONE"
    cat "$LOGDIR/${VM}.log"
    exit 0
fi

echo "Discovering VMs..."
VMS=$(gcloud compute instances list --project="$PROJECT" \
    --filter="name~^ainm-astar AND status:RUNNING" \
    --format="csv[no-heading](name,zone)" 2>/dev/null)
TOTAL=$(echo "$VMS" | grep -c . || echo 0)
echo "Found ${TOTAL} VMs. Deploying ${PARALLEL} at a time..."

COUNT=0
while IFS=, read -r VM ZONE; do
    [ -z "$VM" ] && continue
    deploy_vm "$VM" "$ZONE" &
    COUNT=$((COUNT + 1))
    if [ $((COUNT % PARALLEL)) -eq 0 ]; then
        wait
        OK=$(grep -rl "STARTED" "$LOGDIR"/ 2>/dev/null | wc -l | tr -d ' ')
        echo "--- $OK/$COUNT deployed ---"
    fi
done <<< "$VMS"
wait

OK=$(grep -rl "STARTED" "$LOGDIR"/ 2>/dev/null | wc -l | tr -d ' ')
FAIL=$((COUNT - OK))
echo ""
echo "=== $OK/$COUNT succeeded, $FAIL failed ==="
