#!/usr/bin/env bash
# Reusable GCP sweep orchestrator for astar-island LORO experiments.
# Usage: scripts/gcp/sweep-astar.sh [--teardown]
set -euo pipefail

ZONE="europe-west4-a"
VM_NAME="ainm-astar-sweep"
MACHINE_TYPE="c2-standard-30"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DIR="~/astar-island"

TEARDOWN=false
for arg in "$@"; do
  [[ "$arg" == "--teardown" ]] && TEARDOWN=true
done

# --- Step 1: Ensure VM exists ---
STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
if [[ "$STATUS" == "NOT_FOUND" ]]; then
  echo "Creating $VM_NAME ($MACHINE_TYPE) in $ZONE..."
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB \
    --quiet
  echo "Waiting for VM to boot..."
  sleep 15
elif [[ "$STATUS" == "TERMINATED" ]]; then
  echo "Starting stopped VM $VM_NAME..."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE" --quiet
  sleep 10
else
  echo "VM $VM_NAME already running."
fi

# --- Step 2: Deploy code ---
echo "Deploying astar-island code..."
gcloud compute scp --recurse "$PROJECT_ROOT/tasks/astar-island/"* "$VM_NAME:$REMOTE_DIR/" --zone="$ZONE"

# --- Step 3: Install deps ---
echo "Installing dependencies..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command=\
  "pip3 install --break-system-packages -q xgboost numpy scipy scikit-learn pydantic matplotlib httpx 2>&1 | tail -1"

# --- Step 4: Run all sweeps in parallel ---
echo ""
echo "========================================="
echo "  Running sweeps on $VM_NAME (30 vCPUs)"
echo "========================================="
echo ""

# Define sweep commands — edit these values for each round
# PYTHONUNBUFFERED=1 ensures real-time output to log files
PY="PYTHONUNBUFFERED=1 python3"
CLS0="$PY sweep_l7_class.py 0 1.0,1.1,1.2,1.3,1.4,1.5 0.30 > /tmp/sweep_cls0.log 2>&1"
CLS1="$PY sweep_l7_class.py 1 0.7,0.8,0.9,1.0,1.1,1.2 0.30 > /tmp/sweep_cls1.log 2>&1"
CLS3="$PY sweep_l7_class.py 3 0.6,0.7,0.8,0.9,1.0 0.30 > /tmp/sweep_cls3.log 2>&1"
CLS4="$PY sweep_l7_class.py 4 0.9,1.0,1.1,1.2,1.3,1.4 0.30 > /tmp/sweep_cls4.log 2>&1"
GBT="$PY sweep_gbt_blend.py 0.20,0.25,0.30,0.35,0.40 > /tmp/sweep_gbt.log 2>&1"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command=\
  "(cd $REMOTE_DIR && $CLS0) & \
   (cd $REMOTE_DIR && $CLS1) & \
   (cd $REMOTE_DIR && $CLS3) & \
   (cd $REMOTE_DIR && $CLS4) & \
   (cd $REMOTE_DIR && $GBT) & \
   wait && echo 'ALL SWEEPS DONE'"

# --- Step 5: Collect results ---
echo ""
echo "========================================="
echo "  Results"
echo "========================================="
echo ""

for log in sweep_cls0 sweep_cls1 sweep_cls3 sweep_cls4 sweep_gbt; do
  echo "--- $log ---"
  gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="cat /tmp/${log}.log" 2>/dev/null
  echo ""
done

# --- Step 6: Teardown ---
if $TEARDOWN; then
  echo "Tearing down $VM_NAME..."
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
  echo "VM deleted."
else
  echo "VM $VM_NAME still running. Use --teardown to delete."
fi
