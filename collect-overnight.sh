#!/bin/bash
# Collect autoresearch results from all astar swarm VMs.
#
# Usage:
#   bash scripts/gcp/collect-astar.sh           # Pull + show results
#   bash scripts/gcp/collect-astar.sh --top 20   # Show top 20
#   bash scripts/gcp/collect-astar.sh --best      # Download best train.py
#
# VMs: ainm-astar-autoresearch, ainm-astar-swarm-a, ainm-astar-swarm-b

set -euo pipefail

PROJECT="ai-nm26osl-1823"
ZONE="europe-west4-a"
TASK_DIR="tasks/astar-island"
RESULTS_DIR="$TASK_DIR/gcp_results"
mkdir -p "$RESULTS_DIR"

VMS=(
  "ainm-astar-autoresearch:beast176"
  "ainm-astar-swarm-a:swarm-a"
  "ainm-astar-swarm-b:swarm-b"
  "ainm-astar-s1:s1"
  "ainm-astar-s2:s2"
  "ainm-astar-s3:s3"
  "ainm-astar-s4:s4"
  "ainm-astar-s5:s5"
  "ainm-astar-s6:s6"
  "ainm-astar-s7:s7"
  "ainm-astar-s8:s8"
  "ainm-astar-s9:s9"
  "ainm-astar-s10:s10"
  "ainm-astar-n1:n1"
  "ainm-astar-n2:n2"
  "ainm-astar-n3:n3"
  "ainm-astar-n4:n4"
  "ainm-astar-n5:n5"
  "ainm-astar-n6:n6"
  "ainm-astar-n7:n7"
  "ainm-astar-n8:n8"
  "ainm-astar-n9:n9"
  "ainm-astar-n10:n10"
  "ainm-astar-c1:c1"
  "ainm-astar-c2:c2"
  "ainm-astar-c3:c3"
  "ainm-astar-c4:c4"
  "ainm-astar-c5:c5"
)

TOP_N=${2:-10}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ASTAR SWARM COLLECTOR                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 1. Check status and pull results from each VM
for VM_PAIR in "${VMS[@]}"; do
  VM=$(echo "$VM_PAIR" | cut -d: -f1)
  VMID=$(echo "$VM_PAIR" | cut -d: -f2)

  echo "--- $VM ($VMID) ---"

  # Check if running
  STATUS=$(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
    --command='pgrep -f autoresearch_swarm > /dev/null 2>&1 && echo "RUNNING" || echo "IDLE"' 2>/dev/null || echo "UNREACHABLE")

  echo "  Status: $STATUS"

  # Pull results TSV
  gcloud compute scp "$VM:/tmp/astar/swarm_results_${VMID}.tsv" \
    "$RESULTS_DIR/" --zone="$ZONE" --project="$PROJECT" 2>/dev/null && \
    echo "  Results: pulled" || echo "  Results: none"

  # Pull latest log tail
  TAIL=$(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
    --command='tail -3 /tmp/astar/swarm.log 2>/dev/null' 2>/dev/null || echo "no log")
  echo "  Last log: $TAIL"
  echo ""
done

# 2. Merge all results
echo "Merging results..."
MERGED="$RESULTS_DIR/merged_results.tsv"
echo -e "timestamp\tvm_id\tval_metric\tduration_min\tstatus\tdescription" > "$MERGED"

for f in "$RESULTS_DIR"/swarm_results_*.tsv; do
  [ -f "$f" ] && tail -n +2 "$f" >> "$MERGED" 2>/dev/null
done

TOTAL=$(tail -n +2 "$MERGED" | wc -l | tr -d ' ')
KEPT=$(grep -c "kept" "$MERGED" 2>/dev/null || echo "0")
echo "Total experiments: $TOTAL | Improvements: $KEPT"
echo ""

# 3. Show top results
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TOP $TOP_N RESULTS (sorted by val_metric)               ║"
echo "╠══════════════════════════════════════════════════════════╣"

tail -n +2 "$MERGED" | sort -t$'\t' -k3 -rn | head -n "$TOP_N" | while IFS=$'\t' read -r ts vmid metric dur status desc; do
  printf "║  %-8s  %8.4f  %-8s  %s\n" "$vmid" "$metric" "$status" "${desc:0:40}"
done

echo "╚══════════════════════════════════════════════════════════╝"

# 4. Download best train.py if requested
if [ "${1:-}" = "--best" ]; then
  BEST_VM=""
  BEST_METRIC="0"

  for VM_PAIR in "${VMS[@]}"; do
    VM=$(echo "$VM_PAIR" | cut -d: -f1)
    VMID=$(echo "$VM_PAIR" | cut -d: -f2)

    METRIC=$(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
      --command='grep "kept" /tmp/astar/swarm_results_*.tsv 2>/dev/null | sort -t"	" -k3 -rn | head -1 | cut -f3' 2>/dev/null || echo "0")

    if [ "$(echo "$METRIC > $BEST_METRIC" | bc 2>/dev/null || echo 0)" = "1" ]; then
      BEST_METRIC="$METRIC"
      BEST_VM="$VM"
    fi
  done

  if [ -n "$BEST_VM" ]; then
    echo ""
    echo "Best VM: $BEST_VM (metric: $BEST_METRIC)"
    echo "Downloading best train.py..."
    gcloud compute scp "$BEST_VM:/tmp/astar/train.py.best" \
      "$TASK_DIR/train.py.best_from_swarm" --zone="$ZONE" --project="$PROJECT" \
      --ssh-flag="-o StrictHostKeyChecking=no" 2>/dev/null && \
      echo "Saved: $TASK_DIR/train.py.best_from_swarm" || echo "Download failed"
  fi
fi

# 5. Verify + integrate if requested
if [ "${1:-}" = "--integrate" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  INTEGRATION PIPELINE                                   ║"
  echo "╚══════════════════════════════════════════════════════════╝"

  # Download best train.py from winning VM
  BEST_VM=""
  BEST_METRIC="0"
  for VM_PAIR in "${VMS[@]}"; do
    VM=$(echo "$VM_PAIR" | cut -d: -f1)
    VMID=$(echo "$VM_PAIR" | cut -d: -f2)
    METRIC=$(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
      --ssh-flag="-o StrictHostKeyChecking=no" \
      --command='grep "kept" /tmp/astar/swarm_results_*.tsv 2>/dev/null | sort -t"	" -k3 -rn | head -1 | cut -f3' 2>/dev/null || echo "0")
    if [ "$(echo "$METRIC > $BEST_METRIC" | bc 2>/dev/null || echo 0)" = "1" ]; then
      BEST_METRIC="$METRIC"
      BEST_VM="$VM"
    fi
  done

  if [ -z "$BEST_VM" ]; then
    echo "No kept results found. Nothing to integrate."
    exit 0
  fi

  echo "Step 1: Download best train.py from $BEST_VM (metric: $BEST_METRIC)"
  gcloud compute scp "$BEST_VM:/tmp/astar/train.py.best" \
    "$TASK_DIR/train.py.best_from_swarm" --zone="$ZONE" --project="$PROJECT" \
    --ssh-flag="-o StrictHostKeyChecking=no" 2>/dev/null

  echo "Step 2: Verify locally with full LORO..."
  # Backup current train.py, swap in swarm best, run LORO
  cp "$TASK_DIR/train.py" "$TASK_DIR/train.py.pre_swarm_backup"
  cp "$TASK_DIR/train.py.best_from_swarm" "$TASK_DIR/train.py"

  cd "$TASK_DIR"
  source ../../.venv/bin/activate 2>/dev/null || true
  RESULT=$(python train.py 2>&1)
  VAL_METRIC=$(echo "$RESULT" | grep "^val_metric:" | head -1 | awk '{print $2}')

  echo "  Swarm best LORO (local verify): $VAL_METRIC"
  echo "  VM reported: $BEST_METRIC"
  echo ""

  # Restore original
  cp "$TASK_DIR/train.py.pre_swarm_backup" "$TASK_DIR/train.py"

  echo "$RESULT" | grep "^round_"

  echo ""
  echo "Step 3: To apply, run:"
  echo "  cp $TASK_DIR/train.py.best_from_swarm $TASK_DIR/train.py"
  echo "  # Then port parameter changes to model.py build_prediction()"
  echo "  # Then: python retrain_gbt.py"
  echo "  # Then: ASTAR_TOKEN=... python run.py"
fi
