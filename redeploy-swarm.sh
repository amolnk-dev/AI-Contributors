#!/usr/bin/env bash
# Fleet experiment runner for Astar Island
# Deploys code to GCP VMs and runs experiments in parallel
#
# Usage:
#   scripts/gcp/fleet-experiment.sh deploy          # Deploy code to all VMs
#   scripts/gcp/fleet-experiment.sh gen              # Generate experiment files
#   scripts/gcp/fleet-experiment.sh sweep            # Run all experiments
#   scripts/gcp/fleet-experiment.sh collect          # Collect results
#   scripts/gcp/fleet-experiment.sh all              # Gen + deploy + sweep + collect
#   scripts/gcp/fleet-experiment.sh check            # Check which VMs are idle
#
set -euo pipefail

PROJECT="ai-nm26osl-1823"
ZONE_A="us-central1-a"
ZONE_B="europe-west1-b"
TASK_DIR="tasks/astar-island"
REMOTE_DIR="/tmp/astar"  # Use /tmp — guaranteed writable, no path issues
EXP_DIR="$TASK_DIR/experiments"

# Available VMs (idle g2-standard-8 machines)
VMS_A=(ainm-norgesgruppen-exp5 ainm-norgesgruppen-exp6 ainm-norgesgruppen-exp7 ainm-norgesgruppen-exp8 ainm-norgesgruppen-exp9 ainm-norgesgruppen-exp10 ainm-norgesgruppen-exp11 ainm-norgesgruppen-exp12 ainm-norgesgruppen-exp13 ainm-norgesgruppen-cls)
VMS_B=(ainm-norgesgruppen-ar2 ainm-norgesgruppen-exp4)

ALL_VMS=("${VMS_A[@]}" "${VMS_B[@]}")

get_zone() {
  local vm=$1
  if [[ "$vm" == *ar2* ]] || [[ "$vm" == *exp4* ]]; then
    echo "$ZONE_B"
  else
    echo "$ZONE_A"
  fi
}

ssh_cmd() {
  local vm=$1 zone=$2; shift 2
  gcloud compute ssh "$vm" --zone="$zone" --project="$PROJECT" --command="$*" 2>/dev/null
}

gen() {
  echo "=== Generating experiment files ==="
  cd "$TASK_DIR"
  python3 gen_experiments.py
  cd - > /dev/null
  echo ""
  echo "Manifest:"
  cat "$EXP_DIR/manifest.txt"
}

deploy() {
  echo "=== Deploying to all VMs ==="
  local pids=()

  # Read manifest to know which experiments to deploy
  if [ ! -f "$EXP_DIR/manifest.txt" ]; then
    echo "ERROR: No manifest found. Run 'gen' first."
    exit 1
  fi

  for vm in "${ALL_VMS[@]}"; do
    zone=$(get_zone "$vm")
    echo "  $vm ($zone)..."
    (
      # Create clean remote dir
      ssh_cmd "$vm" "$zone" "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR/data $REMOTE_DIR/experiments"

      # Copy essential files
      for f in train.py model.py evaluate.py utils.py dtos.py simulator.py retrain_gbt.py; do
        gcloud compute scp "$TASK_DIR/$f" "$vm:$REMOTE_DIR/$f" --zone="$zone" --project="$PROJECT" 2>/dev/null
      done

      # Copy experiment files
      for f in "$EXP_DIR"/train_*.py; do
        [ -f "$f" ] && gcloud compute scp "$f" "$vm:$REMOTE_DIR/experiments/" --zone="$zone" --project="$PROJECT" 2>/dev/null
      done
      gcloud compute scp "$EXP_DIR/manifest.txt" "$vm:$REMOTE_DIR/experiments/" --zone="$zone" --project="$PROJECT" 2>/dev/null

      # Copy data files
      for f in $TASK_DIR/data/round*_initial.json $TASK_DIR/data/gt_r*_seed*.npy $TASK_DIR/data/obs_*.jsonl $TASK_DIR/data/gbt_models.pkl $TASK_DIR/data/calibration.json; do
        [ -f "$f" ] && gcloud compute scp "$f" "$vm:$REMOTE_DIR/data/" --zone="$zone" --project="$PROJECT" 2>/dev/null
      done

      # Install deps
      ssh_cmd "$vm" "$zone" "pip3 install --break-system-packages -q xgboost numpy scipy scikit-learn 2>/dev/null"

      # Verify
      ssh_cmd "$vm" "$zone" "cd $REMOTE_DIR && python3 -c 'import train; print(\"OK\")' 2>&1"
    ) &
    pids+=($!)
  done

  # Wait for all deploys
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || ((failed++))
  done
  echo "Deploy complete. $failed failures."
}

sweep() {
  echo "=== Running sweep across fleet ==="

  # Read manifest and assign experiments to VMs
  local exp_idx=0
  local vm_count=${#ALL_VMS[@]}

  while IFS='|' read -r name file desc; do
    if [ $exp_idx -ge $vm_count ]; then
      echo "  WARNING: More experiments ($((exp_idx+1))) than VMs ($vm_count), skipping: $name"
      ((exp_idx++))
      continue
    fi

    vm="${ALL_VMS[$exp_idx]}"
    zone=$(get_zone "$vm")
    short=$(echo "$vm" | sed 's/ainm-norgesgruppen-//')
    echo "  $short: $desc ($file)"

    # Copy experiment file as train.py and run it
    ssh_cmd "$vm" "$zone" "cd $REMOTE_DIR && cp experiments/$file train.py && python3 train.py > /tmp/result_${name}.log 2>&1" &

    ((exp_idx++))
  done < "$EXP_DIR/manifest.txt"

  echo "All $exp_idx experiments launched. Waiting..."
  wait
  echo "=== DONE ==="
}

collect() {
  echo "=== Collecting results ==="
  echo ""
  printf "%-15s %-12s %-12s %s\n" "EXPERIMENT" "WAVG" "UNWTD" "DESCRIPTION"
  printf "%s\n" "------------------------------------------------------------"

  local best_wavg=0
  local best_name=""

  while IFS='|' read -r name file desc; do
    # Find which VM ran this experiment (same order as sweep)
    local exp_idx=0
    local target_vm=""
    while IFS='|' read -r n f d; do
      if [ "$n" == "$name" ]; then
        target_vm="${ALL_VMS[$exp_idx]}"
        break
      fi
      ((exp_idx++))
    done < "$EXP_DIR/manifest.txt"

    if [ -z "$target_vm" ]; then
      printf "%-15s %-12s %-12s %s\n" "$name" "NO_VM" "-" "$desc"
      continue
    fi

    zone=$(get_zone "$target_vm")
    result=$(ssh_cmd "$target_vm" "$zone" "cat /tmp/result_${name}.log 2>/dev/null" || echo "NO_RESULT")
    wavg=$(echo "$result" | grep '^val_metric:' | head -1 | awk '{print $2}')
    unwtd=$(echo "$result" | grep '^val_metric_unweighted:' | head -1 | awk '{print $2}')

    printf "%-15s %-12s %-12s %s\n" "$name" "${wavg:-FAILED}" "${unwtd:--}" "$desc"

    # Track best
    if [ -n "$wavg" ] && [ "$wavg" != "FAILED" ]; then
      if python3 -c "exit(0 if $wavg > $best_wavg else 1)" 2>/dev/null; then
        best_wavg="$wavg"
        best_name="$name"
      fi
    fi
  done < "$EXP_DIR/manifest.txt"

  echo ""
  echo "BEST: $best_name = $best_wavg"
}

check() {
  echo "=== Fleet Status ==="
  printf "%-20s %-8s %-8s %-8s\n" "VM" "LOAD" "USERS" "PYTHON"
  printf "%s\n" "----------------------------------------------"
  for vm in "${ALL_VMS[@]}"; do
    zone=$(get_zone "$vm")
    short=$(echo "$vm" | sed 's/ainm-norgesgruppen-//')
    result=$(ssh_cmd "$vm" "$zone" "echo \$(cat /proc/loadavg | cut -d' ' -f1) \$(who | wc -l) \$(ps aux | grep 'python3.*train' | grep -v grep | wc -l)" 2>/dev/null || echo "OFFLINE 0 0")
    load=$(echo "$result" | awk '{print $1}')
    users=$(echo "$result" | awk '{print $2}')
    python=$(echo "$result" | awk '{print $3}')
    status=""
    if [ "$users" != "0" ] || [ "$python" != "0" ]; then status="IN USE"; fi
    printf "%-20s %-8s %-8s %-8s %s\n" "$short" "$load" "$users" "$python" "$status"
  done
}

case "${1:-help}" in
  gen)     gen ;;
  deploy)  deploy ;;
  sweep)   sweep ;;
  collect) collect ;;
  check)   check ;;
  all)     gen && deploy && sweep && echo "Waiting 150s for experiments to complete..." && sleep 150 && collect ;;
  *)
    echo "Usage: $0 {gen|deploy|sweep|collect|check|all}"
    echo "  gen     - Generate experiment files (Python-based, not sed)"
    echo "  deploy  - Push code + experiments to all VMs"
    echo "  sweep   - Run one experiment per VM in parallel"
    echo "  collect - Gather and compare results"
    echo "  check   - Check which VMs are idle"
    echo "  all     - Gen + deploy + sweep + wait + collect"
    ;;
esac
