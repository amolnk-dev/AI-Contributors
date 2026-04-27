#!/bin/bash
# Deploy overnight autoresearch to all running ainm-norgesgruppen* VMs.
#
# Usage:
#   scripts/gcp/deploy-overnight.sh              — deploy + start on all running VMs
#   scripts/gcp/deploy-overnight.sh --dry-run    — show what would be deployed

cd "$(dirname "$0")/../.."

DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then DRY_RUN=true; fi
done

# Find all running ainm-norgesgruppen* VMs (includes a100, a100-2, ar2, exp4, etc.)
VMS=$(gcloud compute instances list \
    --filter="name~^ainm-norgesgruppen AND status=RUNNING" \
    --format="csv[no-heading](name,zone,EXTERNAL_IP)" 2>/dev/null)

if [ -z "$VMS" ]; then
    echo "No running ainm-norgesgruppen VMs found."
    echo "Create VMs first with scripts/gcp/create-vm.sh norgesgruppen heavy --name=<name>"
    exit 0
fi

echo "=== Deploying overnight autoresearch ==="
echo ""

# Classify VMs by GPU type
while IFS=',' read -r VM ZONE IP; do
    # Determine GPU type from VM name or machine type
    GPU_TYPE="l4"  # default
    if echo "$VM" | grep -qi "a100"; then
        GPU_TYPE="a100"
    fi

    # Use VM name suffix as VM_ID
    VM_ID=$(echo "$VM" | sed 's/^ainm-norgesgruppen-*//' | sed 's/^ainm-norgesgruppen/main/')
    if [ -z "$VM_ID" ]; then VM_ID="main"; fi

    echo "VM: $VM  Zone: $ZONE  IP: $IP  GPU: $GPU_TYPE  VM_ID: $VM_ID"

    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would deploy and start autoresearch"
        echo ""
        continue
    fi

    # Deploy files
    (
        echo "  [$VM] Copying files..."
        gcloud compute scp --zone="$ZONE" \
            tasks/norgesgruppen/autoresearch_overnight.py \
            tasks/norgesgruppen/train.py \
            "$VM":~/task/ 2>/dev/null

        echo "  [$VM] Installing ultralytics if needed..."
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            pip list 2>/dev/null | grep -q ultralytics || \
            pip install --break-system-packages ultralytics==8.1.0 'numpy<2' 2>&1 | tail -2
        " 2>/dev/null

        echo "  [$VM] Stopping any existing autoresearch..."
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            pkill -f autoresearch_overnight.py 2>/dev/null; true
            sleep 2
        " 2>/dev/null

        echo "  [$VM] Starting autoresearch (VM_ID=$VM_ID, GPU=$GPU_TYPE)..."
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            cd ~/task && \
            VM_ID=$VM_ID GPU_TYPE=$GPU_TYPE \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            nohup python3 -u autoresearch_overnight.py > overnight.log 2>&1 &
            sleep 2
            if pgrep -f autoresearch_overnight.py > /dev/null; then
                echo 'STARTED OK'
            else
                echo 'FAILED TO START — check overnight.log'
                tail -20 overnight.log
            fi
        " 2>/dev/null

        echo "  [$VM] Done."
        echo ""
    ) &
done <<< "$VMS"

wait
echo "=== All deployments complete ==="
echo ""
echo "Monitor with:"
echo "  gcloud compute ssh <VM> --zone=<ZONE> --command='tail -50 ~/task/overnight.log'"
echo ""
echo "Collect results:"
echo "  bash scripts/gcp/collect-overnight.sh"
