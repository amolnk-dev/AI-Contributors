#!/bin/bash
# Deploy task code to all running ainm-* VMs for a given task (or all tasks).
#
# Usage:
#   scripts/gcp/fleet-deploy.sh         — deploy all tasks to all their VMs
#   scripts/gcp/fleet-deploy.sh cv      — deploy cv to all ainm-cv* VMs

cd "$(dirname "$0")/../.."

TASK_FILTER=$1

if [ -n "$TASK_FILTER" ]; then
    FILTER="name~^ainm-${TASK_FILTER}"
else
    FILTER="name~^ainm-"
fi

VMS=$(gcloud compute instances list --filter="$FILTER AND status=RUNNING" --format="csv[no-heading](name)" 2>/dev/null)

if [ -z "$VMS" ]; then
    echo "No running VMs found."
    exit 0
fi

echo "Deploying to all matching VMs..."
while IFS= read -r VM; do
    # Extract task from VM name (ainm-cv-vit → cv)
    TASK=$(echo "$VM" | sed 's/^ainm-//' | cut -d- -f1)
    SUFFIX=$(echo "$VM" | sed "s/^ainm-${TASK}//" | sed 's/^-//')
    NAME_FLAG=""
    if [ -n "$SUFFIX" ]; then NAME_FLAG="--name=$SUFFIX"; fi

    echo "--- Deploying ${TASK} to ${VM} ---"
    bash scripts/gcp/deploy-task.sh "$TASK" $NAME_FLAG &
done <<< "$VMS"

wait
echo ""
echo "=== All deployments complete ==="
bash scripts/gcp/status.sh
