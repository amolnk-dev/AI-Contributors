#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

TASK=$1
SUFFIX=""
shift 2>/dev/null || true
for arg in "$@"; do
    case $arg in --name=*) SUFFIX="-${arg#--name=}" ;; esac
done

if [ -z "$TASK" ]; then
    echo "Usage: scripts/gcp/train-task.sh <cv|ml|nlp> [--name=suffix]"
    exit 1
fi

VM="ainm-${TASK}${SUFFIX}"

# Auto-detect zone
ZONE=$(gcloud compute instances list --filter="name=${VM}" --format="value(zone)" 2>/dev/null)
if [ -z "$ZONE" ]; then
    echo "ERROR: VM ${VM} not found. Create it first with scripts/gcp/create-vm.sh ${TASK}"
    exit 1
fi

echo "Running training for ${TASK} on ${VM} (zone: ${ZONE})..."

# Run training
gcloud compute ssh "${VM}" --zone="${ZONE}" -- bash -c "
    cd ~/task
    TASK_MODE=train python3 train.py
"

# Pull trained models back
echo "Downloading models..."
mkdir -p "./tasks/${TASK}/models/"
gcloud compute scp --zone="${ZONE}" --recurse "${VM}:~/task/models/" "./tasks/${TASK}/models/"

echo "Models downloaded to tasks/${TASK}/models/"
