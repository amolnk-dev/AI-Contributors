#!/bin/bash
# Deploy task code to a GCP VM and start the API.
#
# Usage:
#   scripts/gcp/deploy-task.sh cv              — deploy to ainm-cv
#   scripts/gcp/deploy-task.sh cv --name=vit   — deploy to ainm-cv-vit

cd "$(dirname "$0")/../.."

TASK=$1
SUFFIX=""
shift 2>/dev/null || true
for arg in "$@"; do
    case $arg in --name=*) SUFFIX="-${arg#--name=}" ;; esac
done

if [ -z "$TASK" ]; then
    echo "Usage: scripts/gcp/deploy-task.sh <cv|ml|nlp> [--name=suffix]"
    exit 1
fi

VM="ainm-${TASK}${SUFFIX}"

# Auto-detect zone
ZONE=$(gcloud compute instances list --filter="name=${VM}" --format="value(zone)" 2>/dev/null)
if [ -z "$ZONE" ]; then
    echo "ERROR: VM ${VM} not found. Create it first with scripts/gcp/create-vm.sh ${TASK}"
    exit 1
fi

echo "Deploying ${TASK} to ${VM} (zone: ${ZONE})..."

# Create target directories on VM
gcloud compute ssh "${VM}" --zone="${ZONE}" --command="mkdir -p ~/task ~/shared"

# Copy code to VM
gcloud compute scp --zone="${ZONE}" --recurse "./tasks/${TASK}/"* "${VM}:~/task/"
gcloud compute scp --zone="${ZONE}" --recurse ./shared/* "${VM}:~/shared/" 2>/dev/null || true
gcloud compute scp --zone="${ZONE}" ./requirements.txt "${VM}:~/requirements.txt"

# Install deps (Deep Learning VM has PyTorch+CUDA pre-installed)
PORT=$((9049 + $(echo "cv ml nlp" | tr ' ' '\n' | grep -n "^${TASK}$" | cut -d: -f1)))
gcloud compute ssh "${VM}" --zone="${ZONE}" --command="pip install -q --break-system-packages \
    fastapi uvicorn pydantic transformers scikit-learn xgboost lightgbm \
    opencv-python Pillow optuna wandb numpy pandas httpx pytest scipy 2>&1 | tail -3; true"

# Stop old API, start new one via systemd (survives SSH disconnect)
gcloud compute ssh "${VM}" --zone="${ZONE}" --command="systemctl stop ainm-api.service 2>/dev/null; true"
gcloud compute ssh "${VM}" --zone="${ZONE}" --command="systemd-run --unit=ainm-api --remain-after-exit bash -c 'cd /root/task && python3 api.py > /tmp/api.log 2>&1'"

# Wait for startup and verify
sleep 3
IP=$(gcloud compute instances describe "${VM}" --zone="${ZONE}" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
HEALTH=$(gcloud compute ssh "${VM}" --zone="${ZONE}" --command="curl -s http://localhost:${PORT}/" 2>/dev/null)

echo ""
echo "VM:       ${VM} (${ZONE})"
echo "Endpoint: http://${IP}:${PORT}"
echo "Health:   ${HEALTH}"
