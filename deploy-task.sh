#!/bin/bash
set -e

# Usage: scripts/gcp/create-vm.sh <cv|ml|nlp> [light|medium|heavy] [--spot] [--name=suffix]
# Examples:
#   scripts/gcp/create-vm.sh cv medium              → ainm-cv (L4, on-demand)
#   scripts/gcp/create-vm.sh cv heavy --spot         → ainm-cv (A100, preemptible)
#   scripts/gcp/create-vm.sh cv heavy --name=vit     → ainm-cv-vit (A100, for parallel experiments)

TASK=$1
TIER=${2:-heavy}
SPOT=""
SUFFIX=""

shift 2 2>/dev/null || true
for arg in "$@"; do
    case $arg in
        --spot) SPOT="--provisioning-model=SPOT --instance-termination-action=STOP" ;;
        --name=*) SUFFIX="-${arg#--name=}" ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: scripts/gcp/create-vm.sh <cv|ml|nlp> [light|medium|heavy] [--spot] [--name=suffix]"
    echo ""
    echo "Tiers:"
    echo "  light   n1-standard-8 + T4 (16GB)    ~\$0.35/hr  (spot: ~\$0.11/hr)"
    echo "  medium  g2-standard-8 + L4 (24GB)    ~\$0.70/hr  (spot: ~\$0.21/hr)"
    echo "  heavy   a2-highgpu-1g + A100 (40GB)   ~\$2.95/hr  (spot: ~\$0.89/hr)"
    echo ""
    echo "Options:"
    echo "  --spot         Use preemptible/spot pricing (70% cheaper, may be reclaimed)"
    echo "  --name=suffix  Custom VM name suffix for parallel experiments"
    exit 1
fi

VM="ainm-${TASK}${SUFFIX}"

case $TIER in
    light)  MACHINE="n1-standard-8"  ACCEL="--accelerator=type=nvidia-tesla-t4,count=1" ;;
    medium) MACHINE="g2-standard-8"  ACCEL="" ;;  # g2 includes L4 GPU
    heavy)  MACHINE="a2-highgpu-1g"  ACCEL="" ;;  # a2-highgpu includes A100 GPU
    *)      echo "Unknown tier: $TIER"; exit 1 ;;
esac

# Zones to try in order (prefer Europe, fallback to US)
ZONES="europe-west4-a europe-west4-b europe-west1-b us-central1-f us-central1-a us-central1-b us-central1-c us-east1-b us-west1-b"

SPOT_LABEL=""
if [ -n "$SPOT" ]; then SPOT_LABEL=" [SPOT]"; fi
echo "Creating VM ${VM} (${TIER}: ${MACHINE}${SPOT_LABEL})..."

CREATED_ZONE=""
for ZONE in $ZONES; do
    echo "Trying zone ${ZONE}..."
    if gcloud compute instances create "${VM}" \
        --zone=$ZONE \
        --machine-type=$MACHINE \
        $ACCEL \
        $SPOT \
        --boot-disk-size=200GB \
        --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True,task=${TASK}" \
        --labels="task=${TASK},tier=${TIER}" \
        --tags=ainm-task 2>&1; then
        CREATED_ZONE=$ZONE
        break
    else
        echo "Zone ${ZONE} unavailable, trying next..."
    fi
done

if [ -z "$CREATED_ZONE" ]; then
    echo "ERROR: Could not create VM in any zone. Try again later."
    exit 1
fi

# Open firewall for task port (shared rule, idempotent)
PORT=$((9049 + $(echo "cv ml nlp" | tr ' ' '\n' | grep -n "^${TASK}$" | cut -d: -f1)))
gcloud compute firewall-rules create "ainm-${TASK}-port" \
    --allow=tcp:${PORT} --target-tags=ainm-task 2>/dev/null || true

IP=$(gcloud compute instances describe "${VM}" --zone=$CREATED_ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "VM ${VM} created in ${CREATED_ZONE}."
echo "Zone:     ${CREATED_ZONE}"
echo "SSH:      gcloud compute ssh ${VM} --zone=${CREATED_ZONE}"
echo "Endpoint: http://${IP}:${PORT}"
