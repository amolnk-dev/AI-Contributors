#!/bin/bash
# Scale astar autoresearch swarm to a new region.
#
# Usage:
#   bash scripts/gcp/scale-astar.sh europe-west1-b 50   # 50 VMs in europe-west1-b
#   bash scripts/gcp/scale-astar.sh us-central1-a 100   # 100 VMs in us-central1-a
#   bash scripts/gcp/scale-astar.sh --status             # Show all astar VMs across regions

set -euo pipefail

PROJECT="ai-nm26osl-1823"
IMAGE="astar-swarm-image"
MACHINE="n2-highcpu-32"

STARTUP_SCRIPT='#!/bin/bash
mkdir -p /tmp/astar
cd /tmp/astar
gsutil -q cp gs://ainm-astar-data/deploy.tar.gz /tmp/astar-deploy.tar.gz
gsutil -q cp gs://ainm-astar-data/swarm-start-remote.sh /tmp/swarm-start-remote.sh
bash /tmp/swarm-start-remote.sh'

if [ "${1:-}" = "--status" ]; then
  echo "ASTAR SWARM FLEET (all regions):"
  gcloud compute instances list --project="$PROJECT" \
    --filter="name~ainm-astar" \
    --format="table(name,zone,machineType.basename(),status)" 2>/dev/null
  echo ""
  TOTAL=$(gcloud compute instances list --project="$PROJECT" --filter="name~ainm-astar AND status:RUNNING" --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
  echo "Total running: $TOTAL"
  exit 0
fi

ZONE="${1:?Usage: scale-astar.sh ZONE COUNT}"
COUNT="${2:?Usage: scale-astar.sh ZONE COUNT}"

echo "Creating $COUNT VMs in $ZONE from golden image..."

# Create in batches of 10
CREATED=0
BATCH=1
while [ $CREATED -lt $COUNT ]; do
  REMAINING=$((COUNT - CREATED))
  BATCH_SIZE=$((REMAINING > 10 ? 10 : REMAINING))

  # Generate VM names
  NAMES=""
  for i in $(seq 1 $BATCH_SIZE); do
    IDX=$((CREATED + i))
    REGION_SHORT=$(echo "$ZONE" | sed 's/-[a-z]$//' | tr -d '-')
    NAMES="$NAMES ainm-astar-${REGION_SHORT}${IDX}"
  done

  echo "  Batch $BATCH: creating $BATCH_SIZE VMs..."
  gcloud compute instances create $NAMES \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --image="$IMAGE" \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-ssd \
    --metadata=startup-script="$STARTUP_SCRIPT" \
    2>/dev/null || echo "  Some VMs may have failed (quota?)"

  CREATED=$((CREATED + BATCH_SIZE))
  BATCH=$((BATCH + 1))
done

echo ""
echo "Created $CREATED VMs in $ZONE"
echo "VMs auto-start autoresearch via startup script"
