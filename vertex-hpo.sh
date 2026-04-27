#!/bin/bash
set -e
# Tear down VMs. Handles both single-task and fleet teardown.
#
# Usage:
#   scripts/gcp/teardown.sh           — tear down ALL ainm-* VMs
#   scripts/gcp/teardown.sh cv        — tear down only ainm-cv (primary)
#   scripts/gcp/teardown.sh cv --all  — tear down ainm-cv, ainm-cv-b, ainm-cv-c, etc.

TASK=$1
ALL_VARIANTS=""
if [ "$2" = "--all" ]; then ALL_VARIANTS=1; fi

if [ -z "$TASK" ]; then
    echo "Tearing down ALL ainm-* VMs..."
    FILTER="name~^ainm-"
else
    if [ -n "$ALL_VARIANTS" ]; then
        echo "Tearing down all ainm-${TASK}* VMs..."
        FILTER="name~^ainm-${TASK}"
    else
        echo "Tearing down ainm-${TASK}..."
        FILTER="name=ainm-${TASK}"
    fi
fi

# Find and delete matching VMs
VMS=$(gcloud compute instances list --filter="$FILTER" --format="csv[no-heading](name,zone)" 2>/dev/null)

if [ -z "$VMS" ]; then
    echo "No matching VMs found."
    exit 0
fi

while IFS=, read -r NAME ZONE; do
    echo "Deleting ${NAME} (${ZONE})..."
    gcloud compute instances delete "${NAME}" --zone="$ZONE" --quiet &
done <<< "$VMS"
wait

# Clean up firewall rules
for TASK_NAME in cv ml nlp; do
    gcloud compute firewall-rules delete "ainm-${TASK_NAME}-port" --quiet 2>/dev/null || true
done

echo "Done."
