#!/bin/bash
# Show status of all ainm-* VMs, including parallel experiment VMs.

echo "=== GCP Fleet Status ==="
echo ""

# List all ainm-* instances
VMS=$(gcloud compute instances list --filter="name~^ainm-" --format="csv[no-heading](name,zone,machineType.basename(),status,networkInterfaces[0].accessConfigs[0].natIP,scheduling.provisioningModel)" --sort-by="name" 2>/dev/null)

if [ -z "$VMS" ]; then
    echo "No VMs running."
    exit 0
fi

TOTAL_COST=0
COUNT=0

while IFS=, read -r NAME ZONE MACHINE STATUS IP PROVISIONING; do
    # Extract task from name (ainm-cv, ainm-cv-b → cv)
    TASK=$(echo "$NAME" | sed 's/^ainm-//' | cut -d- -f1)
    PORT=$((9049 + $(echo "cv ml nlp" | tr ' ' '\n' | grep -n "^${TASK}$" | cut -d: -f1)))

    # Determine GPU type and cost
    case $MACHINE in
        n1-standard-8)  GPU="T4 16GB";  COST="0.35" ;;
        g2-standard-8)  GPU="L4 24GB";  COST="0.70" ;;
        a2-highgpu-1g)  GPU="A100 40GB"; COST="2.95" ;;
        *)               GPU="?";        COST="0.00" ;;
    esac

    SPOT_LABEL=""
    if [ "$PROVISIONING" = "SPOT" ]; then
        COST=$(echo "$COST * 0.3" | bc 2>/dev/null || echo "$COST")
        SPOT_LABEL=" [SPOT]"
    fi

    # Health check
    HEALTH="—"
    if [ "$STATUS" = "RUNNING" ] && [ -n "$IP" ]; then
        HEALTH=$(curl -s --max-time 2 "http://${IP}:${PORT}/" 2>/dev/null || echo "NO RESPONSE")
    fi

    TOTAL_COST=$(echo "$TOTAL_COST + $COST" | bc 2>/dev/null || echo "?")
    COUNT=$((COUNT + 1))

    printf "%-14s %-18s %-10s %-12s http://%-15s:%-5s %s\n" \
        "$NAME" "$ZONE" "$GPU" "$STATUS${SPOT_LABEL}" "$IP" "$PORT" "$HEALTH"
done <<< "$VMS"

echo ""
echo "Total: ${COUNT} VMs, ~\$${TOTAL_COST}/hr"
