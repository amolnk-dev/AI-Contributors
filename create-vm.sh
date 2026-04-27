#!/bin/bash
# Collect best results from entire fleet using a GCP VM as relay.
# Only 3 light operations from Mac — all heavy SSH on the VM.
#
# Usage:
#   bash scripts/gcp/collect-fleet.sh

set -euo pipefail

PROJECT="ai-nm26osl-1823"
COLLECTOR="ainm-astar-e1"
COLLECTOR_ZONE="europe-west4-a"
LOCAL_RESULTS="/tmp/fleet-results.tsv"

echo "=== Fleet Collection via Collector VM ==="

# Step 1: Get VM list from Mac (Mac has compute list permissions)
echo "Discovering VMs..."
gcloud compute instances list --project="$PROJECT" \
    --filter="name~^ainm-astar AND status:RUNNING" \
    --format="csv[no-heading](name,zone)" 2>/dev/null > /tmp/vm-list.csv
TOTAL=$(wc -l < /tmp/vm-list.csv | tr -d ' ')
echo "Found $TOTAL VMs"

# Step 2: Create collector script
cat > /tmp/vm-collector.sh << 'COLLECTOR_SCRIPT'
#!/bin/bash
PROJECT="ai-nm26osl-1823"
VMLIST="/tmp/vm-list.csv"
OUTFILE="/tmp/fleet-results.tsv"
echo -e "score\tvm\tzone\tdescription\texperiments" > "$OUTFILE"

TOTAL=$(wc -l < "$VMLIST" | tr -d ' ')
echo "Collecting from $TOTAL VMs..."

COUNT=0
while IFS=, read -r VM ZONE; do
    [ -z "$VM" ] && continue
    (
        RESULT=$(gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
            --internal-ip \
            --ssh-flag="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes" \
            --command='
                BEST=$(grep "keep" /tmp/astar/results.tsv 2>/dev/null | sort -t"	" -k2 -rn | head -1)
                TOTAL=$(grep -vc "^commit" /tmp/astar/results.tsv 2>/dev/null || echo 0)
                if [ -n "$BEST" ]; then
                    SCORE=$(echo "$BEST" | cut -f2)
                    DESC=$(echo "$BEST" | cut -f6)
                    echo "${SCORE}	'"$VM"'	'"$ZONE"'	${DESC}	${TOTAL}"
                fi
            ' </dev/null 2>/dev/null) || true
        [ -n "$RESULT" ] && echo "$RESULT" >> "$OUTFILE"
    ) &
    COUNT=$((COUNT + 1))
    if [ $((COUNT % 50)) -eq 0 ]; then
        wait
        DONE=$(( $(wc -l < "$OUTFILE") - 1 ))
        echo "  $DONE results from $COUNT/$TOTAL VMs..."
    fi
done < "$VMLIST"
wait

TOTAL_RESULTS=$(( $(wc -l < "$OUTFILE") - 1 ))
echo ""
echo "=== TOP 20 ==="
sort -t"	" -k1 -rn "$OUTFILE" | head -21
echo ""
echo "Total VMs with results: $TOTAL_RESULTS"
COLLECTOR_SCRIPT

# Step 3: Upload VM list + collector script to relay VM
echo "Uploading to collector VM..."
gcloud compute scp /tmp/vm-list.csv /tmp/vm-collector.sh \
    "$COLLECTOR:/tmp/" \
    --zone="$COLLECTOR_ZONE" --project="$PROJECT" 2>/dev/null

# Step 4: Run collection on the VM
echo "Running collection on $COLLECTOR (~2 min)..."
gcloud compute ssh "$COLLECTOR" --zone="$COLLECTOR_ZONE" --project="$PROJECT" \
    --command="bash /tmp/vm-collector.sh" </dev/null 2>&1

# Step 5: Download results
echo "Downloading results..."
gcloud compute scp "$COLLECTOR:/tmp/fleet-results.tsv" "$LOCAL_RESULTS" \
    --zone="$COLLECTOR_ZONE" --project="$PROJECT" 2>/dev/null

echo ""
echo "=== RESULTS ==="
echo "File: $LOCAL_RESULTS"
echo ""
sort -t"	" -k1 -rn "$LOCAL_RESULTS" | head -11
echo ""
ABOVE=$(awk -F'\t' 'NR>1 && $1 > 90.103' "$LOCAL_RESULTS" | wc -l | tr -d ' ')
echo "VMs with improvements above baseline: $ABOVE"
