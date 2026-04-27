#!/bin/bash
# Submit a Vertex AI hyperparameter tuning job.
#
# Usage:
#   scripts/gcp/vertex-hpo.sh cv config.yaml           # Run HPO
#   scripts/gcp/vertex-hpo.sh cv config.yaml --spot     # Spot pricing
#
# Config YAML format (see docs/hpo-example.yaml):
#   studySpec:
#     metrics:
#       - metricId: val_metric
#         goal: MAXIMIZE
#     parameters:
#       - parameterId: learning_rate
#         doubleValueSpec:
#           minValue: 0.0001
#           maxValue: 0.01
#       - parameterId: batch_size
#         discreteValueSpec:
#           values: [8, 16, 32, 64]
#   maxTrialCount: 20
#   parallelTrialCount: 3

set -e

cd "$(dirname "$0")/../.."

TASK=$1
CONFIG=$2
SPOT=""
TIER="heavy"

shift 2 2>/dev/null || true
for arg in "$@"; do
    case $arg in
        --spot) SPOT="true" ;;
        --tier=*) TIER="${arg#--tier=}" ;;
    esac
done

if [ -z "$TASK" ] || [ -z "$CONFIG" ]; then
    echo "Usage: scripts/gcp/vertex-hpo.sh <cv|ml|nlp> <config.yaml> [--spot] [--tier=heavy]"
    echo ""
    echo "Config YAML specifies parameters to tune and metric to optimize."
    echo "See docs/hpo-example.yaml for format."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

PROJECT=$(gcloud config get-value project 2>/dev/null)
REGION="europe-west4"
BUCKET="gs://ainm-2026-${PROJECT}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="ainm-${TASK}-hpo-${TIMESTAMP}"
GCS_STAGING="${BUCKET}/${TASK}/staging/hpo-${TIMESTAMP}"
GCS_OUTPUT="${BUCKET}/${TASK}/models/hpo-${TIMESTAMP}"
CONTAINER="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4:latest"

# GPU tier mapping
case $TIER in
    light)
        MACHINE="n1-standard-8"
        ACCEL_TYPE="NVIDIA_TESLA_T4"
        ACCEL_COUNT=1
        ;;
    medium)
        MACHINE="n1-standard-8"
        ACCEL_TYPE="NVIDIA_L4"
        ACCEL_COUNT=1
        ;;
    heavy)
        MACHINE="a2-highgpu-1g"
        ACCEL_TYPE="NVIDIA_TESLA_A100"
        ACCEL_COUNT=1
        ;;
    *)
        echo "Unknown tier: $TIER"
        exit 1
        ;;
esac

# Read HPO config values
MAX_TRIALS=$(python3 -c "import yaml,sys; d=yaml.safe_load(open('${CONFIG}')); print(d.get('maxTrialCount', 20))")
PARALLEL=$(python3 -c "import yaml,sys; d=yaml.safe_load(open('${CONFIG}')); print(d.get('parallelTrialCount', 3))")

echo "=== Vertex AI HPO Job ==="
echo "Task:       ${TASK}"
echo "Tier:       ${TIER} (${MACHINE} + ${ACCEL_TYPE})"
echo "Job:        ${JOB_NAME}"
echo "Trials:     ${MAX_TRIALS} max, ${PARALLEL} parallel"
echo "Output:     ${GCS_OUTPUT}"
if [ -n "$SPOT" ]; then echo "Pricing:    SPOT"; fi
echo ""

# Upload task code to GCS
echo "Uploading code to ${GCS_STAGING}..."
gsutil -m cp -r "./tasks/${TASK}/"* "${GCS_STAGING}/task/"
gsutil -m cp -r ./shared/* "${GCS_STAGING}/shared/" 2>/dev/null || true
gsutil cp ./requirements.txt "${GCS_STAGING}/requirements.txt"

# Build the full HPO job config
# Read study spec from user config, combine with worker pool spec
python3 << PYEOF
import yaml
import json
import sys

# Load user's HPO config
with open("${CONFIG}") as f:
    user_config = yaml.safe_load(f)

study_spec = user_config.get("studySpec", user_config)
max_trials = user_config.get("maxTrialCount", 20)
parallel_trials = user_config.get("parallelTrialCount", 3)

# Build the full HPO config
hpo_config = {
    "displayName": "${JOB_NAME}",
    "studySpec": study_spec,
    "maxTrialCount": max_trials,
    "parallelTrialCount": parallel_trials,
    "trialJobSpec": {
        "workerPoolSpecs": [{
            "machineSpec": {
                "machineType": "${MACHINE}",
                "acceleratorType": "${ACCEL_TYPE}",
                "acceleratorCount": ${ACCEL_COUNT}
            },
            "replicaCount": 1,
            "containerSpec": {
                "imageUri": "${CONTAINER}",
                "command": ["bash", "-c"],
                "args": [
                    "pip install -q google-cloud-storage cloudml-hypertune transformers scikit-learn xgboost lightgbm opencv-python-headless Pillow optuna wandb numpy pandas httpx scipy pyyaml 2>&1 | tail -3 && "
                    "gsutil -m cp -r ${GCS_STAGING}/task/* /workspace/task/ && "
                    "mkdir -p /workspace/shared && "
                    "gsutil -m cp -r ${GCS_STAGING}/shared/* /workspace/shared/ 2>/dev/null; "
                    "cd /workspace/task && "
                    "TASK_MODE=train PYTHONPATH=/workspace python3 train.py && "
                    "if [ -d /workspace/task/models ] && [ \"\$(ls -A /workspace/task/models 2>/dev/null)\" ]; then "
                    "gsutil -m cp -r /workspace/task/models/* ${GCS_OUTPUT}/\$CLOUD_ML_TRIAL_ID/; fi"
                ]
            }
        }]
    }
}

$([ -n "$SPOT" ] && echo 'hpo_config["trialJobSpec"]["scheduling"] = {"strategy": "SPOT"}')

with open("/tmp/vertex-hpo-config.yaml", "w") as f:
    yaml.dump(hpo_config, f, default_flow_style=False)

print("HPO config written to /tmp/vertex-hpo-config.yaml")
PYEOF

# Submit the HPO job
echo "Submitting Vertex AI hyperparameter tuning job..."

gcloud ai hp-tuning-jobs create \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config=/tmp/vertex-hpo-config.yaml

echo ""
echo "HPO job submitted: ${JOB_NAME}"

# Get job ID
JOB_ID=$(gcloud ai hp-tuning-jobs list \
    --region="${REGION}" \
    --filter="displayName=${JOB_NAME}" \
    --format="value(name)" \
    --sort-by=~createTime \
    --limit=1)

if [ -n "$JOB_ID" ]; then
    echo "Job ID: ${JOB_ID}"
    echo ""
    echo "Monitor:"
    echo "  gcloud ai hp-tuning-jobs describe ${JOB_ID} --region=${REGION}"
    echo "  gcloud ai hp-tuning-jobs stream-logs ${JOB_ID} --region=${REGION}"
    echo ""

    # Wait for completion
    echo "Waiting for HPO job to complete..."
    while true; do
        STATUS=$(gcloud ai hp-tuning-jobs describe "${JOB_ID}" --region="${REGION}" --format="value(state)" 2>/dev/null)
        case "$STATUS" in
            JOB_STATE_SUCCEEDED)
                echo "HPO job completed successfully!"
                break
                ;;
            JOB_STATE_FAILED|JOB_STATE_CANCELLED)
                echo "HPO job ${STATUS}. Check logs for details."
                exit 1
                ;;
            *)
                echo "Status: ${STATUS} (waiting...)"
                sleep 30
                ;;
        esac
    done

    # Show best trial
    echo ""
    echo "=== Best Trial ==="
    gcloud ai hp-tuning-jobs describe "${JOB_ID}" \
        --region="${REGION}" \
        --format="yaml(trials)"

    # Download best trial's model
    echo ""
    echo "Downloading best trial model artifacts..."
    BEST_TRIAL=$(gcloud ai hp-tuning-jobs describe "${JOB_ID}" \
        --region="${REGION}" \
        --format="value(trials[0].id)" 2>/dev/null || echo "1")

    mkdir -p "./tasks/${TASK}/models/"
    gsutil -m cp -r "${GCS_OUTPUT}/${BEST_TRIAL}/*" "./tasks/${TASK}/models/" 2>/dev/null \
        || echo "No artifacts found for best trial"
    echo "Models saved to tasks/${TASK}/models/"
else
    echo "WARNING: Could not find job ID. Check the Vertex AI console."
fi
