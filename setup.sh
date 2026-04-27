#!/bin/bash
# Submit a Vertex AI custom training job for a task.
#
# Usage:
#   scripts/gcp/vertex-train.sh cv heavy           # A100 training job
#   scripts/gcp/vertex-train.sh ml light            # T4 training job
#   scripts/gcp/vertex-train.sh cv heavy --spot     # Spot pricing (~70% off)

set -e

cd "$(dirname "$0")/../.."

TASK=$1
TIER=${2:-heavy}
SPOT=""

shift 2 2>/dev/null || true
for arg in "$@"; do
    case $arg in
        --spot) SPOT="true" ;;
    esac
done

if [ -z "$TASK" ]; then
    echo "Usage: scripts/gcp/vertex-train.sh <cv|ml|nlp> [light|medium|heavy] [--spot]"
    echo ""
    echo "Tiers:"
    echo "  light   n1-standard-8 + T4 (16GB)     ~\$0.35/hr"
    echo "  medium  n1-standard-8 + L4 (24GB)     ~\$0.70/hr"
    echo "  heavy   a2-highgpu-1g + A100 (40GB)    ~\$2.95/hr"
    echo ""
    echo "Options:"
    echo "  --spot    Use spot VMs (cheaper, may be preempted)"
    exit 1
fi

PROJECT=$(gcloud config get-value project 2>/dev/null)
REGION="europe-west4"
BUCKET="gs://ainm-2026-${PROJECT}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="ainm-${TASK}-train-${TIMESTAMP}"
GCS_STAGING="${BUCKET}/${TASK}/staging/${TIMESTAMP}"
GCS_OUTPUT="${BUCKET}/${TASK}/models/"
CONTAINER="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4:latest"

# GPU tier mapping for Vertex AI
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

# Ensure bucket exists
if ! gsutil ls "${BUCKET}" &>/dev/null; then
    echo "Bucket ${BUCKET} not found. Run scripts/gcp/vertex-setup.sh first."
    exit 1
fi

echo "=== Vertex AI Training Job ==="
echo "Task:      ${TASK}"
echo "Tier:      ${TIER} (${MACHINE} + ${ACCEL_TYPE})"
echo "Job:       ${JOB_NAME}"
echo "Output:    ${GCS_OUTPUT}"
if [ -n "$SPOT" ]; then echo "Pricing:   SPOT"; fi
echo ""

# Upload task code + shared + requirements to GCS
echo "Uploading code to ${GCS_STAGING}..."
gsutil -m cp -r "./tasks/${TASK}/"* "${GCS_STAGING}/task/"
gsutil -m cp -r ./shared/* "${GCS_STAGING}/shared/" 2>/dev/null || true
gsutil cp ./requirements.txt "${GCS_STAGING}/requirements.txt"

# Create the training entry script on GCS
cat > /tmp/vertex-entrypoint.sh << 'ENTRY'
#!/bin/bash
set -e

# Install dependencies (PyTorch + CUDA already in container)
pip install -q fastapi uvicorn pydantic transformers scikit-learn xgboost \
    lightgbm opencv-python Pillow optuna wandb numpy pandas httpx scipy \
    cloudml-hypertune 2>&1 | tail -3

# Copy code from GCS staging
gsutil -m cp -r ${GCS_STAGING}/task/* /workspace/task/
gsutil -m cp -r ${GCS_STAGING}/shared/* /workspace/shared/ 2>/dev/null || true

# Run training
cd /workspace/task
TASK_MODE=train PYTHONPATH=/workspace python3 train.py

# Upload model artifacts to GCS output
if [ -d /workspace/task/models ] && [ "$(ls -A /workspace/task/models 2>/dev/null)" ]; then
    gsutil -m cp -r /workspace/task/models/* ${GCS_OUTPUT}
    echo "Models uploaded to ${GCS_OUTPUT}"
else
    echo "WARNING: No models found in /workspace/task/models/"
fi
ENTRY

# Replace the GCS_STAGING and GCS_OUTPUT placeholders in the entrypoint
sed -i.bak "s|\${GCS_STAGING}|${GCS_STAGING}|g" /tmp/vertex-entrypoint.sh
sed -i.bak "s|\${GCS_OUTPUT}|${GCS_OUTPUT}|g" /tmp/vertex-entrypoint.sh
rm -f /tmp/vertex-entrypoint.sh.bak

gsutil cp /tmp/vertex-entrypoint.sh "${GCS_STAGING}/entrypoint.sh"

# Build the worker pool spec
WORKER_SPEC="machine-type=${MACHINE},accelerator-type=${ACCEL_TYPE},accelerator-count=${ACCEL_COUNT}"
WORKER_SPEC="${WORKER_SPEC},container-image-uri=${CONTAINER}"
WORKER_SPEC="${WORKER_SPEC},local-package-path=."

# Submit the job
echo "Submitting Vertex AI custom job..."

SPOT_FLAG=""
if [ -n "$SPOT" ]; then
    SPOT_FLAG="--scheduling-strategy=SPOT"
fi

# Create a config YAML for the job
cat > /tmp/vertex-job-config.yaml << EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
      acceleratorType: ${ACCEL_TYPE}
      acceleratorCount: ${ACCEL_COUNT}
    replicaCount: 1
    containerSpec:
      imageUri: ${CONTAINER}
      command:
        - bash
        - -c
        - |
          pip install -q google-cloud-storage cloudml-hypertune transformers scikit-learn xgboost lightgbm opencv-python-headless Pillow optuna wandb numpy pandas httpx scipy 2>&1 | tail -3
          gsutil -m cp -r ${GCS_STAGING}/task/* /workspace/task/
          mkdir -p /workspace/shared
          gsutil -m cp -r ${GCS_STAGING}/shared/* /workspace/shared/ 2>/dev/null || true
          cd /workspace/task
          TASK_MODE=train PYTHONPATH=/workspace python3 train.py
          if [ -d /workspace/task/models ] && [ "\$(ls -A /workspace/task/models 2>/dev/null)" ]; then
            gsutil -m cp -r /workspace/task/models/* ${GCS_OUTPUT}
            echo "Models uploaded to ${GCS_OUTPUT}"
          fi
EOF

if [ -n "$SPOT" ]; then
    echo "scheduling:" >> /tmp/vertex-job-config.yaml
    echo "  strategy: SPOT" >> /tmp/vertex-job-config.yaml
fi

gcloud ai custom-jobs create \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config=/tmp/vertex-job-config.yaml

echo ""
echo "Job submitted: ${JOB_NAME}"
echo ""

# Stream logs
echo "Streaming logs (Ctrl+C to stop watching, job continues running)..."
echo ""

# Get the job ID and stream logs
JOB_ID=$(gcloud ai custom-jobs list \
    --region="${REGION}" \
    --filter="displayName=${JOB_NAME}" \
    --format="value(name)" \
    --sort-by=~createTime \
    --limit=1)

if [ -n "$JOB_ID" ]; then
    echo "Job ID: ${JOB_ID}"
    echo "Monitor: gcloud ai custom-jobs describe ${JOB_ID} --region=${REGION}"
    echo "Logs:    gcloud ai custom-jobs stream-logs ${JOB_ID} --region=${REGION}"
    echo ""

    # Stream logs and wait
    gcloud ai custom-jobs stream-logs "${JOB_ID}" --region="${REGION}" || true

    # Check final status
    STATUS=$(gcloud ai custom-jobs describe "${JOB_ID}" --region="${REGION}" --format="value(state)")
    echo ""
    echo "Job status: ${STATUS}"

    if [ "$STATUS" = "JOB_STATE_SUCCEEDED" ]; then
        echo "Downloading model artifacts..."
        mkdir -p "./tasks/${TASK}/models/"
        gsutil -m cp -r "${GCS_OUTPUT}*" "./tasks/${TASK}/models/" 2>/dev/null || echo "No artifacts to download"
        echo "Models saved to tasks/${TASK}/models/"
    else
        echo "Job did not succeed. Check logs for details."
    fi
else
    echo "WARNING: Could not find job ID. Check the Vertex AI console."
fi
