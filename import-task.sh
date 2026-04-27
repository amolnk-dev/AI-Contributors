#!/bin/bash
# One-time Vertex AI setup: enable API + create GCS bucket.
#
# Usage:
#   scripts/gcp/vertex-setup.sh

set -e

PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "ERROR: No GCP project set. Run: gcloud config set project ai-nm26osl-1823"
    exit 1
fi

BUCKET="gs://ainm-2026-${PROJECT}"
REGION="europe-west4"

echo "=== Vertex AI Setup ==="
echo "Project: ${PROJECT}"
echo "Region:  ${REGION}"
echo "Bucket:  ${BUCKET}"
echo ""

# Enable Vertex AI API
echo "Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com --quiet
echo "✓ Vertex AI API enabled"

# Enable Cloud Storage API (should already be enabled, but just in case)
gcloud services enable storage.googleapis.com --quiet

# Create GCS bucket (idempotent — ignore if exists)
echo "Creating GCS bucket ${BUCKET}..."
if gsutil ls "${BUCKET}" &>/dev/null; then
    echo "✓ Bucket ${BUCKET} already exists"
else
    gsutil mb -l "${REGION}" "${BUCKET}"
    echo "✓ Bucket ${BUCKET} created"
fi

# Verify access
echo ""
echo "Verifying access..."
gsutil ls "${BUCKET}" >/dev/null
echo "✓ Bucket accessible"

gcloud ai custom-jobs list --region="${REGION}" --limit=1 >/dev/null 2>&1
echo "✓ Vertex AI API accessible"

echo ""
echo "=== Setup complete ==="
echo "Bucket: ${BUCKET}"
echo "You can now use:"
echo "  scripts/gcp/vertex-train.sh <task> <tier>    — submit training job"
echo "  scripts/gcp/vertex-hpo.sh <task> <config>    — submit HPO job"
