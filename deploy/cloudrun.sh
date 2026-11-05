#!/usr/bin/env bash
# Deploy the CiteLume public demo to Google Cloud Run.
#
# Prerequisites (one time):
#   1. gcloud auth login            (the Google account that owns the billing account)
#   2. A Google Cloud project with billing linked:  PROJECT=<id> below
#   3. The shared OpenRouter key stored in Secret Manager as "demo-openrouter-key":
#        printf '%s' 'sk-or-...' | gcloud secrets create demo-openrouter-key --data-file=- --project "$PROJECT"
#
# Usage: PROJECT=my-project ./deploy/cloudrun.sh   (optional: REGION, SERVICE, BUCKET)
set -euo pipefail
: "${PROJECT:?Set PROJECT to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-citelume}"
BUCKET="${BUCKET:-${PROJECT}-citelume-demo}"
cd "$(dirname "$0")/.."

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com --project "$PROJECT"

# Bucket that keeps the per-address daily counts across restarts and scale-to-zero
gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access

# The service's identity may read the secret and write the usage bucket
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding demo-openrouter-key --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null

# One instance at most: the daily counts live in one place, and costs stay bounded
gcloud run deploy "$SERVICE" --source . --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated --cpu 1 --memory 2Gi --min-instances 0 --max-instances 1 \
  --concurrency 20 --timeout 1800 --execution-environment gen2 \
  --set-env-vars CITELUME_DEMO=1,DEMO_STATE_DIR=/data,DEMO_PROXY_HOPS=1 \
  --set-secrets DEMO_OPENROUTER_API_KEY=demo-openrouter-key:latest \
  --add-volume name=usage,type=cloud-storage,bucket="$BUCKET" --add-volume-mount volume=usage,mount-path=/data

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)'
