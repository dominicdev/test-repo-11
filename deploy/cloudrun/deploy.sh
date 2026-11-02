#!/usr/bin/env bash
# Deploy the agent service to Cloud Run. Run from the repo root:
#   PROJECT=my-project bash deploy/cloudrun/deploy.sh
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-agent2027}"
SA="${SA:-$SERVICE@$PROJECT.iam.gserviceaccount.com}"

gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com firestore.googleapis.com artifactregistry.googleapis.com

# Gemini API key in Secret Manager (create once; export GEMINI_API_KEY first).
if ! gcloud secrets describe gemini-api-key >/dev/null 2>&1; then
  printf '%s' "${GEMINI_API_KEY:?export GEMINI_API_KEY}" | gcloud secrets create gemini-api-key --data-file=-
fi

# The service gets its own identity: Firestore access + that one secret, nothing else.
if ! gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE"
fi
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" --role roles/datastore.user >/dev/null
gcloud secrets add-iam-policy-binding gemini-api-key --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor >/dev/null

# --no-cpu-throttling (instance-based billing) + --min-instances=1: the run keeps
# working in background threads after POST /runs has already returned.
cp deploy/cloudrun/Dockerfile ./Dockerfile
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --service-account "$SA" \
  --no-cpu-throttling \
  --min-instances 1 \
  --timeout 3600 \
  --set-env-vars "STORE=firestore,GOOGLE_CLOUD_PROJECT=$PROJECT" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --allow-unauthenticated   # workshop only: use IAM auth for anything real
rm -f ./Dockerfile
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
