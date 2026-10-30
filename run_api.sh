#!/usr/bin/env bash
# Start the FairScore FastAPI Backend Server
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${DIR}:${PYTHONPATH}"

echo "Starting FairScore FastAPI Backend on http://0.0.0.0:8000 ..."
"${DIR}/venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8000 --reload
