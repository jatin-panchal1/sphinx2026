#!/usr/bin/env bash
# Start the FairScore Streamlit Frontend Application
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${DIR}:${PYTHONPATH}"

echo "Starting FairScore Streamlit Frontend on http://localhost:8501 ..."
"${DIR}/venv/bin/streamlit" run "${DIR}/app/streamlit_app.py" --server.port 8501 --server.headless true
