# FairScore Streamlit App Launcher (Windows PowerShell)
# Run from the project root: .\run_app.ps1

$env:PYTHONIOENCODING = "utf-8"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Starting FairScore Streamlit frontend..."
Write-Host "App will open at: http://localhost:8501"
Write-Host "Ensure the FastAPI backend is running first (run_api.ps1)."
Write-Host "Press Ctrl+C to stop."

python -m streamlit run app/streamlit_app.py --server.port 8501

