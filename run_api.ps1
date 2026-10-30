# FairScore API Launcher (Windows PowerShell)
# Run from the project root: .\run_api.ps1

$env:PYTHONIOENCODING = "utf-8"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Starting FairScore FastAPI backend..."
Write-Host "API docs will be available at: http://localhost:8000/docs"
Write-Host "Press Ctrl+C to stop."

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

