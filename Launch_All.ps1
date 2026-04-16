# 🏭 TE Connectivity: One-Click Launch Utility
# Performs health check and starts full V5 suite

$ErrorActionPreference = "Stop"
$Root = Get-Location

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING TE CONNECTIVITY V5 SYSTEM" -ForegroundColor Cyan
Write-Host "===============================================`n"

# 1. Run Integrity Check
Write-Host "[1/3] Running Handoff Integrity Check..." -ForegroundColor White
python scripts/verify_handoff.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n🚨 INTEGRITY CHECK FAILED!" -ForegroundColor Red
    Write-Host "Please ensure datasets are placed in the root directory as per HANDOFF_GUIDE.md" -ForegroundColor Yellow
    exit
}

# 2. Launch Backend (Port 8000)
Write-Host "`n[2/3] Launching Production Backend..." -ForegroundColor White
if (-not (Test-Path ".\.venv")) {
    Write-Host "⚠️ Warning: .venv not found. Attempting to use system python..." -ForegroundColor Yellow
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; & .\.venv\Scripts\Activate.ps1; Write-Host '📡 BACKEND: http://localhost:8000' -ForegroundColor Green; python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000"

# 3. Launch Frontend (Port 5173)
Write-Host "[3/3] Launching Digital Twin Dashboard..." -ForegroundColor White
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; Write-Host '🕹️ DASHBOARD: http://localhost:5173' -ForegroundColor Green; npm run dev"

Write-Host "`n🎉 ALL SYSTEMS INITIALIZED." -ForegroundColor Cyan
Write-Host "Backend: http://localhost:8000" -ForegroundColor Gray
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Gray
Write-Host "===============================================`n"
