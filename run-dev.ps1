$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $root "backend"
$frontendPath = Join-Path $root "frontend"

function Wait-ForPort {
  param(
    [string]$HostName,
    [int]$Port,
    [int]$TimeoutSeconds = 60
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $async = $client.BeginConnect($HostName, $Port, $null, $null)
      if ($async.AsyncWaitHandle.WaitOne(1000, $false) -and $client.Connected) {
        $client.Close()
        return $true
      }
      $client.Close()
    } catch {
      # Keep waiting until the backend is actually listening.
    }
    Start-Sleep -Seconds 1
  }

  return $false
}

function Test-PortListening {
  param(
    [string]$HostName,
    [int]$Port
  )

  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $async = $client.BeginConnect($HostName, $Port, $null, $null)
    if ($async.AsyncWaitHandle.WaitOne(1000, $false) -and $client.Connected) {
      $client.Close()
      return $true
    }
    $client.Close()
  } catch {
    return $false
  }

  return $false
}

if (-not (Test-PortListening -HostName "127.0.0.1" -Port 8000)) {
  Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$backendPath'; `$env:LOCAL_DEV='true'; python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload"
  )
} else {
  Write-Host "Backend already listening on http://127.0.0.1:8000, reusing it."
}

if (-not (Wait-ForPort -HostName "127.0.0.1" -Port 8000 -TimeoutSeconds 90)) {
  throw "Backend did not start on 127.0.0.1:8000 within 90 seconds."
}

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$frontendPath'; npm run dev"
)

Write-Host "Started backend on http://127.0.0.1:8000 and frontend on http://localhost:5173"
