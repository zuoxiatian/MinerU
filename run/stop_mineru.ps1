$ErrorActionPreference = "SilentlyContinue"

$RunDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidDir = Join-Path $RunDir "pids"

foreach ($name in @("gradio", "api")) {
    $pidFile = Join-Path $PidDir "$name.pid"
    if (-not (Test-Path $pidFile)) {
        Write-Host "$name pid file not found."
        continue
    }

    $pidValue = Get-Content $pidFile -Raw
    $pidValue = $pidValue.Trim()
    if (-not $pidValue) {
        Write-Host "$name pid file is empty."
        continue
    }

    Stop-Process -Id ([int]$pidValue) -Force
    Remove-Item -Force $pidFile
    Write-Host "$name stopped. PID=$pidValue"
}
