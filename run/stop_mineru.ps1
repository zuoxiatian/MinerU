param(
    [int[]]$Ports = @(7860, 8000)
)

$ErrorActionPreference = "SilentlyContinue"

$RunDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidDir = Join-Path $RunDir "pids"

function Stop-ProcessTree {
    param([int]$ProcessId)
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }
    & taskkill.exe /PID $ProcessId /T /F | Out-Null
}

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

    Stop-ProcessTree -ProcessId ([int]$pidValue)
    Remove-Item -Force $pidFile
    Write-Host "$name stopped. PID=$pidValue"
}

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $pidValue = [int]$connection.OwningProcess
        if ($pidValue -le 0) {
            continue
        }
        Stop-ProcessTree -ProcessId $pidValue
        Write-Host "listener on port $port stopped. PID=$pidValue"
    }
}
