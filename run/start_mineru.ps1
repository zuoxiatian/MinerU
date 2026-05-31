param(
    [string]$HostName = "127.0.0.1",
    [int]$GradioPort = 7860,
    [int]$ApiPort = 8000,
    [string]$CondaEnv = "mineru",
    [string]$CondaExe = "C:\Users\LZ-DSJ-01\miniconda3\Scripts\conda.exe",
    [string]$CudaPath = "C:\Users\LZ-DSJ-01\miniconda3\envs\mineru\Library"
)

$ErrorActionPreference = "Stop"

$RunDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $RunDir
$LogDir = Join-Path $RunDir "logs"
$PidDir = Join-Path $RunDir "pids"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $PidDir | Out-Null

function Start-MinerUProcess {
    param(
        [string]$Name,
        [string]$Command,
        [string]$OutLog,
        [string]$ErrLog,
        [string]$PidFile
    )

    $script = @"
`$env:CUDA_PATH = "$CudaPath"
`$env:Path = "`$env:CUDA_PATH\bin;`$env:Path"
`$env:MINERU_VLM_ENGINE = "transformers"
`$env:MINERU_MODEL_SOURCE = "local"
Set-Location "$RepoRoot"
$Command
"@

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -PassThru

    Set-Content -Path $PidFile -Value $process.Id -Encoding ASCII
    Write-Host "$Name started. PID=$($process.Id)"
    Write-Host "  stdout: $OutLog"
    Write-Host "  stderr: $ErrLog"
}

$gradioOut = Join-Path $LogDir "gradio.out.log"
$gradioErr = Join-Path $LogDir "gradio.err.log"
$apiOut = Join-Path $LogDir "api.out.log"
$apiErr = Join-Path $LogDir "api.err.log"

Start-MinerUProcess `
    -Name "MinerU Gradio" `
    -Command "& `"$CondaExe`" run -n `"$CondaEnv`" mineru-gradio --server-name $HostName --server-port $GradioPort --api-url http://${HostName}:$ApiPort" `
    -OutLog $gradioOut `
    -ErrLog $gradioErr `
    -PidFile (Join-Path $PidDir "gradio.pid")

Start-Sleep -Seconds 3

Start-MinerUProcess `
    -Name "MinerU API" `
    -Command "& `"$CondaExe`" run -n `"$CondaEnv`" mineru-api --host $HostName --port $ApiPort" `
    -OutLog $apiOut `
    -ErrLog $apiErr `
    -PidFile (Join-Path $PidDir "api.pid")

Write-Host ""
Write-Host "MinerU startup commands have been launched."
Write-Host "Gradio URL: http://${HostName}:$GradioPort/"
Write-Host "API URL:     http://${HostName}:$ApiPort/"
Write-Host ""
Write-Host "To stop both services, run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$RunDir\stop_mineru.ps1`""
