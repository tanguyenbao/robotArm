# Sets up the shared virtual environment (<project root>\.venv) when needed, then starts the chess arm simulator.
#   run_gui.ps1            create .venv if missing, install libraries only when missing, start the GUI
#   run_gui.ps1 -Setup     also reinstall / update every library from Python\requirements.txt first
#   run_gui.ps1 -SetupOnly do the setup and stop (do not start the GUI)
param(
    [switch]$Setup,
    [switch]$SetupOnly
)
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path $PSScriptRoot -Parent
$venvDir     = Join-Path $projectRoot ".venv"
$venvPython  = Join-Path $venvDir "Scripts\python.exe"
$pythonDir   = Join-Path $projectRoot "Python"
$requirement = Join-Path $pythonDir "requirements.txt"

function Install-Libraries {
    Write-Host "Installing / updating libraries from $requirement ..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirement
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit code $LASTEXITCODE)." }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv yet, creating it in $venvDir ..."
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3 -m venv $venvDir } else { & python -m venv $venvDir }
    if (-not (Test-Path $venvPython)) { throw "Could not create the virtual environment. Is Python 3.10+ installed?" }
    $Setup = $true
}

if ($Setup) {
    Install-Libraries
} else {
    & $venvPython -c "import PySide6, numpy, chess" 2>$null
    if ($LASTEXITCODE -ne 0) { Install-Libraries }
}

if ($SetupOnly) { Write-Host "Setup done."; exit 0 }

Set-Location $pythonDir
& $venvPython main.py
