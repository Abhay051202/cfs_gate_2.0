<#
.SYNOPSIS
    Complete one-command setup for GateVision (cfs-gate).
.DESCRIPTION
    Delegates to the universal hardware-aware setup module (setup_system.py).
    Auto-detects NVIDIA RTX 50-series Blackwell (sm_120), Ada/Ampere/Turing CUDA, Intel NPU, or CPU,
    sets up the virtual environment, installs the exact matching PyTorch build, configures storage,
    initializes the database, and builds the Angular frontend.
#>
[CmdletBinding()]
param(
    [switch]$SkipAI,
    [switch]$BuildFrontend
)

$ErrorActionPreference = 'Stop'
$rootDir = $PSScriptRoot
$setupScript = Join-Path $rootDir 'setup_system.py'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python 3 is not found in PATH. Please install Python 3.10+ from python.org and re-run."
}

$argsList = @()
if ($SkipAI) { $argsList += '--skip-ai' }
if ($BuildFrontend) { $argsList += '--build-frontend' }

& python $setupScript @argsList
exit $LASTEXITCODE
