<#
.SYNOPSIS
    Universal One-Command Starter for GateVision
.DESCRIPTION
    Runs GateVision using python run.py.
    Automatically ensures setup is complete (running setup_system.py if needed),
    and starts the server on the requested host and port.
#>
[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$Setup,
    [switch]$SkipAI
)

$ErrorActionPreference = 'Stop'
$rootDir = $PSScriptRoot
$runScript = Join-Path $rootDir 'run.py'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python 3 is not found in PATH. Please install Python 3.10+ from python.org and re-run."
}

$argsList = @('--host', $HostAddress, '--port', "$Port")
if ($Setup) { $argsList += '--setup' }
if ($SkipAI) { $argsList += '--skip-ai' }

& python $runScript @argsList
exit $LASTEXITCODE
