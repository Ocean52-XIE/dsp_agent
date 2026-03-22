<#
.SYNOPSIS
Run minimal context-regression checks.

.DESCRIPTION
Wrapper script for `src/workflow/eval/run_context_regression.py`.
#>

param(
    [string]$OutputPath = "src/workflow/eval/results/latest_context_regression_report.json",
    [string]$PythonCommand = "python",
    [switch]$DebugVerbose
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $ScriptDir))
$SourceRoot = Join-Path $ProjectRoot "src"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}" -f $PythonCommand)
    }

    Write-Host "Running context regression..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "SourceRoot : $SourceRoot"
    Write-Host "OutputPath : $OutputPath"
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    $args = @("src/workflow/eval/run_context_regression.py", "--output", $OutputPath)
    if ($DebugVerbose.IsPresent) {
        $args += "--debug-verbose"
    }

    & $PythonCommand @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
