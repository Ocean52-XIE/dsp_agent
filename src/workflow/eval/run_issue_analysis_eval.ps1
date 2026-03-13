<#
.SYNOPSIS
Run issue-analysis routing evaluation.

.DESCRIPTION
Wrapper script for `src/workflow/eval/run_issue_analysis_eval.py`.
#>

param(
    [string]$ConfigPath = "src/workflow/eval/config.issue_analysis.template.json",
    [string]$PythonCommand = "python"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $ScriptDir))
$SourceRoot = Join-Path $ProjectRoot "src"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}" -f $PythonCommand)
    }

    Write-Host "Running issue-analysis evaluation..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "SourceRoot : $SourceRoot"
    Write-Host "ConfigPath : $ConfigPath"
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    & $PythonCommand "src/workflow/eval/run_issue_analysis_eval.py" "--config" $ConfigPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
