<#
.SYNOPSIS
Run retrieval evaluation (Recall@K, MRR, citation hit rate).

.DESCRIPTION
Wrapper script for `workflow/eval/run_retrieval_eval.py`.
You can pass a custom config path or output path override.
#>

param(
    [string]$ConfigPath = "workflow/eval/config.template.json",
    [string]$PythonCommand = "python"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}" -f $PythonCommand)
    }

    Write-Host "Running retrieval evaluation..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "ConfigPath : $ConfigPath"

    & $PythonCommand "workflow/eval/run_retrieval_eval.py" "--config" $ConfigPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
