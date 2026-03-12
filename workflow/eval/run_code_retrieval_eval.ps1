<#
.SYNOPSIS
Run code retrieval evaluation for retrieve_code v1.

.DESCRIPTION
Wrapper script for `workflow/eval/run_code_retrieval_eval.py`.
#>

param(
    [string]$ConfigPath = "workflow/eval/config.code.template.json",
    [string]$PythonCommand = "python"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}" -f $PythonCommand)
    }

    Write-Host "Running code retrieval evaluation..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "ConfigPath : $ConfigPath"

    & $PythonCommand "workflow/eval/run_code_retrieval_eval.py" "--config" $ConfigPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

