<#
.SYNOPSIS
Export wiki chunk snapshot to JSON and Markdown.
#>

param(
    [string]$WikiDir = "domain/ad_engine/wiki",
    [string]$OutputJson = "src/workflow/eval/results/wiki_chunk_export.json",
    [string]$OutputMd = "src/workflow/eval/results/wiki_chunk_export.md",
    [int]$MaxContentChars = 180,
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

    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    & $PythonCommand "src/workflow/eval/export_wiki_chunks.py" `
        "--wiki-dir" $WikiDir `
        "--output-json" $OutputJson `
        "--output-md" $OutputMd `
        "--max-content-chars" "$MaxContentChars"

    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
