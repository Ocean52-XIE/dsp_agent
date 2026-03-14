<#
.SYNOPSIS
Extract and pretty-print system_prompt / user_prompt from LLM debug_request logs.

.DESCRIPTION
Supports two input modes:
1) Parse from a log file and pick latest matched entries.
2) Parse from a single raw log line via -Line.

Examples:
- .\extract_llm_prompts.ps1
- .\extract_llm_prompts.ps1 -Path logs/workflow.log -Event workflow.llm_qa.debug_request -Last 3
- .\extract_llm_prompts.ps1 -Line '2026-... | INFO | workflow.llm_qa.debug_request | {"system_prompt":"...","user_prompt":"..."}'
#>

param(
    [string]$Path = "logs/workflow.log",
    [string]$Event = "workflow.llm_qa.debug_request",
    [string]$ResponseEvent = "",
    [int]$Last = 1,
    [string]$Line = "",
    [string]$OutFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$script:RenderLines = New-Object System.Collections.Generic.List[string]

function Add-RenderLine {
    param(
        [AllowEmptyString()]
        [string]$Line = ""
    )
    $script:RenderLines.Add($Line)
}

function Extract-JsonStringField {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JsonText,
        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    $pattern = '"' + [regex]::Escape($FieldName) + '"\s*:\s*"(?<value>(?:\\.|[^"\\])*)"'
    $m = [regex]::Match($JsonText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $m.Success) {
        return ""
    }

    $encoded = '"' + $m.Groups["value"].Value + '"'
    try {
        return [System.Text.Json.JsonSerializer]::Deserialize($encoded, [string])
    } catch {
        return [regex]::Unescape($m.Groups["value"].Value)
    }
}

function Extract-JsonNumberField {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JsonText,
        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    $pattern = '"' + [regex]::Escape($FieldName) + '"\s*:\s*(?<value>-?\d+(?:\.\d+)?)'
    $m = [regex]::Match($JsonText, $pattern)
    if (-not $m.Success) {
        return ""
    }
    return $m.Groups["value"].Value
}

function Parse-LogLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputLine
    )

    $trimmed = $InputLine.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return $null
    }

    $jsonStart = $trimmed.IndexOf("{")
    if ($jsonStart -lt 0) {
        throw "No JSON payload found in line: $trimmed"
    }

    $prefix = $trimmed.Substring(0, $jsonStart).TrimEnd()
    $jsonText = $trimmed.Substring($jsonStart)

    $m = [regex]::Match(
        $prefix,
        "^(?<timestamp>[^|]+)\s+\|\s+(?<level>[^|]+)\s+\|\s+(?<event>[^|]+)\s+\|$"
    )

    if (-not $m.Success) {
        throw "Cannot parse log prefix: $prefix"
    }

    $payload = $null
    try {
        $payload = $jsonText | ConvertFrom-Json -ErrorAction Stop
    } catch {
        # Fallback: extract only key fields needed for readable prompt debugging.
        $payload = [pscustomobject]@{
            node_name = (Extract-JsonStringField -JsonText $jsonText -FieldName "node_name")
            model = (Extract-JsonStringField -JsonText $jsonText -FieldName "model")
            base_url = (Extract-JsonStringField -JsonText $jsonText -FieldName "base_url")
            attempt = (Extract-JsonNumberField -JsonText $jsonText -FieldName "attempt")
            max_attempts = (Extract-JsonNumberField -JsonText $jsonText -FieldName "max_attempts")
            system_prompt = (Extract-JsonStringField -JsonText $jsonText -FieldName "system_prompt")
            user_prompt = (Extract-JsonStringField -JsonText $jsonText -FieldName "user_prompt")
        }
    }

    return [pscustomobject]@{
        Timestamp = $m.Groups["timestamp"].Value.Trim()
        Level = $m.Groups["level"].Value.Trim()
        Event = $m.Groups["event"].Value.Trim()
        Payload = $payload
    }
}

function Get-FieldString {
    param(
        [Parameter(Mandatory = $true)]
        $Payload,
        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )
    try {
        return [string]($Payload.$FieldName)
    } catch {
        return ""
    }
}

function Normalize-MultilineText {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Text
    )
    $out = $Text -replace "\\r\\n", "`r`n"
    $out = $out -replace "\\n", "`n"
    return $out
}

function Resolve-ResponseEventName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RequestEvent,
        [string]$ResponseEventName = ""
    )
    if (-not [string]::IsNullOrWhiteSpace($ResponseEventName)) {
        return $ResponseEventName
    }
    if ($RequestEvent -match "\.debug_request$") {
        return ($RequestEvent -replace "\.debug_request$", ".debug_response")
    }
    return "$RequestEvent.response"
}

function Find-CorrespondingResponse {
    param(
        [Parameter(Mandatory = $true)]
        $RequestEntry,
        [Parameter(Mandatory = $true)]
        [System.Collections.IList]$ParsedEntries,
        [Parameter(Mandatory = $true)]
        [string]$ResponseEventName
    )

    $reqNode = Get-FieldString -Payload $RequestEntry.Payload -FieldName "node_name"
    $reqAttempt = Get-FieldString -Payload $RequestEntry.Payload -FieldName "attempt"
    $reqModel = Get-FieldString -Payload $RequestEntry.Payload -FieldName "model"
    $reqModule = Get-FieldString -Payload $RequestEntry.Payload -FieldName "module_name"
    $reqPreview = Get-FieldString -Payload $RequestEntry.Payload -FieldName "user_query_preview"

    $requestPos = -1
    for ($pos = 0; $pos -lt $ParsedEntries.Count; $pos++) {
        if ($ParsedEntries[$pos].LineIndex -eq $RequestEntry.LineIndex) {
            $requestPos = $pos
            break
        }
    }
    if ($requestPos -lt 0) {
        return $null
    }

    for ($i = $requestPos + 1; $i -lt $ParsedEntries.Count; $i++) {
        $candidate = $ParsedEntries[$i]
        if ($candidate.Event -ne $ResponseEventName) {
            continue
        }

        $candNode = Get-FieldString -Payload $candidate.Payload -FieldName "node_name"
        $candAttempt = Get-FieldString -Payload $candidate.Payload -FieldName "attempt"
        $candModel = Get-FieldString -Payload $candidate.Payload -FieldName "model"
        $candModule = Get-FieldString -Payload $candidate.Payload -FieldName "module_name"
        $candPreview = Get-FieldString -Payload $candidate.Payload -FieldName "user_query_preview"

        if ($candNode -ne $reqNode) {
            continue
        }
        if ($candAttempt -ne $reqAttempt) {
            continue
        }
        if ($reqModel -and $candModel -and $candModel -ne $reqModel) {
            continue
        }
        if ($reqModule -and $candModule -and $candModule -ne $reqModule) {
            continue
        }
        if ($reqPreview -and $candPreview -and $candPreview -ne $reqPreview) {
            continue
        }

        return $candidate
    }

    return $null
}

function Get-TraceIdForRequest {
    param(
        [Parameter(Mandatory = $true)]
        $RequestEntry,
        [Parameter(Mandatory = $true)]
        [System.Collections.IList]$ParsedEntries
    )

    $requestPos = -1
    for ($idx = 0; $idx -lt $ParsedEntries.Count; $idx++) {
        if ($ParsedEntries[$idx].LineIndex -eq $RequestEntry.LineIndex) {
            $requestPos = $idx
            break
        }
    }
    if ($requestPos -lt 0) {
        return ""
    }

    for ($i = $requestPos; $i -ge 0; $i--) {
        $candidate = $ParsedEntries[$i]
        if ($candidate.Event -ne "workflow.invoke.start") {
            continue
        }
        $traceId = Get-FieldString -Payload $candidate.Payload -FieldName "trace_id"
        if (-not [string]::IsNullOrWhiteSpace($traceId)) {
            return $traceId
        }
    }
    return ""
}

function Resolve-OutFilePath {
    param(
        [string]$ProvidedOutFile,
        [string]$TraceId
    )
    if (-not [string]::IsNullOrWhiteSpace($ProvidedOutFile)) {
        return $ProvidedOutFile
    }

    $safeTraceId = if (-not [string]::IsNullOrWhiteSpace($TraceId)) { $TraceId } else { "unknown_trace" }
    return "test/test_${safeTraceId}.txt"
}

function Write-RenderedOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $directory = Split-Path -Path $OutputPath -Parent
    if (-not [string]::IsNullOrWhiteSpace($directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    $content = $script:RenderLines -join [Environment]::NewLine
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($OutputPath, $content, $utf8NoBom)
    Write-Output "Saved to: $OutputPath"
}

function Print-Entry {
    param(
        [Parameter(Mandatory = $true)]
        $Entry,
        $ResponseEntry = $null,
        [int]$Index = 1
    )

    $payload = $Entry.Payload
    $nodeName = Get-FieldString -Payload $payload -FieldName "node_name"
    $model = Get-FieldString -Payload $payload -FieldName "model"
    $attempt = Get-FieldString -Payload $payload -FieldName "attempt"
    $maxAttempts = Get-FieldString -Payload $payload -FieldName "max_attempts"
    $baseUrl = Get-FieldString -Payload $payload -FieldName "base_url"

    $systemPrompt = Normalize-MultilineText -Text (Get-FieldString -Payload $payload -FieldName "system_prompt")
    $userPrompt = Normalize-MultilineText -Text (Get-FieldString -Payload $payload -FieldName "user_prompt")

    Add-RenderLine "==================== Entry #$Index ===================="
    Add-RenderLine "Timestamp : $($Entry.Timestamp)"
    Add-RenderLine "Level     : $($Entry.Level)"
    Add-RenderLine "Event     : $($Entry.Event)"
    Add-RenderLine "Node      : $nodeName"
    Add-RenderLine "Model     : $model"
    Add-RenderLine "Attempt   : $attempt/$maxAttempts"
    Add-RenderLine "Base URL  : $baseUrl"
    Add-RenderLine ""
    Add-RenderLine "----- System Prompt -----"
    Add-RenderLine $systemPrompt
    Add-RenderLine ""
    Add-RenderLine "----- User Prompt -----"
    Add-RenderLine $userPrompt
    Add-RenderLine ""

    if ($null -ne $ResponseEntry) {
        $rp = $ResponseEntry.Payload
        $reason = Get-FieldString -Payload $rp -FieldName "reason"
        $latencyMs = Get-FieldString -Payload $rp -FieldName "latency_ms"
        $responseLength = Get-FieldString -Payload $rp -FieldName "response_length"
        $errorType = Get-FieldString -Payload $rp -FieldName "error_type"
        $errorMessage = Normalize-MultilineText -Text (Get-FieldString -Payload $rp -FieldName "error_message")
        $responseText = Normalize-MultilineText -Text (Get-FieldString -Payload $rp -FieldName "response_text")

        Add-RenderLine "----- Debug Response -----"
        Add-RenderLine "Timestamp : $($ResponseEntry.Timestamp)"
        Add-RenderLine "Level     : $($ResponseEntry.Level)"
        Add-RenderLine "Event     : $($ResponseEntry.Event)"
        Add-RenderLine "Reason    : $reason"
        Add-RenderLine "LatencyMs : $latencyMs"
        Add-RenderLine "RespLen   : $responseLength"
        if ($errorType) {
            Add-RenderLine "ErrorType : $errorType"
        }
        if ($errorMessage) {
            Add-RenderLine "ErrorMsg  : $errorMessage"
        }
        Add-RenderLine ""
        Add-RenderLine "Response Text:"
        Add-RenderLine $responseText
        Add-RenderLine ""
    } else {
        Add-RenderLine "----- Debug Response -----"
        Add-RenderLine "No matching debug_response found."
        Add-RenderLine ""
    }
}

function Parse-EntriesFromFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [Parameter(Mandatory = $true)]
        [string[]]$Events
    )
    $lines = Get-Content -LiteralPath $LogPath -Encoding UTF8
    $entries = New-Object System.Collections.Generic.List[object]
    for ($idx = 0; $idx -lt $lines.Count; $idx++) {
        $line = $lines[$idx]
        $isMatch = $false
        foreach ($evt in $Events) {
            if ($line -like "*| $evt | *") {
                $isMatch = $true
                break
            }
        }
        if (-not $isMatch) {
            continue
        }

        $parsed = Parse-LogLine -InputLine $line
        if ($null -eq $parsed) {
            continue
        }
        $entries.Add([pscustomobject]@{
            LineIndex = $idx
            RawLine = $line
            Timestamp = $parsed.Timestamp
            Level = $parsed.Level
            Event = $parsed.Event
            Payload = $parsed.Payload
        })
    }
    return $entries
}

$resolvedResponseEvent = Resolve-ResponseEventName -RequestEvent $Event -ResponseEventName $ResponseEvent

if (-not [string]::IsNullOrWhiteSpace($Line)) {
    $entry = Parse-LogLine -InputLine $Line
    if ($null -eq $entry) {
        throw "Input line is empty."
    }

    $responseEntry = $null
    $traceId = ""
    if ((Test-Path -LiteralPath $Path)) {
        $parsedEntries = Parse-EntriesFromFile -LogPath $Path -Events @($Event, $resolvedResponseEvent, "workflow.invoke.start")
        $lineIndexInFile = -1
        for ($idx = 0; $idx -lt $parsedEntries.Count; $idx++) {
            if ($parsedEntries[$idx].RawLine.Trim() -eq $Line.Trim()) {
                $lineIndexInFile = $parsedEntries[$idx].LineIndex
                break
            }
        }
        if ($lineIndexInFile -ge 0) {
            $requestInFile = $parsedEntries | Where-Object { $_.LineIndex -eq $lineIndexInFile } | Select-Object -First 1
            if ($null -ne $requestInFile) {
                $responseEntry = Find-CorrespondingResponse -RequestEntry $requestInFile -ParsedEntries $parsedEntries -ResponseEventName $resolvedResponseEvent
                $traceId = Get-TraceIdForRequest -RequestEntry $requestInFile -ParsedEntries $parsedEntries
            }
        }
    }

    $entryForPrint = [pscustomobject]@{
        LineIndex = -1
        Timestamp = $entry.Timestamp
        Level = $entry.Level
        Event = $entry.Event
        Payload = $entry.Payload
    }
    Print-Entry -Entry $entryForPrint -ResponseEntry $responseEntry -Index 1
    $outputPath = Resolve-OutFilePath -ProvidedOutFile $OutFile -TraceId $traceId
    Write-RenderedOutput -OutputPath $outputPath
    exit 0
}

if (-not (Test-Path -LiteralPath $Path)) {
    throw "Log file not found: $Path"
}

$parsedEntries = Parse-EntriesFromFile -LogPath $Path -Events @($Event, $resolvedResponseEvent, "workflow.invoke.start")
$requestEntries = @($parsedEntries | Where-Object { $_.Event -eq $Event })
if (-not $requestEntries -or $requestEntries.Count -eq 0) {
    throw "No matched entries found. event=$Event path=$Path"
}

$take = [Math]::Max(1, $Last)
$selected = @($requestEntries | Select-Object -Last $take)
$defaultTraceId = ""
if ($selected.Count -gt 0) {
    $defaultTraceId = Get-TraceIdForRequest -RequestEntry $selected[0] -ParsedEntries $parsedEntries
}

$i = 1
foreach ($entry in $selected) {
    $responseEntry = Find-CorrespondingResponse -RequestEntry $entry -ParsedEntries $parsedEntries -ResponseEventName $resolvedResponseEvent
    Print-Entry -Entry $entry -ResponseEntry $responseEntry -Index $i
    $i++
}

$outputPath = Resolve-OutFilePath -ProvidedOutFile $OutFile -TraceId $defaultTraceId
Write-RenderedOutput -OutputPath $outputPath
