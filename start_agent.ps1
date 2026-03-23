<#
.SYNOPSIS
Start Agent backend service (FastAPI + LangGraph).

.DESCRIPTION
This script lets you:
1) Configure LLM settings in one place.
2) Export settings as environment variables.
3) Start `api.main:app` with uvicorn (with `src` added to PYTHONPATH).

.EXAMPLE
.\start_agent.ps1
.EXAMPLE
.\start_agent.ps1 -Port 8080
.EXAMPLE
.\start_agent.ps1 -DryRun
#>

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$DomainDir = "domain/ad_engine",
    [switch]$DisableReload,
    [switch]$DryRun
)

# ============================
# AGENT LLM CONFIG
# ============================
$AGENT_LLM_ENABLED = "true"
$AGENT_LLM_BASE_URL = "https://api.deepseek.com/v1"
$AGENT_LLM_API_KEY = "sk-08be8a56a88949788ddf5bde3d498c43"
$AGENT_LLM_MODEL = "deepseek-chat"
$AGENT_LLM_TIMEOUT_SECONDS = "120"
$AGENT_LLM_TEMPERATURE = "0.2"
$AGENT_LLM_MAX_TOKENS = "1600"

# ============================
# AGENT RUNTIME CONFIG
# ============================
$AGENT_DOMAIN_DIR = $DomainDir
$AGENT_DEBUG_VERBOSE = "true"
$AGENT_MCP_ENABLED = "true"

# ============================
# LOGGING CONFIG
# ============================
$AGENT_FILE_LOG_ENABLED = "true"
$AGENT_FILE_LOG_LEVEL = "INFO"
$AGENT_FILE_LOG_DIR = "logs"
$AGENT_FILE_LOG_MAX_BYTES = "5242880"
$AGENT_FILE_LOG_BACKUP_COUNT = "3"

# ============================
# DATABASE CONFIG
# ============================
# PostgreSQL connection string
$AGENT_PG_DSN = "postgresql://postgres:123456@127.0.0.1:5432/dsp_agent"

# Observability store
$AGENT_OBS_PG_ENABLED = "true"
$AGENT_OBS_PG_DSN = $AGENT_PG_DSN
$AGENT_OBS_PG_SCHEMA = "public"
$AGENT_OBS_PG_CONNECT_TIMEOUT_SECONDS = "5"

# LangGraph Checkpointer
$AGENT_CHECKPOINTER_BACKEND = "postgres"
$AGENT_CHECKPOINTER_PG_ENABLED = "true"
$AGENT_CHECKPOINTER_PG_DSN = $AGENT_PG_DSN
$AGENT_CHECKPOINTER_PG_SETUP = "true"
$AGENT_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = "5"

# ============================
# AGENT CONFIG
# ============================
# Agent loop configuration
$AGENT_MAX_STEPS = "10"
$AGENT_TIMEOUT_SECONDS = "120"

# Python executable command
$PythonCommand = "python"

function Mask-Secret([string]$raw) {
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return "<EMPTY>"
    }
    if ($raw.Length -le 8) {
        return ("*" * $raw.Length)
    }
    return ($raw.Substring(0, 4) + "****" + $raw.Substring($raw.Length - 4, 4))
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceRoot = Join-Path $ProjectRoot "src"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}. Update `$PythonCommand in start_agent.ps1." -f $PythonCommand)
    }
    if (-not (Test-Path $SourceRoot)) {
        throw ("Source root not found: {0}" -f $SourceRoot)
    }

    # Set PYTHONPATH
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    # Export LLM config
    $env:AGENT_LLM_ENABLED = $AGENT_LLM_ENABLED
    $env:AGENT_LLM_BASE_URL = $AGENT_LLM_BASE_URL
    $env:AGENT_LLM_API_KEY = $AGENT_LLM_API_KEY
    $env:AGENT_LLM_MODEL = $AGENT_LLM_MODEL
    $env:AGENT_LLM_TIMEOUT_SECONDS = $AGENT_LLM_TIMEOUT_SECONDS
    $env:AGENT_LLM_TEMPERATURE = $AGENT_LLM_TEMPERATURE
    $env:AGENT_LLM_MAX_TOKENS = $AGENT_LLM_MAX_TOKENS

    # Export runtime config
    $env:AGENT_DOMAIN_DIR = $AGENT_DOMAIN_DIR
    $env:AGENT_DEBUG_VERBOSE = $AGENT_DEBUG_VERBOSE
    $env:AGENT_MCP_ENABLED = $AGENT_MCP_ENABLED

    # Export Logging config
    $env:AGENT_FILE_LOG_ENABLED = $AGENT_FILE_LOG_ENABLED
    $env:AGENT_FILE_LOG_LEVEL = $AGENT_FILE_LOG_LEVEL
    $env:AGENT_FILE_LOG_DIR = $AGENT_FILE_LOG_DIR
    $env:AGENT_FILE_LOG_MAX_BYTES = $AGENT_FILE_LOG_MAX_BYTES
    $env:AGENT_FILE_LOG_BACKUP_COUNT = $AGENT_FILE_LOG_BACKUP_COUNT

    # Export database config
    $env:AGENT_OBS_PG_ENABLED = $AGENT_OBS_PG_ENABLED
    $env:AGENT_OBS_PG_DSN = $AGENT_OBS_PG_DSN
    $env:AGENT_OBS_PG_SCHEMA = $AGENT_OBS_PG_SCHEMA
    $env:AGENT_OBS_PG_CONNECT_TIMEOUT_SECONDS = $AGENT_OBS_PG_CONNECT_TIMEOUT_SECONDS
    $env:AGENT_CHECKPOINTER_BACKEND = $AGENT_CHECKPOINTER_BACKEND
    $env:AGENT_CHECKPOINTER_PG_ENABLED = $AGENT_CHECKPOINTER_PG_ENABLED
    $env:AGENT_CHECKPOINTER_PG_DSN = $AGENT_CHECKPOINTER_PG_DSN
    $env:AGENT_CHECKPOINTER_PG_SETUP = $AGENT_CHECKPOINTER_PG_SETUP
    $env:AGENT_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = $AGENT_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS

    # Export Agent config
    $env:AGENT_MAX_STEPS = $AGENT_MAX_STEPS
    $env:AGENT_TIMEOUT_SECONDS = $AGENT_TIMEOUT_SECONDS

    # Print config info
    Write-Host "==== Agent Startup Config ===="
    Write-Host "ProjectRoot                    : $ProjectRoot"
    Write-Host "SourceRoot                     : $SourceRoot"
    Write-Host "PYTHONPATH                     : $env:PYTHONPATH"
    Write-Host "Host / Port                    : $BindHost / $Port"
    Write-Host "Reload                         : $(-not $DisableReload)"
    Write-Host ""
    Write-Host "==== LLM Config ===="
    Write-Host "AGENT_LLM_ENABLED              : $env:AGENT_LLM_ENABLED"
    Write-Host "AGENT_LLM_BASE_URL             : $env:AGENT_LLM_BASE_URL"
    Write-Host "AGENT_LLM_API_KEY              : $(Mask-Secret $env:AGENT_LLM_API_KEY)"
    Write-Host "AGENT_LLM_MODEL                : $env:AGENT_LLM_MODEL"
    Write-Host "AGENT_LLM_TIMEOUT_SECONDS      : $env:AGENT_LLM_TIMEOUT_SECONDS"
    Write-Host "AGENT_LLM_TEMPERATURE          : $env:AGENT_LLM_TEMPERATURE"
    Write-Host "AGENT_LLM_MAX_TOKENS           : $env:AGENT_LLM_MAX_TOKENS"
    Write-Host ""
    Write-Host "==== Agent Runtime Config ===="
    Write-Host "AGENT_DOMAIN_DIR              : $env:AGENT_DOMAIN_DIR"
    Write-Host "AGENT_DEBUG_VERBOSE           : $env:AGENT_DEBUG_VERBOSE"
    Write-Host "AGENT_MCP_ENABLED             : $env:AGENT_MCP_ENABLED"
    Write-Host ""
    Write-Host "==== Database Config ===="
    Write-Host "AGENT_OBS_PG_ENABLED          : $env:AGENT_OBS_PG_ENABLED"
    Write-Host "AGENT_OBS_PG_DSN              : $(Mask-Secret $env:AGENT_OBS_PG_DSN)"
    Write-Host "AGENT_CHECKPOINTER_BACKEND    : $env:AGENT_CHECKPOINTER_BACKEND"
    Write-Host "AGENT_CHECKPOINTER_PG_ENABLED : $env:AGENT_CHECKPOINTER_PG_ENABLED"
    Write-Host "AGENT_CHECKPOINTER_PG_DSN     : $(Mask-Secret $env:AGENT_CHECKPOINTER_PG_DSN)"
    Write-Host ""
    Write-Host "==== Agent Config ===="
    Write-Host "AGENT_MAX_STEPS                : $env:AGENT_MAX_STEPS"
    Write-Host "AGENT_TIMEOUT_SECONDS          : $env:AGENT_TIMEOUT_SECONDS"
    Write-Host "==============================="

    $uvicornArgs = @(
        "-m", "uvicorn", "api.main:app",
        "--host", $BindHost,
        "--port", "$Port",
        "--log-level", "info"
    )
    if (-not $DisableReload) {
        $uvicornArgs += "--reload"
    }

    Write-Host "Command: $PythonCommand $($uvicornArgs -join ' ')"
    if ($DryRun) {
        Write-Host "DryRun enabled. Config check passed. Service not started."
        exit 0
    }

    & $PythonCommand @uvicornArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
