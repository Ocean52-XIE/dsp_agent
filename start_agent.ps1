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
# LLM CONFIG
# ============================
# Unified LLM configuration
$WORKFLOW_QA_LLM_ENABLED = "true"
$WORKFLOW_QA_LLM_BASE_URL = "https://api.deepseek.com/v1"
$WORKFLOW_QA_LLM_API_KEY = "sk-522e4a7d4b3545fea8e493039fe38a03"
$WORKFLOW_QA_LLM_MODEL = "deepseek-chat"
$WORKFLOW_QA_LLM_TIMEOUT_SECONDS = "120"
$WORKFLOW_QA_LLM_TEMPERATURE = "0.2"
$WORKFLOW_QA_LLM_MAX_TOKENS = "1600"

# ============================
# WORKFLOW CONFIG
# ============================
$WORKFLOW_DOMAIN_DIR = $DomainDir
$WORKFLOW_DEBUG_VERBOSE = "true"
$WORKFLOW_MCP_ENABLED = "true"

# ============================
# LOGGING CONFIG
# ============================
$WORKFLOW_FILE_LOG_ENABLED = "true"
$WORKFLOW_FILE_LOG_LEVEL = "INFO"
$WORKFLOW_FILE_LOG_DIR = "logs"
$WORKFLOW_FILE_LOG_MAX_BYTES = "5242880"
$WORKFLOW_FILE_LOG_BACKUP_COUNT = "3"

# ============================
# DATABASE CONFIG
# ============================
# PostgreSQL connection string
$PG_DSN = "postgresql://postgres:123456@127.0.0.1:5432/dsp_agent"

# Observability store
$OBS_PG_ENABLED = "true"
$OBS_PG_DSN = $PG_DSN
$OBS_PG_SCHEMA = "public"
$OBS_PG_CONNECT_TIMEOUT_SECONDS = "5"

# LangGraph Checkpointer
$CHECKPOINTER_BACKEND = "postgres"
$CHECKPOINTER_PG_ENABLED = "true"
$CHECKPOINTER_PG_DSN = $PG_DSN
$CHECKPOINTER_PG_SETUP = "true"
$CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = "5"

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
    $env:WORKFLOW_QA_LLM_ENABLED = $WORKFLOW_QA_LLM_ENABLED
    $env:WORKFLOW_QA_LLM_BASE_URL = $WORKFLOW_QA_LLM_BASE_URL
    $env:WORKFLOW_QA_LLM_API_KEY = $WORKFLOW_QA_LLM_API_KEY
    $env:WORKFLOW_QA_LLM_MODEL = $WORKFLOW_QA_LLM_MODEL
    $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS = $WORKFLOW_QA_LLM_TIMEOUT_SECONDS
    $env:WORKFLOW_QA_LLM_TEMPERATURE = $WORKFLOW_QA_LLM_TEMPERATURE
    $env:WORKFLOW_QA_LLM_MAX_TOKENS = $WORKFLOW_QA_LLM_MAX_TOKENS

    # Export Workflow config
    $env:WORKFLOW_DOMAIN_DIR = $WORKFLOW_DOMAIN_DIR
    $env:WORKFLOW_DEBUG_VERBOSE = $WORKFLOW_DEBUG_VERBOSE
    $env:WORKFLOW_MCP_ENABLED = $WORKFLOW_MCP_ENABLED

    # Export Logging config
    $env:WORKFLOW_FILE_LOG_ENABLED = $WORKFLOW_FILE_LOG_ENABLED
    $env:WORKFLOW_FILE_LOG_LEVEL = $WORKFLOW_FILE_LOG_LEVEL
    $env:WORKFLOW_FILE_LOG_DIR = $WORKFLOW_FILE_LOG_DIR
    $env:WORKFLOW_FILE_LOG_MAX_BYTES = $WORKFLOW_FILE_LOG_MAX_BYTES
    $env:WORKFLOW_FILE_LOG_BACKUP_COUNT = $WORKFLOW_FILE_LOG_BACKUP_COUNT

    # Export Database config
    $env:WORKFLOW_OBS_PG_ENABLED = $OBS_PG_ENABLED
    $env:WORKFLOW_OBS_PG_DSN = $OBS_PG_DSN
    $env:WORKFLOW_OBS_PG_SCHEMA = $OBS_PG_SCHEMA
    $env:WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS = $OBS_PG_CONNECT_TIMEOUT_SECONDS
    $env:WORKFLOW_CHECKPOINTER_BACKEND = $CHECKPOINTER_BACKEND
    $env:WORKFLOW_CHECKPOINTER_PG_ENABLED = $CHECKPOINTER_PG_ENABLED
    $env:WORKFLOW_CHECKPOINTER_PG_DSN = $CHECKPOINTER_PG_DSN
    $env:WORKFLOW_CHECKPOINTER_PG_SETUP = $CHECKPOINTER_PG_SETUP
    $env:WORKFLOW_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = $CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS

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
    Write-Host "WORKFLOW_QA_LLM_ENABLED        : $env:WORKFLOW_QA_LLM_ENABLED"
    Write-Host "WORKFLOW_QA_LLM_BASE_URL       : $env:WORKFLOW_QA_LLM_BASE_URL"
    Write-Host "WORKFLOW_QA_LLM_API_KEY        : $(Mask-Secret $env:WORKFLOW_QA_LLM_API_KEY)"
    Write-Host "WORKFLOW_QA_LLM_MODEL          : $env:WORKFLOW_QA_LLM_MODEL"
    Write-Host "WORKFLOW_QA_LLM_TIMEOUT_SECONDS: $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS"
    Write-Host "WORKFLOW_QA_LLM_TEMPERATURE    : $env:WORKFLOW_QA_LLM_TEMPERATURE"
    Write-Host "WORKFLOW_QA_LLM_MAX_TOKENS     : $env:WORKFLOW_QA_LLM_MAX_TOKENS"
    Write-Host ""
    Write-Host "==== Workflow Config ===="
    Write-Host "WORKFLOW_DOMAIN_DIR            : $env:WORKFLOW_DOMAIN_DIR"
    Write-Host "WORKFLOW_DEBUG_VERBOSE         : $env:WORKFLOW_DEBUG_VERBOSE"
    Write-Host "WORKFLOW_MCP_ENABLED           : $env:WORKFLOW_MCP_ENABLED"
    Write-Host ""
    Write-Host "==== Database Config ===="
    Write-Host "OBS_PG_ENABLED                 : $env:WORKFLOW_OBS_PG_ENABLED"
    Write-Host "OBS_PG_DSN                     : $(Mask-Secret $env:WORKFLOW_OBS_PG_DSN)"
    Write-Host "CHECKPOINTER_BACKEND           : $env:WORKFLOW_CHECKPOINTER_BACKEND"
    Write-Host "CHECKPOINTER_PG_ENABLED        : $env:WORKFLOW_CHECKPOINTER_PG_ENABLED"
    Write-Host "CHECKPOINTER_PG_DSN            : $(Mask-Secret $env:WORKFLOW_CHECKPOINTER_PG_DSN)"
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
