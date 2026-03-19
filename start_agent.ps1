<#
.SYNOPSIS
Start Agent backend service (FastAPI + LangGraph).

.DESCRIPTION
This script lets you:
1) Configure knowledge_answer LLM settings in one place.
2) Export settings as environment variables.
3) Start `api.main:app` with uvicorn (with `src` added to PYTHONPATH).
4) Choose between v1 (Workflow) and v2 (Agent) backend.

Examples:
- Start service (v1 default): `.\start_agent.ps1`
- Start with v2 backend: `.\start_agent.ps1 -BackendVersion v2`
- Custom port: `.\start_agent.ps1 -Port 8080`
- Validate config only: `.\start_agent.ps1 -DryRun`
#>

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$DomainDir = "domain/ad_engine",
    [ValidateSet("v1", "v2")]
    [string]$BackendVersion = "v2",  # v1 = Workflow, v2 = Agent
    [switch]$DisableReload,
    [switch]$DryRun
)

# ============================
# LLM CONFIG (EDIT THIS BLOCK)
# ============================
$QA_LLM_ENABLED = "true"
$QA_LLM_BASE_URL = "https://api.deepseek.com/v1"
$QA_LLM_API_KEY = "sk-48f2e9a153124240b88e223145a4be3b"
$QA_LLM_MODEL = "deepseek-chat"
$QA_LLM_TIMEOUT_SECONDS = "120"
$QA_LLM_TEMPERATURE = "0.2"
$QA_LLM_MAX_TOKENS = "1600"
$WORKFLOW_DOMAIN_DIR = $DomainDir
$WORKFLOW_DEBUG_VERBOSE = "true"
# Runtime file logging config
$WORKFLOW_FILE_LOG_ENABLED = "true"
$WORKFLOW_FILE_LOG_LEVEL = "INFO"
$WORKFLOW_FILE_LOG_DIR = "logs"
$WORKFLOW_FILE_LOG_FILE = "workflow.log"
$WORKFLOW_FILE_LOG_MAX_BYTES = "5242880"
$WORKFLOW_FILE_LOG_BACKUP_COUNT = "3"
# 代码检索目录收口：固定指向仓库根目录 `codes`
# 说明：可避免误检索到 workflow/eval 等工具工程目录文件。
$CODE_RETRIEVER_DIRS = ""

# ============================================
# API VERSION CONFIG (v1 vs v2)
# ============================================
# API 后端版本: v1 = Workflow (固定节点链), v2 = Agent (动态 Agent Loop)
# 可通过命令行参数 -BackendVersion 覆盖
$API_BACKEND_VERSION = $BackendVersion
# v2 失败时是否自动回退到 v1
$API_V2_FALLBACK_TO_V1 = "false"

# ============================================
# AGENT LLM CONFIG (v2 后端使用)
# ============================================
# 当 API_BACKEND_VERSION=v2 时，使用以下 LLM 配置
# 这些配置用于 agent/llm/client.py
$AGENT_LLM_ENABLED = "true"
$AGENT_LLM_BASE_URL = "https://api.deepseek.com/v1"
$AGENT_LLM_API_KEY = "sk-48f2e9a153124240b88e223145a4be3b"
$AGENT_LLM_MODEL = "deepseek-chat"
$AGENT_LLM_TIMEOUT_SECONDS = "120"
$AGENT_LLM_TEMPERATURE = "0.1"
$AGENT_LLM_MAX_TOKENS = "4096"
# Agent 循环配置
$AGENT_MAX_STEPS = "10"
$AGENT_TIMEOUT_SECONDS = "120"

# ============================================
# OBSERVABILITY CONFIG (PostgreSQL + Alerting)
# ============================================
# 是否启用可观测数据库落库。若为 false，则服务正常运行但不落库。
$OBS_PG_ENABLED = "true"
# PostgreSQL 连接串示例：
# postgresql://user:password@127.0.0.1:5432/dsp_agent
$OBS_PG_DSN = "postgresql://postgres:123456@127.0.0.1:5432/dsp_agent"
$OBS_PG_SCHEMA = "public"
$OBS_PG_CONNECT_TIMEOUT_SECONDS = "5"
# LangGraph checkpointer backend:
# - postgres: use PostgresSaver (recommended for multi-process / restart recovery)
# - memory: keep previous MemorySaver behavior
$CHECKPOINTER_BACKEND = "postgres"
$CHECKPOINTER_PG_ENABLED = "true"
$CHECKPOINTER_PG_DSN = $OBS_PG_DSN
$CHECKPOINTER_PG_SETUP = "true"
$CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = $OBS_PG_CONNECT_TIMEOUT_SECONDS

# 告警窗口与阈值配置
$OBS_ALERT_WINDOW_MINUTES = "30"
$OBS_ALERT_MIN_SAMPLES = "20"
$OBS_ALERT_SUPPRESS_MINUTES = "30"
$OBS_ALERT_EMPTY_RESPONSE_RATE_MAX = "0.05"
$OBS_ALERT_FALLBACK_RATE_MAX = "0.25"
$OBS_ALERT_INSUFFICIENT_RATE_MAX = "0.20"
$OBS_ALERT_P95_LATENCY_MS_MAX = "3000"
$OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN = "0.70"

# Effective defaults (override any malformed/comment-merged lines above).
$CODE_RETRIEVER_DIRS = ""
$OBS_PG_ENABLED = "true"
$OBS_PG_DSN = "postgresql://postgres:123456@127.0.0.1:5432/dsp_agent"
$OBS_ALERT_WINDOW_MINUTES = "30"
$CHECKPOINTER_BACKEND = "postgres"
$CHECKPOINTER_PG_ENABLED = "true"
$CHECKPOINTER_PG_DSN = $OBS_PG_DSN
$CHECKPOINTER_PG_SETUP = "true"
$CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = $OBS_PG_CONNECT_TIMEOUT_SECONDS

# Python executable command
# Example: ".\.venv\Scripts\python.exe"
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

    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    # Export env vars used by workflow/common/llm_client.py
    $env:WORKFLOW_QA_LLM_ENABLED = $QA_LLM_ENABLED
    $env:WORKFLOW_QA_LLM_BASE_URL = $QA_LLM_BASE_URL
    $env:WORKFLOW_QA_LLM_API_KEY = $QA_LLM_API_KEY
    $env:WORKFLOW_QA_LLM_MODEL = $QA_LLM_MODEL
    $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS = $QA_LLM_TIMEOUT_SECONDS
    $env:WORKFLOW_QA_LLM_TEMPERATURE = $QA_LLM_TEMPERATURE
    $env:WORKFLOW_QA_LLM_MAX_TOKENS = $QA_LLM_MAX_TOKENS
    $env:WORKFLOW_DOMAIN_DIR = $WORKFLOW_DOMAIN_DIR
    $env:WORKFLOW_DEBUG_VERBOSE = $WORKFLOW_DEBUG_VERBOSE
    $env:WORKFLOW_FILE_LOG_ENABLED = $WORKFLOW_FILE_LOG_ENABLED
    $env:WORKFLOW_FILE_LOG_LEVEL = $WORKFLOW_FILE_LOG_LEVEL
    $env:WORKFLOW_FILE_LOG_DIR = $WORKFLOW_FILE_LOG_DIR
    $env:WORKFLOW_FILE_LOG_FILE = $WORKFLOW_FILE_LOG_FILE
    $env:WORKFLOW_FILE_LOG_MAX_BYTES = $WORKFLOW_FILE_LOG_MAX_BYTES
    $env:WORKFLOW_FILE_LOG_BACKUP_COUNT = $WORKFLOW_FILE_LOG_BACKUP_COUNT
    if ([string]::IsNullOrWhiteSpace($CODE_RETRIEVER_DIRS)) {
        Remove-Item Env:WORKFLOW_CODE_RETRIEVER_DIRS -ErrorAction SilentlyContinue
    } else {
        $env:WORKFLOW_CODE_RETRIEVER_DIRS = $CODE_RETRIEVER_DIRS
    }
    $env:WORKFLOW_OBS_PG_ENABLED = $OBS_PG_ENABLED
    $env:WORKFLOW_OBS_PG_DSN = $OBS_PG_DSN
    $env:WORKFLOW_OBS_PG_SCHEMA = $OBS_PG_SCHEMA
    $env:WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS = $OBS_PG_CONNECT_TIMEOUT_SECONDS
    $env:WORKFLOW_CHECKPOINTER_BACKEND = $CHECKPOINTER_BACKEND
    $env:WORKFLOW_CHECKPOINTER_PG_ENABLED = $CHECKPOINTER_PG_ENABLED
    $env:WORKFLOW_CHECKPOINTER_PG_DSN = $CHECKPOINTER_PG_DSN
    $env:WORKFLOW_CHECKPOINTER_PG_SETUP = $CHECKPOINTER_PG_SETUP
    $env:WORKFLOW_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS = $CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS
    $env:WORKFLOW_OBS_ALERT_WINDOW_MINUTES = $OBS_ALERT_WINDOW_MINUTES
    $env:WORKFLOW_OBS_ALERT_MIN_SAMPLES = $OBS_ALERT_MIN_SAMPLES
    $env:WORKFLOW_OBS_ALERT_SUPPRESS_MINUTES = $OBS_ALERT_SUPPRESS_MINUTES
    $env:WORKFLOW_OBS_ALERT_EMPTY_RESPONSE_RATE_MAX = $OBS_ALERT_EMPTY_RESPONSE_RATE_MAX
    $env:WORKFLOW_OBS_ALERT_FALLBACK_RATE_MAX = $OBS_ALERT_FALLBACK_RATE_MAX
    $env:WORKFLOW_OBS_ALERT_INSUFFICIENT_RATE_MAX = $OBS_ALERT_INSUFFICIENT_RATE_MAX
    $env:WORKFLOW_OBS_ALERT_P95_LATENCY_MS_MAX = $OBS_ALERT_P95_LATENCY_MS_MAX
    $env:WORKFLOW_OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN = $OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN

    # ============================================
    # Export API Version Config
    # ============================================
    $env:API_BACKEND_VERSION = $API_BACKEND_VERSION
    $env:API_V2_FALLBACK_TO_V1 = $API_V2_FALLBACK_TO_V1

    # ============================================
    # Export Agent (v2) LLM Config
    # ============================================
    $env:AGENT_LLM_ENABLED = $AGENT_LLM_ENABLED
    $env:AGENT_LLM_BASE_URL = $AGENT_LLM_BASE_URL
    $env:AGENT_LLM_API_KEY = $AGENT_LLM_API_KEY
    $env:AGENT_LLM_MODEL = $AGENT_LLM_MODEL
    $env:AGENT_LLM_TIMEOUT_SECONDS = $AGENT_LLM_TIMEOUT_SECONDS
    $env:AGENT_LLM_TEMPERATURE = $AGENT_LLM_TEMPERATURE
    $env:AGENT_LLM_MAX_TOKENS = $AGENT_LLM_MAX_TOKENS
    $env:AGENT_MAX_STEPS = $AGENT_MAX_STEPS
    $env:AGENT_TIMEOUT_SECONDS = $AGENT_TIMEOUT_SECONDS

    Write-Host "==== Agent Startup Config ===="
    Write-Host "ProjectRoot                    : $ProjectRoot"
    Write-Host "SourceRoot                     : $SourceRoot"
    Write-Host "PYTHONPATH                     : $env:PYTHONPATH"
    Write-Host "Host / Port                    : $BindHost / $Port"
    Write-Host "Reload                         : $(-not $DisableReload)"
    Write-Host "WORKFLOW_QA_LLM_ENABLED        : $env:WORKFLOW_QA_LLM_ENABLED"
    Write-Host "WORKFLOW_QA_LLM_BASE_URL       : $env:WORKFLOW_QA_LLM_BASE_URL"
    Write-Host "WORKFLOW_QA_LLM_API_KEY        : $(Mask-Secret $env:WORKFLOW_QA_LLM_API_KEY)"
    Write-Host "WORKFLOW_QA_LLM_MODEL          : $env:WORKFLOW_QA_LLM_MODEL"
    Write-Host "WORKFLOW_QA_LLM_TIMEOUT_SECONDS: $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS"
    Write-Host "WORKFLOW_QA_LLM_TEMPERATURE    : $env:WORKFLOW_QA_LLM_TEMPERATURE"
    Write-Host "WORKFLOW_QA_LLM_MAX_TOKENS     : $env:WORKFLOW_QA_LLM_MAX_TOKENS"
    Write-Host "WORKFLOW_DOMAIN_DIR            : $env:WORKFLOW_DOMAIN_DIR"
    Write-Host "WORKFLOW_DEBUG_VERBOSE         : $env:WORKFLOW_DEBUG_VERBOSE"
    Write-Host "WORKFLOW_FILE_LOG_ENABLED      : $env:WORKFLOW_FILE_LOG_ENABLED"
    Write-Host "WORKFLOW_FILE_LOG_LEVEL        : $env:WORKFLOW_FILE_LOG_LEVEL"
    Write-Host "WORKFLOW_FILE_LOG_DIR          : $env:WORKFLOW_FILE_LOG_DIR"
    Write-Host "WORKFLOW_FILE_LOG_FILE         : $env:WORKFLOW_FILE_LOG_FILE"
    Write-Host "WORKFLOW_FILE_LOG_MAX_BYTES    : $env:WORKFLOW_FILE_LOG_MAX_BYTES"
    Write-Host "WORKFLOW_FILE_LOG_BACKUP_COUNT : $env:WORKFLOW_FILE_LOG_BACKUP_COUNT"
    Write-Host "WORKFLOW_CODE_RETRIEVER_DIRS   : $env:WORKFLOW_CODE_RETRIEVER_DIRS"
    Write-Host "WORKFLOW_OBS_PG_ENABLED        : $env:WORKFLOW_OBS_PG_ENABLED"
    Write-Host "WORKFLOW_OBS_PG_DSN            : $(Mask-Secret $env:WORKFLOW_OBS_PG_DSN)"
    Write-Host "WORKFLOW_OBS_PG_SCHEMA         : $env:WORKFLOW_OBS_PG_SCHEMA"
    Write-Host "WORKFLOW_CHECKPOINTER_BACKEND  : $env:WORKFLOW_CHECKPOINTER_BACKEND"
    Write-Host "WORKFLOW_CHECKPOINTER_PG_ENABLED: $env:WORKFLOW_CHECKPOINTER_PG_ENABLED"
    Write-Host "WORKFLOW_CHECKPOINTER_PG_DSN   : $(Mask-Secret $env:WORKFLOW_CHECKPOINTER_PG_DSN)"
    Write-Host "WORKFLOW_CHECKPOINTER_PG_SETUP : $env:WORKFLOW_CHECKPOINTER_PG_SETUP"
    Write-Host "WORKFLOW_OBS_ALERT_WINDOW_MINUTES      : $env:WORKFLOW_OBS_ALERT_WINDOW_MINUTES"
    Write-Host "WORKFLOW_OBS_ALERT_MIN_SAMPLES         : $env:WORKFLOW_OBS_ALERT_MIN_SAMPLES"
    Write-Host "WORKFLOW_OBS_ALERT_EMPTY_RESPONSE_MAX  : $env:WORKFLOW_OBS_ALERT_EMPTY_RESPONSE_RATE_MAX"
    Write-Host "WORKFLOW_OBS_ALERT_FALLBACK_RATE_MAX   : $env:WORKFLOW_OBS_ALERT_FALLBACK_RATE_MAX"
    Write-Host "WORKFLOW_OBS_ALERT_INSUFFICIENT_MAX    : $env:WORKFLOW_OBS_ALERT_INSUFFICIENT_RATE_MAX"
    Write-Host "WORKFLOW_OBS_ALERT_P95_LATENCY_MS_MAX  : $env:WORKFLOW_OBS_ALERT_P95_LATENCY_MS_MAX"
    Write-Host "WORKFLOW_OBS_ALERT_EXACT_LIKE_PASS_MIN : $env:WORKFLOW_OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN"
    Write-Host ""
    Write-Host "==== API Version Config ===="
    Write-Host "API_BACKEND_VERSION            : $env:API_BACKEND_VERSION"
    Write-Host "API_V2_FALLBACK_TO_V1          : $env:API_V2_FALLBACK_TO_V1"
    Write-Host ""
    Write-Host "==== Agent (v2) LLM Config ===="
    Write-Host "AGENT_LLM_ENABLED              : $env:AGENT_LLM_ENABLED"
    Write-Host "AGENT_LLM_BASE_URL             : $env:AGENT_LLM_BASE_URL"
    Write-Host "AGENT_LLM_API_KEY              : $(Mask-Secret $env:AGENT_LLM_API_KEY)"
    Write-Host "AGENT_LLM_MODEL                : $env:AGENT_LLM_MODEL"
    Write-Host "AGENT_LLM_TIMEOUT_SECONDS      : $env:AGENT_LLM_TIMEOUT_SECONDS"
    Write-Host "AGENT_LLM_TEMPERATURE          : $env:AGENT_LLM_TEMPERATURE"
    Write-Host "AGENT_LLM_MAX_TOKENS           : $env:AGENT_LLM_MAX_TOKENS"
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
