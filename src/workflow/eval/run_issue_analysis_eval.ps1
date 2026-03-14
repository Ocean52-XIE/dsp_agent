<#
.SYNOPSIS
Run issue-analysis evaluation (LLM-aware).

.DESCRIPTION
Wrapper script for `src/workflow/eval/run_issue_analysis_eval.py`.
Configuration style is intentionally aligned with `run_answer_eval.ps1`.
#>

param(
    [string]$ConfigPath = "src/workflow/eval/config.issue_analysis.template.json",
    [string]$PythonCommand = "python"
)

# ===================================================
# ISSUE ANALYSIS LLM CONFIG（用于 issue_analysis 节点）
# 与 run_answer_eval.ps1 保持一致：统一在脚本顶部集中配置。
# ===================================================
# 明文配置开关（按你的要求支持）：用于在评测脚本中直接写入 Key，便于本地快速验证。
# 注意：
# 1. 当明文变量非空时，脚本会优先使用明文值；
# 2. 当明文变量为空时，会自动回退到环境变量；
# 3. 若你后续希望恢复“仅环境变量”模式，把 QA_LLM_API_KEY_PLAINTEXT 置空即可。
$QA_LLM_API_KEY_PLAINTEXT = "sk-338be4c62f214816aac1e0ad5667fcb4"

# LLM 基础配置：默认使用与 answer_eval 一致的 DeepSeek 配置，确保 issue_analysis 评测可直接走 LLM。
$QA_LLM_ENABLED = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_ENABLED)) { "true" } else { $env:WORKFLOW_QA_LLM_ENABLED }
$QA_LLM_BASE_URL = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_BASE_URL)) { "https://api.deepseek.com/v1" } else { $env:WORKFLOW_QA_LLM_BASE_URL }
$QA_LLM_MODEL = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_MODEL)) { "deepseek-chat" } else { $env:WORKFLOW_QA_LLM_MODEL }
$QA_LLM_TIMEOUT_SECONDS = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS)) { "180" } else { $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS }
$QA_LLM_TEMPERATURE = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_TEMPERATURE)) { "0.1" } else { $env:WORKFLOW_QA_LLM_TEMPERATURE }
$QA_LLM_MAX_TOKENS = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_MAX_TOKENS)) { "1800" } else { $env:WORKFLOW_QA_LLM_MAX_TOKENS }
$QA_LLM_RETRY_COUNT = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_RETRY_COUNT)) { "2" } else { $env:WORKFLOW_QA_LLM_RETRY_COUNT }
$QA_LLM_RETRY_BASE_DELAY_MS = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS)) { "500" } else { $env:WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS }
$WORKFLOW_DEBUG_VERBOSE = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_DEBUG_VERBOSE)) { "false" } else { $env:WORKFLOW_DEBUG_VERBOSE }

# API Key 选择策略：明文优先，其次环境变量，最后为空。
$QA_LLM_API_KEY = if (-not [string]::IsNullOrWhiteSpace($QA_LLM_API_KEY_PLAINTEXT)) {
    $QA_LLM_API_KEY_PLAINTEXT
} elseif (-not [string]::IsNullOrWhiteSpace($env:WORKFLOW_QA_LLM_API_KEY)) {
    $env:WORKFLOW_QA_LLM_API_KEY
} else {
    ""
}

# ===================================================
# DOMAIN CONFIG
# ===================================================
# 默认将评测域指向 ad_engine。若希望评估 logistics，可改成 "domain/logistics"，
# 或者在 issue_analysis 的 config JSON 中配置 domain_dir 覆盖此默认值。
$WORKFLOW_DOMAIN_DIR = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_DOMAIN_DIR)) { "domain/ad_engine" } else { $env:WORKFLOW_DOMAIN_DIR }
# 代码检索目录默认跟随 domain_dir 变化，确保评测环境和领域语料一致。
$CODE_RETRIEVER_DIRS = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_CODE_RETRIEVER_DIRS)) { "$WORKFLOW_DOMAIN_DIR/codes" } else { $env:WORKFLOW_CODE_RETRIEVER_DIRS }

# =====================================================
# EVAL RETRY CONFIG（评测执行重试）
# =====================================================
$EVAL_RETRY_COUNT = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_EVAL_RETRY_COUNT)) { "1" } else { $env:WORKFLOW_EVAL_RETRY_COUNT }
$EVAL_RETRY_BASE_DELAY_MS = if ([string]::IsNullOrWhiteSpace($env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS)) { "300" } else { $env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS }

function Mask-Secret([string]$raw) {
    # 仅用于日志脱敏显示，避免在控制台暴露完整密钥。
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return "<EMPTY>"
    }
    if ($raw.Length -le 8) {
        return ("*" * $raw.Length)
    }
    return ($raw.Substring(0, 4) + "****" + $raw.Substring($raw.Length - 4, 4))
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $ScriptDir))
$SourceRoot = Join-Path $ProjectRoot "src"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw ("Python command not found: {0}" -f $PythonCommand)
    }

    # 将 src 追加到 PYTHONPATH，确保从仓库根目录执行时能正确 import workflow 模块。
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    Write-Host "Running issue-analysis evaluation..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "SourceRoot : $SourceRoot"
    Write-Host "ConfigPath : $ConfigPath"

    # 与 answer_eval 脚本一致：在 wrapper 脚本里统一注入运行时环境变量。
    $env:WORKFLOW_QA_LLM_ENABLED = $QA_LLM_ENABLED
    $env:WORKFLOW_QA_LLM_BASE_URL = $QA_LLM_BASE_URL
    $env:WORKFLOW_QA_LLM_API_KEY = $QA_LLM_API_KEY
    $env:WORKFLOW_QA_LLM_MODEL = $QA_LLM_MODEL
    $env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS = $QA_LLM_TIMEOUT_SECONDS
    $env:WORKFLOW_QA_LLM_TEMPERATURE = $QA_LLM_TEMPERATURE
    $env:WORKFLOW_QA_LLM_MAX_TOKENS = $QA_LLM_MAX_TOKENS
    $env:WORKFLOW_QA_LLM_RETRY_COUNT = $QA_LLM_RETRY_COUNT
    $env:WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS = $QA_LLM_RETRY_BASE_DELAY_MS
    $env:WORKFLOW_DEBUG_VERBOSE = $WORKFLOW_DEBUG_VERBOSE
    $env:WORKFLOW_DOMAIN_DIR = $WORKFLOW_DOMAIN_DIR
    $env:WORKFLOW_CODE_RETRIEVER_DIRS = $CODE_RETRIEVER_DIRS
    $env:WORKFLOW_EVAL_RETRY_COUNT = $EVAL_RETRY_COUNT
    $env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS = $EVAL_RETRY_BASE_DELAY_MS

    # 打印关键配置用于排障，敏感字段做脱敏处理。
    Write-Host "WORKFLOW_QA_LLM_ENABLED             : $($env:WORKFLOW_QA_LLM_ENABLED)"
    Write-Host "WORKFLOW_QA_LLM_BASE_URL            : $($env:WORKFLOW_QA_LLM_BASE_URL)"
    Write-Host "WORKFLOW_QA_LLM_API_KEY             : $(Mask-Secret $env:WORKFLOW_QA_LLM_API_KEY)"
    Write-Host "WORKFLOW_QA_LLM_MODEL               : $($env:WORKFLOW_QA_LLM_MODEL)"
    Write-Host "WORKFLOW_QA_LLM_TIMEOUT_SECONDS     : $($env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS)"
    Write-Host "WORKFLOW_QA_LLM_TEMPERATURE         : $($env:WORKFLOW_QA_LLM_TEMPERATURE)"
    Write-Host "WORKFLOW_QA_LLM_MAX_TOKENS          : $($env:WORKFLOW_QA_LLM_MAX_TOKENS)"
    Write-Host "WORKFLOW_QA_LLM_RETRY_COUNT         : $($env:WORKFLOW_QA_LLM_RETRY_COUNT)"
    Write-Host "WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS : $($env:WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS)"
    Write-Host "WORKFLOW_DEBUG_VERBOSE              : $($env:WORKFLOW_DEBUG_VERBOSE)"
    Write-Host "WORKFLOW_DOMAIN_DIR                 : $($env:WORKFLOW_DOMAIN_DIR)"
    Write-Host "WORKFLOW_CODE_RETRIEVER_DIRS        : $($env:WORKFLOW_CODE_RETRIEVER_DIRS)"
    Write-Host "WORKFLOW_EVAL_RETRY_COUNT           : $($env:WORKFLOW_EVAL_RETRY_COUNT)"
    Write-Host "WORKFLOW_EVAL_RETRY_BASE_DELAY_MS   : $($env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS)"

    & $PythonCommand "src/workflow/eval/run_issue_analysis_eval.py" "--config" $ConfigPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

