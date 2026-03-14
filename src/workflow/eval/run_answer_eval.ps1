<#
.SYNOPSIS
Run answer-quality evaluation for workflow QA responses.

.DESCRIPTION
Wrapper script for `src/workflow/eval/run_answer_eval.py`.
You can pass a custom config path and Python command.
#>

param(
    [string]$ConfigPath = "src/workflow/eval/config.answer.template.json",
    [string]$PythonCommand = "python"
)

# ===================================================
# QA LLM CONFIG（用于 knowledge_answer 节点）
# ===================================================
$QA_LLM_ENABLED = "true"
$QA_LLM_BASE_URL = "https://api.deepseek.com/v1"
$QA_LLM_API_KEY = "sk-a50be0a1740b49efb7cdeaf6b3c6b954"
$QA_LLM_MODEL = "deepseek-chat"
$QA_LLM_TIMEOUT_SECONDS = "180"
$QA_LLM_TEMPERATURE = "0.1"
# 在出现 “has_reasoning=1 但 content 为空” 时，适度提高输出预算，给模型留出最终答案空间。
$QA_LLM_MAX_TOKENS = "1800"
# QA LLM 重试配置：总尝试次数 = 1 + RETRY_COUNT。
$QA_LLM_RETRY_COUNT = "2"
$QA_LLM_RETRY_BASE_DELAY_MS = "500"
$WORKFLOW_DEBUG_VERBOSE = "false"
$WORKFLOW_DOMAIN_DIR = "domain/ad_engine"

# 代码检索目录收口：固定指向仓库根目录 `codes`
# 说明：避免本机环境残留的 WORKFLOW_CODE_RETRIEVER_DIRS 干扰评测。
$CODE_RETRIEVER_DIRS = "$WORKFLOW_DOMAIN_DIR/codes"

# =====================================================
# LLM JUDGE CONFIG（用于 answer eval 的评审器）
# =====================================================
$EVAL_JUDGE_LLM_ENABLED = "true"
$EVAL_JUDGE_LLM_BASE_URL = "https://api.deepseek.com/v1"
$EVAL_JUDGE_LLM_API_KEY = "sk-a50be0a1740b49efb7cdeaf6b3c6b954"
$EVAL_JUDGE_LLM_MODEL = "deepseek-chat"
$EVAL_JUDGE_LLM_TIMEOUT_SECONDS = "120"
$EVAL_JUDGE_LLM_TEMPERATURE = "0.0"
$EVAL_JUDGE_LLM_MAX_TOKENS = "1600"

# =====================================================
# EVAL RETRY CONFIG（评测执行重试）
# =====================================================
# 重试次数：1 表示“失败后额外重试一次”（总共最多2次）
$EVAL_RETRY_COUNT = "1"
# 指数退避基础时长（毫秒）
$EVAL_RETRY_BASE_DELAY_MS = "300"

function Mask-Secret([string]$raw) {
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

    Write-Host "Running answer evaluation..."
    Write-Host "ProjectRoot: $ProjectRoot"
    Write-Host "SourceRoot : $SourceRoot"
    Write-Host "ConfigPath : $ConfigPath"
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SourceRoot
    } else {
        $env:PYTHONPATH = "$SourceRoot;$($env:PYTHONPATH)"
    }

    if ([string]::IsNullOrWhiteSpace($QA_LLM_MAX_TOKENS)) {
        $QA_LLM_MAX_TOKENS = "1800"
    }
    if ([string]::IsNullOrWhiteSpace($QA_LLM_RETRY_COUNT)) {
        $QA_LLM_RETRY_COUNT = "2"
    }
    if ([string]::IsNullOrWhiteSpace($CODE_RETRIEVER_DIRS)) {
        $CODE_RETRIEVER_DIRS = "$WORKFLOW_DOMAIN_DIR/codes"
    }

    # Export env vars used by workflow/nodes/knowledge_answer/llm_qa.py
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

    # Export env vars used by src/workflow/eval/run_answer_eval.py
    $env:WORKFLOW_EVAL_JUDGE_LLM_ENABLED = $EVAL_JUDGE_LLM_ENABLED
    $env:WORKFLOW_EVAL_JUDGE_LLM_BASE_URL = $EVAL_JUDGE_LLM_BASE_URL
    $env:WORKFLOW_EVAL_JUDGE_LLM_API_KEY = $EVAL_JUDGE_LLM_API_KEY
    $env:WORKFLOW_EVAL_JUDGE_LLM_MODEL = $EVAL_JUDGE_LLM_MODEL
    $env:WORKFLOW_EVAL_JUDGE_LLM_TIMEOUT_SECONDS = $EVAL_JUDGE_LLM_TIMEOUT_SECONDS
    $env:WORKFLOW_EVAL_JUDGE_LLM_TEMPERATURE = $EVAL_JUDGE_LLM_TEMPERATURE
    $env:WORKFLOW_EVAL_JUDGE_LLM_MAX_TOKENS = $EVAL_JUDGE_LLM_MAX_TOKENS
    $env:WORKFLOW_EVAL_RETRY_COUNT = $EVAL_RETRY_COUNT
    $env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS = $EVAL_RETRY_BASE_DELAY_MS

    Write-Host "WORKFLOW_QA_LLM_ENABLED        : $($env:WORKFLOW_QA_LLM_ENABLED)"
    Write-Host "WORKFLOW_QA_LLM_BASE_URL       : $($env:WORKFLOW_QA_LLM_BASE_URL)"
    Write-Host "WORKFLOW_QA_LLM_API_KEY        : $(Mask-Secret $env:WORKFLOW_QA_LLM_API_KEY)"
    Write-Host "WORKFLOW_QA_LLM_MODEL          : $($env:WORKFLOW_QA_LLM_MODEL)"
    Write-Host "WORKFLOW_QA_LLM_TIMEOUT_SECONDS: $($env:WORKFLOW_QA_LLM_TIMEOUT_SECONDS)"
    Write-Host "WORKFLOW_QA_LLM_TEMPERATURE    : $($env:WORKFLOW_QA_LLM_TEMPERATURE)"
    Write-Host "WORKFLOW_QA_LLM_MAX_TOKENS     : $($env:WORKFLOW_QA_LLM_MAX_TOKENS)"
    Write-Host "WORKFLOW_QA_LLM_RETRY_COUNT    : $($env:WORKFLOW_QA_LLM_RETRY_COUNT)"
    Write-Host "WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS: $($env:WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS)"
    Write-Host "WORKFLOW_DEBUG_VERBOSE         : $($env:WORKFLOW_DEBUG_VERBOSE)"
    Write-Host "WORKFLOW_DOMAIN_DIR            : $($env:WORKFLOW_DOMAIN_DIR)"
    Write-Host "WORKFLOW_CODE_RETRIEVER_DIRS   : $($env:WORKFLOW_CODE_RETRIEVER_DIRS)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_ENABLED        : $($env:WORKFLOW_EVAL_JUDGE_LLM_ENABLED)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_BASE_URL       : $($env:WORKFLOW_EVAL_JUDGE_LLM_BASE_URL)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_API_KEY        : $(Mask-Secret $env:WORKFLOW_EVAL_JUDGE_LLM_API_KEY)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_MODEL          : $($env:WORKFLOW_EVAL_JUDGE_LLM_MODEL)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_TIMEOUT_SECONDS: $($env:WORKFLOW_EVAL_JUDGE_LLM_TIMEOUT_SECONDS)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_TEMPERATURE    : $($env:WORKFLOW_EVAL_JUDGE_LLM_TEMPERATURE)"
    Write-Host "WORKFLOW_EVAL_JUDGE_LLM_MAX_TOKENS     : $($env:WORKFLOW_EVAL_JUDGE_LLM_MAX_TOKENS)"
    Write-Host "WORKFLOW_EVAL_RETRY_COUNT              : $($env:WORKFLOW_EVAL_RETRY_COUNT)"
    Write-Host "WORKFLOW_EVAL_RETRY_BASE_DELAY_MS      : $($env:WORKFLOW_EVAL_RETRY_BASE_DELAY_MS)"

    & $PythonCommand "src/workflow/eval/run_answer_eval.py" "--config" $ConfigPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
