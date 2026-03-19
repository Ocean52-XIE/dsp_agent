# -*- coding: utf-8 -*-
"""Git 信息查询脚本

这是 git_info skill 的执行脚本，用于查询 Git 仓库信息。

使用方式（JSON+stdin 模式）：
    echo '{"command": "log", "limit": 5}' | python scripts/git_info.py
    echo '{"command": "status"}' | python scripts/git_info.py
    echo '{"command": "branch"}' | python scripts/git_info.py

输入格式：JSON（从 stdin 读取）
{
    "command": "log|status|branch|diff|show",  // 可选，默认 status
    "limit": 10,                                // 可选，默认 10（用于 log 命令）
    "branch": "main"                            // 可选，指定分支
}

返回格式：JSON
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from typing import Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 支持的 Git 命令及其说明
SUPPORTED_COMMANDS = {
    "log": "查看提交历史",
    "status": "查看工作区状态",
    "branch": "列出分支",
    "diff": "查看差异",
    "show": "查看提交详情",
}

# 默认参数值
DEFAULT_PARAMS = {
    "command": "status",
    "limit": 10,
    "branch": None,
}


def build_git_command(command: str, limit: int, branch: str | None) -> list[str]:
    """构建 Git 命令

    Args:
        command: Git 子命令
        limit: 返回数量限制（用于 log 命令）
        branch: 指定分支名称

    Returns:
        Git 命令参数列表
    """
    cmd = ["git"]

    if command == "log":
        cmd.extend(["log", "--oneline", "-n", str(limit)])
        if branch:
            cmd.append(branch)
    elif command == "status":
        cmd.append("status")
    elif command == "branch":
        cmd.extend(["branch", "-a"])
    elif command == "diff":
        cmd.append("diff")
        if branch:
            cmd.append(branch)
    elif command == "show":
        cmd.append("show")
        if branch:
            cmd.append(branch)
    else:
        # 默认执行 status
        cmd.append("status")

    return cmd


def execute_git_command(cmd: list[str]) -> tuple[bool, str, str]:
    """执行 Git 命令

    Args:
        cmd: Git 命令参数列表

    Returns:
        (成功标志, 标准输出, 标准错误)
    """
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "命令执行超时"
    except FileNotFoundError:
        return False, "", "git 命令不存在，请确保已安装 Git"
    except Exception as e:
        return False, "", str(e)


def parse_log_output(output: str) -> list[dict[str, str]]:
    """解析 log 命令输出为结构化数据

    Args:
        output: git log --oneline 的输出

    Returns:
        提交记录列表
    """
    items = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        # 格式: "abc1234 commit message"
        parts = line.split(" ", 1)
        if len(parts) >= 2:
            items.append({
                "hash": parts[0],
                "message": parts[1],
            })
        elif len(parts) == 1:
            items.append({
                "hash": parts[0],
                "message": "",
            })
    return items


def parse_branch_output(output: str) -> dict[str, list[str]]:
    """解析 branch 命令输出

    Args:
        output: git branch -a 的输出

    Returns:
        分支信息
    """
    current_branch = None
    local_branches = []
    remote_branches = []

    for line in output.strip().split("\n"):
        if not line.strip():
            continue

        # 当前分支以 * 开头
        if line.startswith("* "):
            current_branch = line[2:].strip()
            local_branches.append(current_branch)
        elif line.startswith("  "):
            branch_name = line.strip()
            if branch_name.startswith("remotes/"):
                remote_branches.append(branch_name)
            else:
                local_branches.append(branch_name)

    return {
        "current": current_branch,
        "local": local_branches,
        "remote": remote_branches,
    }


def format_result(
    command: str,
    success: bool,
    stdout: str,
    stderr: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """格式化执行结果

    Args:
        command: 执行的 Git 子命令
        success: 是否成功
        stdout: 标准输出
        stderr: 标准错误
        params: 原始参数

    Returns:
        格式化的结果字典
    """
    result = {
        "success": success,
        "command": command,
        "params": params,
        "timestamp": datetime.now().isoformat(),
    }

    if success:
        result["output"] = stdout.strip()

        # 根据命令类型解析结构化数据
        if command == "log":
            result["items"] = parse_log_output(stdout)
            result["count"] = len(result["items"])
        elif command == "branch":
            result["branches"] = parse_branch_output(stdout)
    else:
        result["error"] = stderr.strip() if stderr.strip() else "命令执行失败"
        result["output"] = stdout.strip() if stdout.strip() else None

    return result


def main():
    """主函数：从 stdin 读取 JSON 参数，执行 Git 命令

    脚本从 stdin 读取 JSON 格式的参数，自行处理默认值和验证。
    """
    # 1. 从 stdin 读取 JSON
    params = {}
    try:
        params_text = sys.stdin.read()
        if params_text.strip():
            params = json.loads(params_text)
        logger.info(f"接收参数: {params}")
    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"JSON 解析失败: {e}",
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    # 2. 合并默认值
    merged = {**DEFAULT_PARAMS, **params}
    logger.info(f"合并后参数: {merged}")

    # 3. 验证参数
    command = merged["command"]
    if command not in SUPPORTED_COMMANDS:
        error_result = {
            "success": False,
            "error": f"不支持的命令: {command}",
            "supported_commands": list(SUPPORTED_COMMANDS.keys()),
            "command_descriptions": SUPPORTED_COMMANDS,
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    # 4. 构建 Git 命令
    cmd = build_git_command(
        command=command,
        limit=merged["limit"],
        branch=merged["branch"],
    )

    # 5. 执行命令
    success, stdout, stderr = execute_git_command(cmd)

    # 6. 格式化并输出结果
    result = format_result(
        command=command,
        success=success,
        stdout=stdout,
        stderr=stderr,
        params=merged,
    )

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
