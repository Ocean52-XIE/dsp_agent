# -*- coding: utf-8 -*-
"""Agent Domain Profile - 领域配置

参考 v1 DomainProfile 设计，独立实现以支持 Agent 模式。

核心职责：
1. 加载领域配置 (profile.json)
2. 提供路由配置
3. 提供检索配置
4. 提供提示词配置
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _as_str(value: Any, default: str = "") -> str:
    """转换为字符串"""
    return str(value if value is not None else default).strip()


def _as_int(value: Any, default: int) -> int:
    """转换为整数"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """转换为浮点数"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_tuple(value: Any) -> tuple[str, ...]:
    """转换为字符串元组"""
    if isinstance(value, (list, tuple)):
        return tuple(_as_str(item) for item in value if _as_str(item))
    return ()


def _as_dict(value: Any) -> dict[str, Any]:
    """转换为字典"""
    return dict(value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RouterProfile:
    """路由配置

    Attributes:
        domain_terms: 领域词汇
        offtopic_terms: 领域外词汇
        small_talk_exact: 闲聊精确匹配
        small_talk_substr: 闲聊子串匹配
        threshold: 领域相关性阈值
    """
    domain_terms: tuple[str, ...] = ()
    offtopic_terms: tuple[str, ...] = ()
    small_talk_exact: tuple[str, ...] = ()
    small_talk_substr: tuple[str, ...] = ()
    threshold: float = 0.3

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterProfile":
        """从字典创建"""
        return cls(
            domain_terms=_as_tuple(payload.get("domain_terms")),
            offtopic_terms=_as_tuple(payload.get("offtopic_terms")),
            small_talk_exact=_as_tuple(payload.get("small_talk_exact")),
            small_talk_substr=_as_tuple(payload.get("small_talk_substr")),
            threshold=_as_float(payload.get("threshold"), 0.3),
        )


@dataclass(frozen=True)
class LLMProfile:
    """LLM 配置

    Attributes:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
    """
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMProfile":
        """从字典创建"""
        return cls(
            model=_as_str(payload.get("model"), "gpt-4o"),
            temperature=_as_float(payload.get("temperature"), 0.7),
            max_tokens=_as_int(payload.get("max_tokens"), 4096),
        )


@dataclass(frozen=True)
class LoopProfile:
    """Agent Loop 配置

    Attributes:
        max_steps: 最大循环步数
        timeout_seconds: 超时时间 (秒)
        system_prompt: 系统提示词
    """
    max_steps: int = 10
    timeout_seconds: int = 120
    system_prompt: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoopProfile":
        """从字典创建"""
        return cls(
            max_steps=_as_int(payload.get("max_steps"), 10),
            timeout_seconds=_as_int(payload.get("timeout_seconds"), 120),
            system_prompt=_as_str(payload.get("system_prompt")),
        )


@dataclass(frozen=True)
class PromptsProfile:
    """提示词配置

    Attributes:
        system_prompt: 系统提示词
        qa_system_path: QA 系统提示词路径
        issue_system_path: 问题分析提示词路径
    """
    system_prompt: str = ""
    qa_system_path: str = ""
    issue_system_path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptsProfile":
        """从字典创建"""
        return cls(
            system_prompt=_as_str(payload.get("system_prompt")),
            qa_system_path=_as_str(payload.get("qa_system_path")),
            issue_system_path=_as_str(payload.get("issue_system_path")),
        )


@dataclass
class DomainProfile:
    """领域配置

    参考 v1 DomainProfile 设计，独立实现以支持 Agent 模式。

    Attributes:
        profile_id: 领域 ID
        profile_dir: 领域目录
        router: 路由配置
        llm: LLM 配置
        loop: Agent Loop 配置
        prompts: 提示词配置
        skills_dir: Skills 目录
        wiki_dir: Wiki 目录
    """
    profile_id: str
    profile_dir: Path
    router: RouterProfile = field(default_factory=RouterProfile)
    llm: LLMProfile = field(default_factory=LLMProfile)
    loop: LoopProfile = field(default_factory=LoopProfile)
    prompts: PromptsProfile = field(default_factory=PromptsProfile)
    skills_dir: Path | None = None
    wiki_dir: Path | None = None

    @classmethod
    def from_directory(
        cls,
        domain_root: Path,
        profile_id: str | None = None,
    ) -> "DomainProfile":
        """从目录加载配置

        Args:
            domain_root: 领域根目录
            profile_id: 领域 ID (可选，默认从目录名获取)

        Returns:
            DomainProfile 实例
        """
        domain_root = Path(domain_root)
        if not domain_root.exists():
            raise ValueError(f"领域目录不存在: {domain_root}")

        # 确定 profile_id
        if profile_id is None:
            profile_id = domain_root.name

        # 加载 profile.json
        profile_path = domain_root / "profile.json"
        if not profile_path.exists():
            logger.warning(f"[DomainProfile] profile.json 不存在: {profile_path}")
            return cls(
                profile_id=profile_id,
                profile_dir=domain_root,
            )

        with open(profile_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # 解析各子配置
        router = RouterProfile.from_dict(_as_dict(payload.get("router", {})))
        llm = LLMProfile.from_dict(_as_dict(payload.get("llm", {})))
        loop = LoopProfile.from_dict(_as_dict(payload.get("loop", {})))
        prompts = PromptsProfile.from_dict(_as_dict(payload.get("prompts", {})))

        # 解析目录路径
        skills_dir = None
        if "skills_dir" in payload:
            skills_dir = domain_root / _as_str(payload["skills_dir"])
        elif (domain_root / "skills").exists():
            skills_dir = domain_root / "skills"

        wiki_dir = None
        if "wiki_dir" in payload:
            wiki_dir = domain_root / _as_str(payload["wiki_dir"])
        elif (domain_root / "wiki").exists():
            wiki_dir = domain_root / "wiki"

        logger.info(f"[DomainProfile] 加载成功: {profile_id}, domain_root={domain_root}")

        return cls(
            profile_id=profile_id,
            profile_dir=domain_root,
            router=router,
            llm=llm,
            loop=loop,
            prompts=prompts,
            skills_dir=skills_dir,
            wiki_dir=wiki_dir,
        )

    def resolve_wiki_dir(self, project_root: Path | None = None) -> Path | None:
        """解析 Wiki 目录

        Args:
            project_root: 项目根目录 (可选)

        Returns:
            Wiki 目录路径
        """
        if self.wiki_dir and self.wiki_dir.exists():
            return self.wiki_dir

        if project_root:
            wiki_dir = project_root / "domain" / self.profile_id / "wiki"
            if wiki_dir.exists():
                return wiki_dir

        return None

    def resolve_skills_dir(self, project_root: Path | None = None) -> Path | None:
        """解析 Skills 目录

        Args:
            project_root: 项目根目录 (可选)

        Returns:
            Skills 目录路径
        """
        if self.skills_dir and self.skills_dir.exists():
            return self.skills_dir

        if project_root:
            skills_dir = project_root / "domain" / self.profile_id / "skills"
            if skills_dir.exists():
                return skills_dir

        return None

    def load_system_prompt(self) -> str:
        """加载系统提示词

        Returns:
            系统提示词
        """
        if self.prompts.system_prompt:
            return self.prompts.system_prompt

        # 尝试从文件加载
        if self.prompts.qa_system_path:
            prompt_path = self.profile_dir / self.prompts.qa_system_path
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8").strip()

        return ""


# 全局单例
_profile_singleton: DomainProfile | None = None
_profile_singleton_lock = threading.Lock()
_profile_singleton_root: Path | None = None


def get_domain_profile(
    project_root: Path | None = None,
    profile_id: str | None = None,
    force_reload: bool = False,
) -> DomainProfile:
    """获取领域配置单例

    Args:
        project_root: 项目根目录
        profile_id: 领域 ID
        force_reload: 是否强制重新加载

    Returns:
        DomainProfile 实例
    """
    global _profile_singleton, _profile_singleton_root

    with _profile_singleton_lock:
        if _profile_singleton is not None and not force_reload:
            if project_root is None or _profile_singleton_root == project_root:
                return _profile_singleton

        # 确定 project_root
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]

        # 确定 domain_root
        if profile_id is None:
            profile_id = "ad_engine"  # 默认领域

        domain_root = project_root / "domain" / profile_id

        _profile_singleton = DomainProfile.from_directory(
            domain_root=domain_root,
            profile_id=profile_id,
        )
        _profile_singleton_root = project_root

        return _profile_singleton
