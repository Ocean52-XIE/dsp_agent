# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_ID = "ad_engine"
DEFAULT_PROFILE_DIR = "domain"
_PROFILE_SINGLETON_LOCK = threading.Lock()
_PROFILE_SINGLETON: "DomainProfile | None" = None
_PROFILE_SINGLETON_ROOT: Path | None = None


def _as_str(value: Any, default: str = "") -> str:
    return str(value if value is not None else default).strip()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(_as_str(item) for item in value if _as_str(item))
    return ()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_prompts(payload: dict[str, Any], *, domain_dir: Path) -> dict[str, str]:
    prompts = {_as_str(k): _as_str(v) for k, v in _as_dict(payload).items()}
    qa_system_path = _as_str(prompts.get("qa_system_path"))
    if not qa_system_path:
        return prompts

    path = Path(qa_system_path)
    if not path.is_absolute():
        path = (domain_dir / path).resolve()
    try:
        prompts["qa_system"] = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Invalid qa_system_path: {qa_system_path}") from exc
    return prompts


@dataclass(frozen=True)
class ModuleProfile:
    name: str
    hint: str = ""
    route_priority: int = 100
    keywords: tuple[str, ...] = ()
    symbol_keywords: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    wiki_hints: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModuleProfile":
        return cls(
            name=_as_str(payload.get("name")),
            hint=_as_str(payload.get("hint")),
            route_priority=_as_int(payload.get("route_priority"), 100),
            keywords=_as_tuple(payload.get("keywords")),
            symbol_keywords=_as_tuple(payload.get("symbol_keywords")),
            aliases=_as_tuple(payload.get("aliases")),
            wiki_hints=_as_tuple(payload.get("wiki_hints")),
        )


@dataclass(frozen=True)
class DomainGateProfile:
    threshold: float = 0.5
    weak_in_scope_min_score: float = 0.62
    weak_code_hint_min_score: float = 0.58
    history_memory_bonus: float = 0.18
    offtopic_penalty: float = 0.45
    short_query_penalty: float = 0.25
    short_query_max_len: int = 4
    code_hint_regex: str = r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)"
    laugh_like_regex: str = r"^[鍝堝搱鍛靛懙鍢诲樆鍟婂棷鍝﹀棬~\s!锛?锛?锛屻€傗€+$"
    domain_terms: tuple[str, ...] = ()
    small_talk_exact: tuple[str, ...] = ()
    small_talk_substr: tuple[str, ...] = ()
    offtopic_terms: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DomainGateProfile":
        return cls(
            threshold=_as_float(payload.get("threshold"), 0.5),
            weak_in_scope_min_score=_as_float(payload.get("weak_in_scope_min_score"), 0.62),
            weak_code_hint_min_score=_as_float(payload.get("weak_code_hint_min_score"), 0.58),
            history_memory_bonus=_as_float(payload.get("history_memory_bonus"), 0.18),
            offtopic_penalty=_as_float(payload.get("offtopic_penalty"), 0.45),
            short_query_penalty=_as_float(payload.get("short_query_penalty"), 0.25),
            short_query_max_len=_as_int(payload.get("short_query_max_len"), 4),
            code_hint_regex=_as_str(payload.get("code_hint_regex"), r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)"),
            laugh_like_regex=_as_str(payload.get("laugh_like_regex"), r"^[鍝堝搱鍛靛懙鍢诲樆鍟婂棷鍝﹀棬~\s!锛?锛?锛屻€傗€+$"),
            domain_terms=_as_tuple(payload.get("domain_terms")),
            small_talk_exact=_as_tuple(payload.get("small_talk_exact")),
            small_talk_substr=_as_tuple(payload.get("small_talk_substr")),
            offtopic_terms=_as_tuple(payload.get("offtopic_terms")),
        )


@dataclass(frozen=True)
class RetrievalProfile:
    presets: dict[str, dict[str, int]] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)
    max_per_source: dict[str, int] = field(default_factory=dict)
    enable_wiki: bool = True
    enable_code: bool = True
    enable_cases: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalProfile":
        presets_raw = _as_dict(payload.get("presets"))
        presets: dict[str, dict[str, int]] = {}
        for key, value in presets_raw.items():
            row = _as_dict(value)
            presets[_as_str(key)] = {
                "wiki_top_k": _as_int(row.get("wiki_top_k"), 4),
                "code_top_k": _as_int(row.get("code_top_k"), 4),
                "case_top_k": _as_int(row.get("case_top_k"), 2),
                "final_top_k": _as_int(row.get("final_top_k"), 6),
            }
        weights_raw = _as_dict(payload.get("source_weights"))
        max_raw = _as_dict(payload.get("max_per_source"))
        return cls(
            presets=presets,
            source_weights={
                "wiki": _as_float(weights_raw.get("wiki"), 1.0),
                "code": _as_float(weights_raw.get("code"), 1.0),
                "case": _as_float(weights_raw.get("case"), 0.6),
            },
            max_per_source={
                "wiki": _as_int(max_raw.get("wiki"), 4),
                "code": _as_int(max_raw.get("code"), 4),
                "case": _as_int(max_raw.get("case"), 1),
            },
            enable_wiki=bool(payload.get("enable_wiki", True)),
            enable_code=bool(payload.get("enable_code", True)),
            enable_cases=bool(payload.get("enable_cases", False)),
        )

    def preset(self, strategy: str) -> dict[str, int]:
        if strategy in self.presets:
            return dict(self.presets[strategy])
        return dict(self.presets.get("hybrid", {"wiki_top_k": 4, "code_top_k": 4, "case_top_k": 2, "final_top_k": 6}))


@dataclass(frozen=True)
class QueryRewriteProfile:
    synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    abbreviations: dict[str, str] = field(default_factory=dict)
    symbol_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    intent_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    query_templates: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryRewriteProfile":
        synonyms_raw = _as_dict(payload.get("synonyms"))
        abbr_raw = _as_dict(payload.get("abbreviations"))
        symbols_raw = _as_dict(payload.get("symbol_aliases"))
        intent_terms_raw = _as_dict(payload.get("intent_terms"))
        return cls(
            synonyms={_as_str(k): _as_tuple(v) for k, v in synonyms_raw.items()},
            abbreviations={_as_str(k): _as_str(v) for k, v in abbr_raw.items()},
            symbol_aliases={_as_str(k): _as_tuple(v) for k, v in symbols_raw.items()},
            intent_terms={_as_str(k): _as_tuple(v) for k, v in intent_terms_raw.items()},
            query_templates=tuple(_as_dict(item) for item in payload.get("query_templates", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class AnsweringProfile:
    calibration_terms: tuple[str, ...] = ()
    bid_terms: tuple[str, ...] = ()
    bid_entry_terms: tuple[str, ...] = ()
    default_entry_symbol: str = "main_entry"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnsweringProfile":
        return cls(
            calibration_terms=_as_tuple(payload.get("calibration_terms")),
            bid_terms=_as_tuple(payload.get("bid_terms")),
            bid_entry_terms=_as_tuple(payload.get("bid_entry_terms")),
            default_entry_symbol=_as_str(payload.get("default_entry_symbol"), "main_entry"),
        )


@dataclass(frozen=True)
class DomainProfile:
    profile_id: str
    display_name: str
    language: str
    schema_version: int
    sources: dict[str, Any]
    routing: dict[str, Any]
    modules: tuple[ModuleProfile, ...]
    domain_gate: DomainGateProfile
    query_rewrite: QueryRewriteProfile
    retrieval: RetrievalProfile
    answering: AnsweringProfile
    prompts: dict[str, str]
    code_generation: dict[str, Any]
    eval: dict[str, str]
    domain_dir: Path
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, domain_dir: Path) -> "DomainProfile":
        module_rows = payload.get("modules", [])
        modules = tuple(
            ModuleProfile.from_dict(item)
            for item in module_rows
            if isinstance(item, dict) and _as_str(item.get("name"))
        )
        profile_id = _as_str(payload.get("profile_id"), DEFAULT_PROFILE_ID)
        return cls(
            profile_id=profile_id,
            display_name=_as_str(payload.get("display_name"), profile_id),
            language=_as_str(payload.get("language"), "zh-CN"),
            schema_version=_as_int(payload.get("schema_version"), 1),
            sources=_as_dict(payload.get("sources")),
            routing=_as_dict(payload.get("routing")),
            modules=modules,
            domain_gate=DomainGateProfile.from_dict(_as_dict(payload.get("domain_gate"))),
            query_rewrite=QueryRewriteProfile.from_dict(_as_dict(payload.get("query_rewrite"))),
            retrieval=RetrievalProfile.from_dict(_as_dict(payload.get("retrieval"))),
            answering=AnsweringProfile.from_dict(_as_dict(payload.get("answering"))),
            prompts=_build_prompts(_as_dict(payload.get("prompts")), domain_dir=domain_dir),
            code_generation=_as_dict(payload.get("code_generation")),
            eval={_as_str(k): _as_str(v) for k, v in _as_dict(payload.get("eval")).items()},
            domain_dir=domain_dir,
            raw=dict(payload),
        )

    @property
    def default_module(self) -> str:
        configured = _as_str(self.routing.get("default_module"))
        if configured:
            return configured
        if self.modules:
            return self.modules[0].name
        return "default-module"

    def module_by_name(self, module_name: str) -> ModuleProfile | None:
        normalized = module_name.strip().lower()
        for item in self.modules:
            if item.name.strip().lower() == normalized:
                return item
        return None

    def module_hint(self, module_name: str) -> str:
        item = self.module_by_name(module_name)
        return item.hint if item else ""

    def normalize_query_text(self, text: str) -> str:
        """
        统一归一化用户查询文本。

        说明：
            节点侧对短问句、追问和意图词的判断都依赖同一份标准化文本，
            因此这里统一做小写化与空白折叠，避免不同节点因为预处理不一致
            造成同一句话在不同阶段命中不同规则。
        """
        return " ".join(str(text or "").strip().lower().split())

    def query_intent_terms(self, intent_name: str) -> tuple[str, ...]:
        """
        返回指定意图对应的配置词表。

        说明：
            节点不应直接读取 `query_rewrite.intent_terms`，
            而是通过该实例方法访问，保证意图判定入口统一。
        """
        return self.query_rewrite.intent_terms.get(str(intent_name or "").strip(), ())

    def has_query_intent(self, text: str, intent_name: str) -> bool:
        """
        判断查询文本是否命中指定意图词。

        说明：
            该方法统一封装“文本归一化 + 词表命中”逻辑，
            供 load_context、query_rewriter、knowledge_answer 等节点共用。
        """
        normalized = self.normalize_query_text(text)
        if not normalized:
            return False
        terms = self.query_intent_terms(intent_name)
        return any(term and term.lower() in normalized for term in terms)

    def infer_query_flags(self, text: str) -> dict[str, bool]:
        """
        统一推断查询文本的基础意图标记。

        说明：
            当前主要用于检索改写与回答阶段的策略判断。
            其中 `code_location` 显式复用 `is_code_location_query`，
            保证代码定位问题在不同节点拥有完全一致的判定结果。
        """
        return {
            "metric": self.has_query_intent(text, "metric"),
            "pipeline": self.has_query_intent(text, "pipeline"),
            "architecture": self.has_query_intent(text, "architecture"),
            "troubleshoot": self.has_query_intent(text, "troubleshoot"),
            "code": self.has_query_intent(text, "code"),
            "code_location": self.is_code_location_query(text),
        }

    def is_code_location_query(self, text: str) -> bool:
        """
        判断查询是否属于“代码定位”类问题。

        说明：
            这是跨节点共用的统一判定入口。优先读取 domain profile 中配置的
            `intent_terms.code_location`，只保留极少量英文兜底表达，防止配置
            缺项时基础能力完全失效。
        """
        normalized = self.normalize_query_text(text)
        if not normalized:
            return False

        if self.has_query_intent(normalized, "code_location"):
            return True

        fallback_terms = (
            "code location",
            "line",
            "where is the code",
            "where is code",
        )
        return any(token in normalized for token in fallback_terms)

    def is_pronoun_followup(self, text: str) -> bool:
        pronouns = (
            "\u5b83",  # 它
            "\u8fd9\u4e2a",  # 这个
            "\u8fd9\u4e2a\u95ee\u9898",  # 这个问题
            "\u90a3\u4e2a",  # 那个
            "\u90a3\u8fd9\u4e2a",  # 那这个
            "\u8fd9\u5757",  # 这块
            "\u8fd9\u91cc",  # 这里
            "\u4e0a\u9762\u8fd9\u4e2a",  # 上面这个
            "it",
            "that",
            "this",
        )
        normalized = str(text or "").lower()
        return any(token in normalized for token in pronouns)

    def infer_module(self, text: str) -> tuple[str, str]:
        """Infer target module name and hint from query text."""
        default_module = self.default_module
        default_hint = self.module_hint(default_module)
        if not text:
            return default_module, default_hint

        lowered = text.lower()
        modules = sorted(self.modules, key=lambda item: item.route_priority)

        # Symbol-level routing takes precedence for code-location style queries.
        for module in modules:
            if module.symbol_keywords and any(token.lower() in lowered for token in module.symbol_keywords):
                return module.name, module.hint

        best_module_name = default_module
        best_module_hint = default_hint
        best_score = 0
        best_priority = 10**9
        for module in modules:
            keyword_score = sum(1 for token in module.keywords if token and token.lower() in lowered)
            alias_score = sum(1 for token in module.aliases if token and token.lower() in lowered)
            score = keyword_score + alias_score
            if score <= 0:
                continue
            if score > best_score or (score == best_score and module.route_priority < best_priority):
                best_score = score
                best_priority = module.route_priority
                best_module_name = module.name
                best_module_hint = module.hint

        return best_module_name, best_module_hint

    def infer_related_modules(
        self,
        text: str,
        *,
        primary_module_name: str = "",
        limit: int = 2,
    ) -> list[dict[str, str]]:
        """
        推断与当前问题相关的辅助模块列表。

        说明：
            该方法用于识别“跨模块问题”，例如同时提到两率预估与出价。
            返回结果不替代主模块，只作为补充上下文提供给检索改写和最终回答。
        """
        normalized = str(text or "").strip().lower()
        primary_normalized = str(primary_module_name or "").strip().lower()
        if not normalized or limit <= 0:
            return []

        scored_modules: list[tuple[int, int, ModuleProfile]] = []
        for module in self.modules:
            module_normalized = module.name.strip().lower()
            if not module_normalized or module_normalized == primary_normalized:
                continue

            # 相关模块使用“宽松召回”策略：
            # 只要关键词、别名或符号关键词在用户问题中命中，就认为该模块可以作为辅助上下文。
            keyword_score = sum(1 for token in module.keywords if token and token.lower() in normalized)
            alias_score = sum(1 for token in module.aliases if token and token.lower() in normalized)
            symbol_score = sum(2 for token in module.symbol_keywords if token and token.lower() in normalized)
            score = keyword_score + alias_score + symbol_score
            if score <= 0:
                continue

            scored_modules.append((score, module.route_priority, module))

        scored_modules.sort(key=lambda item: (-item[0], item[1], item[2].name))
        rows: list[dict[str, str]] = []
        for _, _, module in scored_modules[:limit]:
            rows.append(
                {
                    "module_name": module.name,
                    "module_hint": module.hint,
                }
            )
        return rows

    def looks_like_code_location_query(self, text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return False

        code_location_terms = self.query_rewrite.intent_terms.get("code_location", ())
        if any(token and token.lower() in normalized for token in code_location_terms):
            return True

        fallback_terms = (
            "代码在哪儿",
            "代码在哪",
            "代码位置",
            "实现在哪",
            "入口在哪",
            "哪个文件",
            "在哪个文件",
            "路径在哪",
            "哪一行",
            "line",
            "where is the code",
            "where is code",
        )
        return any(token in normalized for token in fallback_terms)

    def module_alias_queries(self, module_name: str) -> list[str]:
        item = self.module_by_name(module_name)
        if not item:
            return []
        return [alias for alias in item.aliases if alias]

    def module_doc_hints(self) -> dict[str, tuple[str, ...]]:
        hints: dict[str, tuple[str, ...]] = {}
        for item in self.modules:
            if item.wiki_hints:
                hints[item.name] = item.wiki_hints
        return hints

    def _resolve_path(self, value: str, *, project_root: Path, must_exist: bool) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path

        candidates = [
            (self.domain_dir / path).resolve(),
            (project_root / path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if not must_exist else candidates[1]

    def resolve_wiki_dir(self, project_root: Path) -> Path:
        wiki_root = _as_str(_as_dict(self.sources.get("wiki")).get("root"), "wiki")
        return self._resolve_path(wiki_root, project_root=project_root, must_exist=False)

    def resolve_code_roots(self, project_root: Path) -> list[Path]:
        code = _as_dict(self.sources.get("code"))
        roots = _as_tuple(code.get("roots")) or ("codes",)
        resolved: list[Path] = []
        for item in roots:
            path = self._resolve_path(item, project_root=project_root, must_exist=True)
            if path.exists():
                resolved.append(path)
        return resolved

    def resolve_eval_path(self, key: str, project_root: Path) -> Path | None:
        raw_path = _as_str(self.eval.get(key))
        if not raw_path:
            return None
        return self._resolve_path(raw_path, project_root=project_root, must_exist=False)

    def build_code_context_paths(self, module_name: str) -> list[str]:
        templates = _as_tuple(self.code_generation.get("file_templates"))
        if not templates:
            templates = (
                "services/{profile_id}/{module_name}/handler.py",
                "tests/{profile_id}/{module_name}/test_handler.py",
            )
        rows: list[str] = []
        for template in templates:
            try:
                rendered = template.format(module_name=module_name, profile_id=self.profile_id)
            except Exception:
                rendered = template
            rows.append(rendered)
        return rows

    def system_prompt(self) -> str:
        return _as_str(self.prompts.get("qa_system"))


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Invalid profile JSON: {path}") from exc


def _resolve_profile_path(project_root: Path) -> Path:
    explicit = os.getenv("WORKFLOW_DOMAIN_PROFILE_PATH", "").strip()
    if explicit:
        custom = Path(explicit)
        if not custom.is_absolute():
            custom = (project_root / custom).resolve()
        return custom

    domain_dir = os.getenv("WORKFLOW_DOMAIN_DIR", "").strip()
    if domain_dir:
        domain_path = Path(domain_dir)
        if not domain_path.is_absolute():
            domain_path = (project_root / domain_path).resolve()
        if domain_path.is_file():
            return domain_path
        return domain_path / "profile.json"

    profile_id = os.getenv("WORKFLOW_DOMAIN_PROFILE", DEFAULT_PROFILE_ID).strip() or DEFAULT_PROFILE_ID
    profile_dir = os.getenv("WORKFLOW_DOMAIN_PROFILE_DIR", DEFAULT_PROFILE_DIR).strip() or DEFAULT_PROFILE_DIR
    base_dir = Path(profile_dir)
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()
    return base_dir / profile_id / "profile.json"


def load_domain_profile(*, project_root: Path) -> DomainProfile:
    profile_path = _resolve_profile_path(project_root)
    payload = _load_json_file(profile_path)
    return DomainProfile.from_dict(payload, domain_dir=profile_path.parent.resolve())


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_domain_profile(*, project_root: Path | None = None, force_reload: bool = False) -> DomainProfile:
    global _PROFILE_SINGLETON, _PROFILE_SINGLETON_ROOT

    resolved_root = (project_root or _default_project_root()).resolve()
    with _PROFILE_SINGLETON_LOCK:
        if force_reload or _PROFILE_SINGLETON is None:
            _PROFILE_SINGLETON = load_domain_profile(project_root=resolved_root)
            _PROFILE_SINGLETON_ROOT = resolved_root
            return _PROFILE_SINGLETON

        if _PROFILE_SINGLETON_ROOT != resolved_root:
            raise ValueError(
                "DomainProfile singleton is already initialized with a different project_root: "
                f"{_PROFILE_SINGLETON_ROOT}; requested={resolved_root}"
            )
        return _PROFILE_SINGLETON


def reset_domain_profile_singleton() -> None:
    global _PROFILE_SINGLETON, _PROFILE_SINGLETON_ROOT

    with _PROFILE_SINGLETON_LOCK:
        _PROFILE_SINGLETON = None
        _PROFILE_SINGLETON_ROOT = None

