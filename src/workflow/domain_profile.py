from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_ID = "ad_engine"
DEFAULT_PROFILE_DIR = "domain"


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
    laugh_like_regex: str = r"^[哈哈呵呵嘻嘻啊嗯哦嗨~\s!！?？,，。…]+$"
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
            laugh_like_regex=_as_str(payload.get("laugh_like_regex"), r"^[哈哈呵呵嘻嘻啊嗯哦嗨~\s!！?？,，。…]+$"),
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
    ui: dict[str, str]
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
            prompts={_as_str(k): _as_str(v) for k, v in _as_dict(payload.get("prompts")).items()},
            code_generation=_as_dict(payload.get("code_generation")),
            ui={_as_str(k): _as_str(v) for k, v in _as_dict(payload.get("ui")).items()},
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

    def welcome_message(self) -> str:
        message = _as_str(self.ui.get("welcome_message"))
        if message:
            return message
        return f"这里已经切换成 LangGraph 工作流入口，当前会话使用“{self.display_name}”领域配置。"


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
