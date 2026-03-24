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
    prompts = {_as_str(key): _as_str(value) for key, value in _as_dict(payload).items()}
    prompt_path_pairs = (("deep_agent_system", "deep_agent_system_path"),)
    for prompt_key, prompt_path_key in prompt_path_pairs:
        prompt_path = _as_str(prompts.get(prompt_path_key))
        if not prompt_path:
            continue

        path = Path(prompt_path)
        if not path.is_absolute():
            path = (domain_dir / path).resolve()
        try:
            prompts[prompt_key] = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Invalid {prompt_path_key}: {prompt_path}") from exc
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
class EmbeddingProfile:
    enabled: bool = True
    model: str = "BAAI/bge-base-zh-v1.5"
    device: str = "cpu"
    top_k: int = 4
    persist_root: str = ".vectorstore"
    cache_dir: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbeddingProfile":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            model=_as_str(payload.get("model"), "BAAI/bge-base-zh-v1.5"),
            device=_as_str(payload.get("device"), "cpu"),
            top_k=_as_int(payload.get("top_k"), 4),
            persist_root=_as_str(payload.get("persist_root"), ".vectorstore"),
            cache_dir=_as_str(payload.get("cache_dir")) or None,
        )


@dataclass(frozen=True)
class RerankerProfile:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    top_k: int = 4
    candidate_top_k: int = 20
    batch_size: int = 8
    max_length: int = 512
    cache_dir: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RerankerProfile":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            model=_as_str(payload.get("model"), "BAAI/bge-reranker-base"),
            device=_as_str(payload.get("device"), "cpu"),
            top_k=_as_int(payload.get("top_k"), 4),
            candidate_top_k=_as_int(payload.get("candidate_top_k"), 20),
            batch_size=_as_int(payload.get("batch_size"), 8),
            max_length=_as_int(payload.get("max_length"), 512),
            cache_dir=_as_str(payload.get("cache_dir")) or None,
        )


@dataclass(frozen=True)
class RetrievalProfile:
    presets: dict[str, dict[str, int]] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)
    max_per_source: dict[str, int] = field(default_factory=dict)
    enable_wiki: bool = True
    enable_code: bool = True
    embedding: EmbeddingProfile = field(default_factory=EmbeddingProfile)
    reranker: RerankerProfile = field(default_factory=RerankerProfile)
    hybrid_weights: dict[str, float] = field(default_factory=lambda: {"bm25": 0.30, "embedding": 0.50, "lexical": 0.20})
    module_prior_boost: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalProfile":
        presets_raw = _as_dict(payload.get("presets"))
        presets: dict[str, dict[str, int]] = {}
        for key, value in presets_raw.items():
            row = _as_dict(value)
            presets[_as_str(key)] = {
                "wiki_top_k": _as_int(row.get("wiki_top_k"), 4),
                "code_top_k": _as_int(row.get("code_top_k"), 4),
                "final_top_k": _as_int(row.get("final_top_k"), 6),
            }

        weights_raw = _as_dict(payload.get("source_weights"))
        max_raw = _as_dict(payload.get("max_per_source"))
        hybrid_weights_raw = _as_dict(payload.get("hybrid_weights"))
        return cls(
            presets=presets,
            source_weights={
                "wiki": _as_float(weights_raw.get("wiki"), 1.0),
                "code": _as_float(weights_raw.get("code"), 1.0),
            },
            max_per_source={
                "wiki": _as_int(max_raw.get("wiki"), 4),
                "code": _as_int(max_raw.get("code"), 4),
            },
            enable_wiki=bool(payload.get("enable_wiki", True)),
            enable_code=bool(payload.get("enable_code", True)),
            embedding=EmbeddingProfile.from_dict(_as_dict(payload.get("embedding"))),
            reranker=RerankerProfile.from_dict(_as_dict(payload.get("reranker"))),
            hybrid_weights={
                "bm25": _as_float(hybrid_weights_raw.get("bm25"), 0.30),
                "embedding": _as_float(hybrid_weights_raw.get("embedding"), 0.50),
                "lexical": _as_float(hybrid_weights_raw.get("lexical"), 0.20),
            },
            module_prior_boost=_as_float(payload.get("module_prior_boost"), 0.25),
        )

    def preset(self, strategy: str) -> dict[str, int]:
        if strategy in self.presets:
            return dict(self.presets[strategy])
        return dict(self.presets.get("hybrid", {"wiki_top_k": 4, "code_top_k": 4, "final_top_k": 6}))


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
    retrieval: RetrievalProfile
    answering: AnsweringProfile
    prompts: dict[str, str]
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
            routing={"default_module": _as_str(_as_dict(payload.get("routing")).get("default_module"))},
            modules=modules,
            retrieval=RetrievalProfile.from_dict(_as_dict(payload.get("retrieval"))),
            answering=AnsweringProfile.from_dict(_as_dict(payload.get("answering"))),
            prompts=_build_prompts(_as_dict(payload.get("prompts")), domain_dir=domain_dir),
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
        return " ".join(str(text or "").strip().lower().split())

    def is_pronoun_followup(self, text: str) -> bool:
        pronouns = (
            "它",
            "这个",
            "这个问题",
            "那个",
            "那这个",
            "这块",
            "这里",
            "上面这个",
            "it",
            "that",
            "this",
        )
        normalized = str(text or "").lower()
        return any(token in normalized for token in pronouns)

    def infer_module(self, text: str) -> tuple[str, str]:
        default_module = self.default_module
        default_hint = self.module_hint(default_module)
        if not text:
            return default_module, default_hint

        lowered = text.lower()
        modules = sorted(self.modules, key=lambda item: item.route_priority)

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
        normalized = str(text or "").strip().lower()
        primary_normalized = str(primary_module_name or "").strip().lower()
        if not normalized or limit <= 0:
            return []

        scored_modules: list[tuple[int, int, ModuleProfile]] = []
        for module in self.modules:
            module_normalized = module.name.strip().lower()
            if not module_normalized or module_normalized == primary_normalized:
                continue

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
            rows.append({"module_name": module.name, "module_hint": module.hint})
        return rows

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

def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid profile JSON: {path}") from exc


def _resolve_profile_path(project_root: Path) -> Path:
    explicit = os.getenv("AGENT_DOMAIN_PROFILE_PATH", "").strip()
    if explicit:
        custom = Path(explicit)
        if not custom.is_absolute():
            custom = (project_root / custom).resolve()
        return custom

    domain_dir = os.getenv("AGENT_DOMAIN_DIR", "").strip()
    if domain_dir:
        domain_path = Path(domain_dir)
        if not domain_path.is_absolute():
            domain_path = (project_root / domain_path).resolve()
        if domain_path.is_file():
            return domain_path
        return domain_path / "profile.json"

    profile_id = os.getenv("AGENT_DOMAIN_PROFILE", DEFAULT_PROFILE_ID).strip() or DEFAULT_PROFILE_ID
    profile_dir = os.getenv("AGENT_DOMAIN_PROFILE_DIR", DEFAULT_PROFILE_DIR).strip() or DEFAULT_PROFILE_DIR
    base_dir = Path(profile_dir)
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()
    return base_dir / profile_id / "profile.json"


def resolve_domain_profile_path(*, project_root: Path) -> Path:
    """Resolve the active domain profile path for the current runtime."""
    return _resolve_profile_path(project_root.resolve())


def load_domain_profile(*, project_root: Path) -> DomainProfile:
    profile_path = resolve_domain_profile_path(project_root=project_root)
    payload = _load_json_file(profile_path)
    return DomainProfile.from_dict(payload, domain_dir=profile_path.parent.resolve())


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def set_domain_profile(profile: "DomainProfile") -> None:
    global _PROFILE_SINGLETON, _PROFILE_SINGLETON_ROOT

    with _PROFILE_SINGLETON_LOCK:
        if _PROFILE_SINGLETON is not None:
            import logging

            logging.getLogger(__name__).warning(
                "[DomainProfile] 单例已存在，跳过设置: "
                f"existing={_PROFILE_SINGLETON.profile_id}, "
                f"new={profile.profile_id}"
            )
            return
        _PROFILE_SINGLETON = profile
        _PROFILE_SINGLETON_ROOT = profile.domain_dir.parent.parent if profile.domain_dir else None


def reset_domain_profile_singleton() -> None:
    global _PROFILE_SINGLETON, _PROFILE_SINGLETON_ROOT

    with _PROFILE_SINGLETON_LOCK:
        _PROFILE_SINGLETON = None
        _PROFILE_SINGLETON_ROOT = None
