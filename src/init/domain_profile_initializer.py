# -*- coding: utf-8 -*-
"""Domain profile initializer."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def init_domain_profile(
    *,
    project_root: Path | None = None,
) -> Any:
    """Load and initialize the domain profile singleton."""
    from domain_profile import get_domain_profile

    if project_root is None:
        return None
    return get_domain_profile(project_root=project_root)


def get_domain_profile_status(profile: Any) -> dict[str, Any]:
    """Return a lightweight status summary for the loaded domain profile."""
    if profile is None:
        return {
            "status": "not_loaded",
        }

    return {
        "status": "loaded",
        "profile_id": getattr(profile, "profile_id", None),
        "display_name": getattr(profile, "display_name", None),
        "modules_count": len(getattr(profile, "modules", [])),
        "language": getattr(profile, "language", None),
    }
