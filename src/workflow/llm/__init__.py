# -*- coding: utf-8 -*-
"""LLM-related shared utilities for workflow nodes."""

from workflow.llm.llm_client import (
    CommonLLMConfig,
    CommonLLMRequest,
    CommonLLMResult,
    WorkflowLLMClient,
    WorkflowLLMConfig,
)
from workflow.llm.llm_prompt_utils import (
    build_evidence_block,
    looks_like_reasoning_dump,
    resolve_system_prompt,
)

__all__ = [
    "CommonLLMConfig",
    "CommonLLMRequest",
    "CommonLLMResult",
    "WorkflowLLMClient",
    "WorkflowLLMConfig",
    "build_evidence_block",
    "looks_like_reasoning_dump",
    "resolve_system_prompt",
]

