# -*- coding: utf-8 -*-
"""Shared helper utilities for workflow nodes."""

from workflow.common.module_inference import infer_module
from workflow.common.node_trace import append_node_trace
from workflow.common.query_followup import (
    is_pronoun_followup,
    looks_like_code_location_query,
    looks_like_context_dependent_followup,
)
from workflow.common.evidence import collect_evidence_hits
from workflow.common.llm_client import (
    CommonLLMCapability,
    CommonLLMConfig,
    CommonLLMRequest,
    CommonLLMResult,
    WorkflowLLMClient,
    WorkflowLLMConfig,
)
from workflow.common.llm_prompt_utils import build_evidence_block, looks_like_reasoning_dump, resolve_system_prompt

__all__ = [
    "append_node_trace",
    "collect_evidence_hits",
    "CommonLLMCapability",
    "CommonLLMConfig",
    "CommonLLMRequest",
    "CommonLLMResult",
    "WorkflowLLMClient",
    "WorkflowLLMConfig",
    "build_evidence_block",
    "looks_like_reasoning_dump",
    "resolve_system_prompt",
    "infer_module",
    "is_pronoun_followup",
    "looks_like_code_location_query",
    "looks_like_context_dependent_followup",
]
