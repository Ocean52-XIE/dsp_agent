# -*- coding: utf-8 -*-
"""Compatibility shim for legacy imports.

Actual LLM client abstraction now lives in `workflow.common.llm_client`.
Node-specific prompt logic is implemented in each node module.
"""
from __future__ import annotations

from workflow.common.llm_client import WorkflowLLMClient, WorkflowLLMConfig

KnowledgeQALLMClient = WorkflowLLMClient
KnowledgeQALLMConfig = WorkflowLLMConfig

__all__ = ["KnowledgeQALLMClient", "KnowledgeQALLMConfig"]
