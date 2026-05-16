"""Merge JSON ``api_credentials`` with environment and push into integration modules."""

from __future__ import annotations

import logging
import os

from bioprovla_agent.schemas import ApiCredentialsConfig

logger = logging.getLogger(__name__)


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    t = str(value).strip()
    return t if t else None


def apply_api_credentials(creds: ApiCredentialsConfig) -> None:
    """
    Resolve LLM/VLM keys and endpoints (JSON overrides env when the JSON field is non-empty).

    VLM key resolution: ``vlm_api_key`` -> ``llm_api_key`` -> ``FASTGPT_VLM_API_KEY`` -> ``OPENAI_API_KEY``.
    """
    from bioprovla_agent.integrations import llm_protocol_parser, vlm_rag_backend

    llm_key = _strip_or_none(creds.llm_api_key) or os.environ.get("FASTGPT_API_KEY", "")
    llm_root = (
        _strip_or_none(creds.llm_api_root)
        or os.environ.get("FASTGPT_API_ROOT")
        or llm_protocol_parser.DEFAULT_LLM_API_ROOT
    )

    vlm_key = (
        _strip_or_none(creds.vlm_api_key)
        or _strip_or_none(creds.llm_api_key)
        or os.environ.get("FASTGPT_VLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    vlm_base = (
        _strip_or_none(creds.vlm_base_url)
        or os.environ.get("FASTGPT_VLM_BASE_URL")
        or vlm_rag_backend.DEFAULT_VLM_BASE_URL
    )

    llm_protocol_parser.configure_api(llm_key, llm_root)
    vlm_rag_backend.configure_api(vlm_key, vlm_base)

    logger.info(
        "API credentials applied (LLM key present: %s, VLM key present: %s)",
        bool(llm_key),
        bool(vlm_key),
    )
