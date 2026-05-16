"""Load BioProVLA run configuration from JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from bioprovla_agent.api_credentials import apply_api_credentials
from bioprovla_agent.schemas import (
    ApiCredentialsConfig,
    ExecutionConfig,
    KnowledgeBaseConfig,
    ModelConfig,
    RobotConfig,
)
from bioprovla_agent.validation import (
    coerce_infer_cli_args,
    sanitize_execution_config_dict,
    sanitize_knowledge_base_config_dict,
    validate_kb_path,
)


def load_run_config(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON config file and return instantiated dataclass objects:
    model_config, robot_config, execution_config, knowledge_base_config, protocol_text.
    """
    p = Path(path).expanduser().resolve()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a JSON object, got {type(raw).__name__}")

    protocol_text = str(raw.get("protocol_text", "")).strip()
    model = ModelConfig(**{k: v for k, v in raw.get("model_config", {}).items() if k in ModelConfig.__dataclass_fields__})
    robot_raw = dict(raw.get("robot_config", {}))
    if "infer_cli_args" in robot_raw:
        robot_raw["infer_cli_args"] = coerce_infer_cli_args(robot_raw.get("infer_cli_args"))
    robot = RobotConfig(
        **{k: v for k, v in robot_raw.items() if k in RobotConfig.__dataclass_fields__}
    )
    kb_raw = sanitize_knowledge_base_config_dict(raw.get("knowledge_base_config") or {})
    kb = KnowledgeBaseConfig(
        **{k: v for k, v in kb_raw.items() if k in KnowledgeBaseConfig.__dataclass_fields__}
    )
    ex_raw = sanitize_execution_config_dict(raw.get("execution_config") or {})
    execution = ExecutionConfig(
        **{k: v for k, v in ex_raw.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    kb_ok, kb_err = validate_kb_path(kb.completion_rag_index_path)
    if not kb_ok:
        logger.warning("Knowledge base path check: %s", kb_err)
    if kb.rag_precondition_enabled:
        pre_ok, pre_err = validate_kb_path(kb.precondition_rag_index_path)
        if not pre_ok:
            logger.warning("Precondition RAG index path check: %s", pre_err)
    elif kb.precondition_rag_index_path:
        logger.warning(
            "precondition_rag_index_path is set but rag_precondition_enabled is false; "
            "the file will not be loaded until you enable rag_precondition_enabled."
        )

    api_raw = raw.get("api_credentials")
    if api_raw is None:
        api_raw = {}
    elif not isinstance(api_raw, dict):
        raise ValueError("api_credentials must be a JSON object if present")
    api_creds = ApiCredentialsConfig(
        **{k: v for k, v in api_raw.items() if k in ApiCredentialsConfig.__dataclass_fields__}
    )
    apply_api_credentials(api_creds)

    from bioprovla_agent.integrations import llm_protocol_parser as _llm_mod

    if not (_llm_mod.API_KEY or "").strip():
        logger.warning(
            "After loading %s: LLM API key is still empty (api_credentials.llm_api_key in JSON "
            "and FASTGPT_API_KEY in the environment are both unset). Save the JSON if you edited it, "
            "then run again.",
            p.name,
        )
    from bioprovla_agent.integrations import vlm_rag_backend as _vlm_mod

    if not (_vlm_mod._VLM_API_KEY or "").strip():
        logger.warning(
            "After loading %s: VLM API key is still empty (set api_credentials.vlm_api_key or llm_api_key, "
            "or FASTGPT_VLM_API_KEY / OPENAI_API_KEY). VLM calls in REAL will fail until fixed.",
            p.name,
        )

    return {
        "protocol_text": protocol_text,
        "model_config": model,
        "robot_config": robot,
        "execution_config": execution,
        "knowledge_base_config": kb,
        "api_credentials": api_creds,
        "raw": raw,
    }


def dump_example_config(path: str | Path) -> None:
    """Write a template JSON config for users to copy."""
    example = {
        "protocol_text": "Open the centrifuge lid, remove the tube, then close the lid.",
        "model_config": {
            "llm_model": "qwen3.5-plus",
            "vlm_model": "qwen-vl-plus",
            "vla_policy_path": None,
        },
        "robot_config": {"infer_cli_args": []},
        "knowledge_base_config": {
            "completion_rag_index_path": None,
            "mock_kb": False,
            "rag_completion_enabled": True,
            "rag_precondition_enabled": False,
            "precondition_rag_index_path": None,
        },
        "api_credentials": {
            "llm_api_key": None,
            "llm_api_root": None,
            "vlm_api_key": None,
            "vlm_base_url": None,
        },
        "execution_config": {
            "mode": "mock",
            "max_parse_retries": 2,
            "max_precondition_verify_retries": 2,
            "max_completion_verify_retries": 2,
            "max_vla_retries": 2,
            "max_reorder_attempts": 3,
            "max_precondition_recovery_loops": 24,
            "prompt_human_on_failure": False,
            "wait_enter_on_precondition_stall": True,
            "run_dir": None,
            "vla_task_time_s": 30.0,
            "vla_fps": 30,
            "save_images": True,
            "vlm_scene_image_source": "processed",
            "vlm_raw_camera_name": "front",
        },
    }
    Path(path).write_text(json.dumps(example, indent=2), encoding="utf-8")


def config_to_dict(
    model: ModelConfig,
    robot: RobotConfig,
    execution: ExecutionConfig,
    kb: KnowledgeBaseConfig,
    api: ApiCredentialsConfig | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_config": asdict(model),
        "robot_config": asdict(robot),
        "execution_config": {**asdict(execution), "mode": execution.mode.value},
        "knowledge_base_config": asdict(kb),
    }
    if api is not None:
        out["api_credentials"] = asdict(api)
    return out
