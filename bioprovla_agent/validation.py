"""Normalize and validate BioProVLA config / plan inputs (defensive helpers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bioprovla_agent.schemas import RunMode

logger = logging.getLogger(__name__)


def coerce_infer_cli_args(value: Any) -> list[str]:
    """Ensure robot infer_cli_args is a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                return [str(x) for x in parsed] if isinstance(parsed, list) else [s]
            except json.JSONDecodeError:
                return [s]
        return [s]
    return [str(value)]


def clamp_int(name: str, value: Any, default: int, lo: int, hi: int | None = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%r, using default %s", name, value, default)
        v = default
    if v < lo:
        logger.warning("Clamping %s from %s to %s", name, v, lo)
        v = lo
    if hi is not None and v > hi:
        logger.warning("Clamping %s from %s to %s", name, v, hi)
        v = hi
    return v


def clamp_float(name: str, value: Any, default: float, lo: float, hi: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r, using default %s", name, value, default)
        v = default
    if v < lo:
        logger.warning("Clamping %s from %s to %s", name, v, lo)
        v = lo
    if hi is not None and v > hi:
        logger.warning("Clamping %s from %s to %s", name, v, hi)
        v = hi
    return v


def parse_run_mode(value: Any, default: RunMode = RunMode.MOCK) -> RunMode:
    if isinstance(value, RunMode):
        return value
    if value is None:
        return default
    try:
        return RunMode(str(value).strip().lower())
    except ValueError:
        logger.warning("Unknown run mode %r, using %s", value, default.value)
        return default


def sanitize_execution_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe for ExecutionConfig(**kwargs)."""
    d = dict(raw)
    d["mode"] = parse_run_mode(d.get("mode"), RunMode.MOCK)
    d["max_parse_retries"] = clamp_int("max_parse_retries", d.get("max_parse_retries", 2), 2, 0, 20)
    d["max_precondition_verify_retries"] = clamp_int(
        "max_precondition_verify_retries", d.get("max_precondition_verify_retries", 2), 2, 0, 50
    )
    d["max_completion_verify_retries"] = clamp_int(
        "max_completion_verify_retries", d.get("max_completion_verify_retries", 2), 2, 0, 50
    )
    d["max_vla_retries"] = clamp_int("max_vla_retries", d.get("max_vla_retries", 2), 2, 0, 20)
    d["max_reorder_attempts"] = clamp_int("max_reorder_attempts", d.get("max_reorder_attempts", 3), 3, 0, 50)
    d["max_precondition_recovery_loops"] = clamp_int(
        "max_precondition_recovery_loops",
        d.get("max_precondition_recovery_loops", 24),
        24,
        1,
        500,
    )
    d["vla_task_time_s"] = clamp_float("vla_task_time_s", d.get("vla_task_time_s", 30.0), 30.0, 1.0, 3600.0)
    d["vla_fps"] = clamp_int("vla_fps", d.get("vla_fps", 30), 30, 1, 120)
    if "prompt_human_on_failure" in d:
        d["prompt_human_on_failure"] = bool(d["prompt_human_on_failure"])
    if "wait_enter_on_precondition_stall" in d:
        d["wait_enter_on_precondition_stall"] = bool(d["wait_enter_on_precondition_stall"])
    if "save_images" in d:
        d["save_images"] = bool(d["save_images"])
    src = str(d.get("vlm_scene_image_source", "processed")).strip().lower()
    if src not in ("processed", "raw_front"):
        logger.warning("Unknown vlm_scene_image_source=%r, using processed", d.get("vlm_scene_image_source"))
        src = "processed"
    d["vlm_scene_image_source"] = src
    raw_cam = d.get("vlm_raw_camera_name", "front")
    d["vlm_raw_camera_name"] = str(raw_cam).strip() or "front"
    rd = d.get("run_dir")
    if rd is not None and rd != "":
        d["run_dir"] = str(Path(str(rd)).expanduser())
    else:
        d["run_dir"] = None
    return d


def validate_kb_path(path: str | None) -> tuple[bool, str | None]:
    """If path is set, ensure file exists."""
    if not path:
        return True, None
    p = Path(path).expanduser()
    if not p.is_file():
        return False, f"knowledge base path is not a file: {p}"
    return True, None


def sanitize_knowledge_base_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``knowledge_base_config`` JSON for ``KnowledgeBaseConfig``."""
    d = dict(raw) if isinstance(raw, dict) else {}
    d["mock_kb"] = bool(d.get("mock_kb", False))
    d["rag_completion_enabled"] = bool(d.get("rag_completion_enabled", True))
    d["rag_precondition_enabled"] = bool(d.get("rag_precondition_enabled", False))
    comp = d.get("completion_rag_index_path")
    comp_s = str(comp).strip() if comp not in (None, "") else None
    if not comp_s and d.get("index_path") not in (None, ""):
        comp_s = str(d["index_path"]).strip()
        if comp_s:
            logger.warning(
                "knowledge_base_config: ``index_path`` is deprecated; rename to "
                "``completion_rag_index_path`` (completion-condition RAG only). Using index_path for this run."
            )
    d["completion_rag_index_path"] = comp_s
    d.pop("index_path", None)
    pre_idx = d.get("precondition_rag_index_path")
    d["precondition_rag_index_path"] = str(pre_idx).strip() if pre_idx not in (None, "") else None
    return d


def validate_real_prerequisites(
    mode: RunMode,
    infer_cli_args: list[str],
    protocol_text: str,
) -> list[str]:
    """Return list of warning or error strings (errors block REAL run)."""
    issues: list[str] = []
    if mode != RunMode.REAL:
        return issues
    if not protocol_text.strip():
        issues.append("REAL mode: protocol_text is empty")
    if not infer_cli_args:
        issues.append("REAL mode: robot_config.infer_cli_args must be non-empty")
    return issues
