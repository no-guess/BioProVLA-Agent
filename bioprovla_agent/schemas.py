"""BioProVLA-Agent shared data models (English-only public fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class RunMode(str, Enum):
    """Execution mode for the closed loop."""

    MOCK = "mock"
    DRY_RUN = "dry_run"
    REAL = "real"


class VerificationType(str, Enum):
    PRECONDITION = "precondition"
    COMPLETION = "completion"


class SuggestedAction(str, Enum):
    CONTINUE = "continue"
    RETRY_VERIFICATION = "retry_verification"
    RETRY_EXECUTION = "retry_execution"
    REORDER = "reorder"
    HUMAN_INTERVENTION = "human_intervention"
    ABORT = "abort"


@dataclass
class SubTask:
    """Single executable subtask derived from LLM atomic_actions."""

    step_id: int
    action_type: str
    target_object: str
    location_reference: str
    natural_language_instruction: str
    precondition: str
    postcondition: str
    knowledge_base_id: str | None = None
    raw_atomic_action: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Structured plan returned by the LLM protocol agent."""

    reasoning_process: dict[str, Any]
    subtasks: list[SubTask]
    raw_llm_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """VLM-RAG verification outcome."""

    passed: bool
    verification_type: VerificationType
    reason: str
    raw_response: str | None
    suggested_action: SuggestedAction
    kb_matched: bool
    image_paths_used: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """VLA embodied execution outcome (no semantic success claim)."""

    success: bool
    duration_s: float
    finish_reason: str
    image_paths: list[str]  # After-subtask snapshot path(s) for completion VLM only (typically one).
    robot_state_before: Any | None
    robot_state_after: Any | None
    error: str | None = None


@dataclass
class ApiCredentialsConfig:
    """
    Optional FastGPT / OpenAI-compatible credentials from the run JSON.

    Non-empty fields override environment variables for this process after
    ``load_run_config``. If ``vlm_api_key`` is omitted, ``llm_api_key`` is used for VLM
    when set (same key for both).
    """

    llm_api_key: str | None = None
    llm_api_root: str | None = None
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None


@dataclass
class ModelConfig:
    """Optional model identifiers (placeholders for CLI / env wiring)."""

    llm_model: str | None = None
    vlm_model: str = "qwen-vl-plus"
    vla_policy_path: str | None = None


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base paths and RAG toggles (completion vs precondition are separate)."""

    #: JSON index for **completion** RAG only (e.g. ``img_index``): success/failure refs.
    completion_rag_index_path: str | None = None
    mock_kb: bool = False
    #: When true, completion checks load ``completion_rag_index_path`` and use motion-sequence RAG.
    rag_completion_enabled: bool = True
    #: When true, precondition checks use ``precondition_rag_index_path`` only (never the completion index).
    rag_precondition_enabled: bool = False
    #: JSON index for **precondition** RAG only; used when ``rag_precondition_enabled``.
    precondition_rag_index_path: str | None = None


@dataclass
class RobotConfig:
    """LeRobot infer CLI fragments for real VLA (parsed with draccus into InferConfig)."""

    # Example: ["--policy.path=hf://...", "--robot.type=koch_follower", "--fps=30"]
    infer_cli_args: list[str] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    """Loop limits and behavior flags."""

    mode: RunMode = RunMode.MOCK
    max_parse_retries: int = 2
    max_precondition_verify_retries: int = 2
    max_completion_verify_retries: int = 2
    max_vla_retries: int = 2
    max_reorder_attempts: int = 3
    # Caps precondition fail -> reorder -> re-verify loops without advancing queue index.
    max_precondition_recovery_loops: int = 24
    prompt_human_on_failure: bool = False
    #: REAL: when automatic reorder cannot fix a failed VLM precondition, block on Enter
    #: and retry precondition (never auto-abort this branch).
    wait_enter_on_precondition_stall: bool = True
    run_dir: str | None = None
    vla_task_time_s: float = 30.0
    vla_fps: int = 30
    save_images: bool = True
    #: ``processed`` = same tensors as policy (after ``robot_observation_processor``).
    #: ``raw_front`` = ``robot.cameras[vlm_raw_camera_name].read_latest()`` (uint8 RGB from driver).
    vlm_scene_image_source: Literal["processed", "raw_front"] = "processed"
    #: Logical camera name in ``robot.cameras`` (matches ``--robot.cameras={ front: ... }``).
    vlm_raw_camera_name: str = "front"


@dataclass
class SubTaskRecord:
    """Per-subtask audit trail."""

    subtask: SubTask
    precondition: VerificationResult | None = None
    execution: ExecutionResult | None = None
    completion: VerificationResult | None = None
    precondition_retries: int = 0
    completion_retries: int = 0
    vla_retries: int = 0
    reorder_events: list[str] = field(default_factory=list)
    human_notes: list[str] = field(default_factory=list)
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"


@dataclass
class RunReport:
    """Final aggregated report."""

    protocol_text: str
    task_plan: TaskPlan | None
    subtask_records: list[SubTaskRecord]
    reorder_log: list[str]
    human_interventions: list[str]
    failure_summaries: list[str]
    total_duration_s: float
    overall_success: bool
    run_directory: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""

        def _sub(s: SubTask) -> dict[str, Any]:
            return {
                "step_id": s.step_id,
                "action_type": s.action_type,
                "target_object": s.target_object,
                "location_reference": s.location_reference,
                "natural_language_instruction": s.natural_language_instruction,
                "precondition": s.precondition,
                "postcondition": s.postcondition,
                "knowledge_base_id": s.knowledge_base_id,
            }

        def _vr(v: VerificationResult | None) -> dict[str, Any] | None:
            if v is None:
                return None
            return {
                "passed": v.passed,
                "verification_type": v.verification_type.value,
                "reason": v.reason,
                "suggested_action": v.suggested_action.value,
                "kb_matched": v.kb_matched,
                "image_paths_used": list(v.image_paths_used),
            }

        def _er(e: ExecutionResult | None) -> dict[str, Any] | None:
            if e is None:
                return None
            return {
                "success": e.success,
                "duration_s": e.duration_s,
                "finish_reason": e.finish_reason,
                "image_paths": list(e.image_paths),
                "error": e.error,
            }

        records = []
        for r in self.subtask_records:
            records.append(
                {
                    "subtask": _sub(r.subtask),
                    "precondition": _vr(r.precondition),
                    "execution": _er(r.execution),
                    "completion": _vr(r.completion),
                    "precondition_retries": r.precondition_retries,
                    "completion_retries": r.completion_retries,
                    "vla_retries": r.vla_retries,
                    "reorder_events": list(r.reorder_events),
                    "human_notes": list(r.human_notes),
                    "status": r.status,
                }
            )

        plan_dict: dict[str, Any] | None = None
        if self.task_plan is not None:
            plan_dict = {
                "reasoning_process": self.task_plan.reasoning_process,
                "subtasks": [_sub(s) for s in self.task_plan.subtasks],
            }

        return {
            "protocol_text": self.protocol_text,
            "task_plan": plan_dict,
            "subtask_records": records,
            "reorder_log": list(self.reorder_log),
            "human_interventions": list(self.human_interventions),
            "failure_summaries": list(self.failure_summaries),
            "total_duration_s": self.total_duration_s,
            "overall_success": self.overall_success,
            "run_directory": self.run_directory,
        }
