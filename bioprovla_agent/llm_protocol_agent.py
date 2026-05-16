"""Tailored LLM Protocol Agent: protocol -> TaskPlan via integrations.llm_protocol_parser.parse_protocol."""

from __future__ import annotations

import logging
from typing import Any, Callable

from bioprovla_agent import adapters
from bioprovla_agent.agent_io_log import log_block, truncate
from bioprovla_agent.schemas import RunMode, SubTask, TaskPlan

logger = logging.getLogger(__name__)

_AGENT = "Tailored LLM Protocol Agent"


def _kb_hint_from_atomic(a: dict[str, Any]) -> str | None:
    """Derive a knowledge-base lookup string from atomic action fields."""
    parts = [
        str(a.get("natural_language_instruction", "")).strip(),
        str(a.get("target_object", "")).strip(),
        str(a.get("action_type", "")).strip(),
    ]
    hint = " ".join(p for p in parts if p).strip()
    return hint or None


def _atomic_to_subtask(a: dict[str, Any]) -> SubTask:
    raw_sid = a.get("step_id", 0)
    try:
        step_id = int(raw_sid)
    except (TypeError, ValueError):
        logger.warning("Invalid step_id %r in atomic_action, using 0", raw_sid)
        step_id = 0
    action_type = str(a.get("action_type", "unknown") or "unknown")
    target = str(a.get("target_object", "") or "")
    loc = str(a.get("location_reference", "") or "")
    instr = str(a.get("natural_language_instruction", "") or "").strip()
    pre = str(a.get("precondition", "") or "").strip()
    post = str(a.get("postcondition", "") or "").strip()
    if not instr:
        instr = f"{action_type} {target}".strip()
    kb = _kb_hint_from_atomic(a)
    return SubTask(
        step_id=step_id,
        action_type=action_type,
        target_object=target,
        location_reference=loc,
        natural_language_instruction=instr,
        precondition=pre or "Workspace ready",
        postcondition=post or "Step executed",
        knowledge_base_id=kb,
        raw_atomic_action=dict(a),
    )


def _validate_subtask(s: SubTask) -> list[str]:
    issues: list[str] = []
    if not s.natural_language_instruction:
        issues.append(f"step {s.step_id}: missing instruction")
    if not s.action_type:
        issues.append(f"step {s.step_id}: missing action_type")
    if not s.precondition:
        issues.append(f"step {s.step_id}: missing precondition")
    if not s.postcondition:
        issues.append(f"step {s.step_id}: missing postcondition")
    return issues


class TailoredLLMProtocolAgent:
    """Wraps LLM parsing only (no vision, no KB, no robot)."""

    def __init__(
        self,
        mode: RunMode,
        parse_protocol_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.mode = mode
        self._parse = parse_protocol_fn or adapters.get_parse_protocol()

    def parse_to_plan(self, protocol_text: str) -> tuple[TaskPlan | None, str | None]:
        """
        Returns (TaskPlan, error_message).
        On success error_message is None.
        """
        log_block(
            f"{_AGENT} | INPUT",
            [
                ("run_mode", self.mode.value),
                ("protocol_char_count", len(protocol_text)),
                ("protocol_text_preview", truncate(protocol_text, 600)),
            ],
        )
        if not protocol_text.strip():
            log_block(f"{_AGENT} | OUTPUT", [("status", "error"), ("error", "empty protocol_text")])
            return None, "empty protocol_text"
        if self.mode == RunMode.MOCK:
            mock_sub = SubTask(
                step_id=1,
                action_type="mock",
                target_object="mock",
                location_reference="mock",
                natural_language_instruction="Mock execution for pipeline test",
                precondition="Mock workspace ready",
                postcondition="Mock step done",
                knowledge_base_id="mock",
                raw_atomic_action={},
            )
            plan = TaskPlan(
                reasoning_process={"intent_analysis": "mock"},
                subtasks=[mock_sub],
                raw_llm_result={"mock": True},
            )
            log_block(
                f"{_AGENT} | OUTPUT",
                [
                    ("status", "ok (mock)"),
                    ("subtask_count", len(plan.subtasks)),
                    ("step_ids", [s.step_id for s in plan.subtasks]),
                    ("instructions", [truncate(s.natural_language_instruction, 120) for s in plan.subtasks]),
                ],
            )
            return plan, None

        raw = self._parse(protocol_text)
        if isinstance(raw, dict) and raw.get("error"):
            log_block(
                f"{_AGENT} | OUTPUT",
                [("status", "error"), ("error", str(raw["error"])), ("raw_keys", list(raw.keys()))],
            )
            return None, str(raw["error"])

        if not isinstance(raw, dict):
            log_block(f"{_AGENT} | OUTPUT", [("status", "error"), ("error", "LLM returned non-dict")])
            return None, "LLM returned non-dict"

        if "atomic_actions" not in raw:
            log_block(f"{_AGENT} | OUTPUT", [("status", "error"), ("error", "missing atomic_actions")])
            return None, "missing atomic_actions"

        actions = raw.get("atomic_actions") or []
        if not isinstance(actions, list) or len(actions) == 0:
            log_block(f"{_AGENT} | OUTPUT", [("status", "error"), ("error", "atomic_actions empty")])
            return None, "atomic_actions empty"

        subtasks = [_atomic_to_subtask(a) for a in actions if isinstance(a, dict)]
        if not subtasks:
            log_block(
                f"{_AGENT} | OUTPUT",
                [("status", "error"), ("error", "atomic_actions contained no valid dict entries")],
            )
            return None, "atomic_actions contained no valid dict entries"
        step_ids = [s.step_id for s in subtasks]
        if len(step_ids) != len(set(step_ids)):
            logger.warning("Duplicate step_id values in TaskPlan: %s", step_ids)
        all_issues: list[str] = []
        for s in subtasks:
            all_issues.extend(_validate_subtask(s))
        if all_issues:
            logger.warning("TaskPlan soft issues: %s", all_issues)

        reasoning = raw.get("reasoning_process")
        if not isinstance(reasoning, dict):
            reasoning = {"note": "reasoning_process missing or invalid"}

        plan = TaskPlan(reasoning_process=reasoning, subtasks=subtasks, raw_llm_result=raw)
        step_summary = [
            {
                "step_id": s.step_id,
                "action_type": s.action_type,
                "instruction": truncate(s.natural_language_instruction, 100),
                "precondition": truncate(s.precondition, 80),
                "postcondition": truncate(s.postcondition, 80),
                "knowledge_base_id": truncate(s.knowledge_base_id or "", 80),
            }
            for s in plan.subtasks
        ]
        log_block(
            f"{_AGENT} | OUTPUT",
            [
                ("status", "ok"),
                ("subtask_count", len(plan.subtasks)),
                ("reasoning_process_keys", list(reasoning.keys())),
                ("subtasks_summary", step_summary),
            ],
        )
        cond_lines: list[tuple[str, Any]] = []
        for s in plan.subtasks:
            label = f"step_id={s.step_id} {s.action_type}"
            cond_lines.append((f"{label} | precondition", s.precondition))
            cond_lines.append((f"{label} | postcondition", s.postcondition))
        log_block(f"{_AGENT} | SUBTASK CONDITIONS (full, for VLM)", cond_lines)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s | OUTPUT raw_llm_result (truncated): %s", _AGENT, truncate(str(raw), 4000))
        return plan, None
