"""Retry, reorder, and human-intervention helpers for Guiding Decision Agent."""

from __future__ import annotations

import logging
from typing import Callable

from bioprovla_agent.agent_io_log import log_block, truncate
from bioprovla_agent.schemas import SubTask, TaskPlan

logger = logging.getLogger(__name__)

_RECOVERY = "Recovery policy (reorder / human)"


def maybe_reorder_for_precondition(
    plan: TaskPlan,
    current_index: int,
    reorder_attempts_used: int,
    max_reorder_attempts: int,
) -> tuple[TaskPlan | None, str | None]:
    """
    If precondition for subs[current_index] likely needs another step first,
    move a candidate later step earlier. Returns (new_plan, log) or (None, None).
    """
    subs = plan.subtasks
    if reorder_attempts_used >= max_reorder_attempts:
        log_block(f"{_RECOVERY} | OUTPUT (reorder)", [("reordered", False), ("reason", "max reorder attempts")])
        return None, None
    if current_index < 0 or current_index >= len(subs):
        log_block(f"{_RECOVERY} | OUTPUT (reorder)", [("reordered", False), ("reason", "bad index")])
        return None, None
    log_block(
        f"{_RECOVERY} | INPUT (reorder check)",
        [
            ("current_index", current_index),
            ("reorder_attempts_used", reorder_attempts_used),
            ("max_reorder_attempts", max_reorder_attempts),
            ("current_step_id", subs[current_index].step_id),
        ],
    )
    cur = subs[current_index]
    cur_pre = (cur.precondition + " " + cur.natural_language_instruction).lower()

    def _score_candidate(other: SubTask, idx: int) -> int:
        if idx <= current_index:
            return 0
        ins = other.natural_language_instruction.lower()
        at = other.action_type.lower()
        score = 0
        if "lid" in cur_pre or "open" in cur_pre or "close" in cur_pre:
            if at == "open_lid" or ("open" in ins and "lid" in ins):
                score += 3
            if at == "close_lid" or ("close" in ins and "lid" in ins):
                score += 1
        if "tube" in cur_pre and ("remove" in ins or "take" in ins or "grasp" in ins):
            score += 2
        return score

    best_j = -1
    best_score = 0
    for j in range(len(subs)):
        if j == current_index:
            continue
        sc = _score_candidate(subs[j], j)
        if sc > best_score:
            best_score = sc
            best_j = j

    if best_j < 0 or best_score == 0:
        log_block(
            f"{_RECOVERY} | OUTPUT (reorder)",
            [("reordered", False), ("reason", "no suitable later step"), ("best_score", best_score)],
        )
        return None, None

    new_list = subs[:]
    step = new_list.pop(best_j)
    insert_at = current_index if best_j > current_index else current_index - 1
    insert_at = max(0, min(insert_at, len(new_list)))
    new_list.insert(insert_at, step)
    sig_old = tuple((s.step_id, s.natural_language_instruction) for s in subs)
    sig_new = tuple((s.step_id, s.natural_language_instruction) for s in new_list)
    if sig_old == sig_new:
        log_block(
            f"{_RECOVERY} | OUTPUT (reorder)",
            [("reordered", False), ("reason", "reorder produced identical task order; skipping")],
        )
        return None, None
    msg = f"reorder: moved step_id={step.step_id} before current queue position for precondition relief"
    logger.info(msg)
    new_plan = TaskPlan(
        reasoning_process=plan.reasoning_process,
        subtasks=new_list,
        raw_llm_result=plan.raw_llm_result,
    )
    log_block(
        f"{_RECOVERY} | OUTPUT (reorder)",
        [
            ("reordered", True),
            ("message", msg),
            ("new_step_order", [s.step_id for s in new_plan.subtasks]),
        ],
    )
    return new_plan, msg


def wait_for_environment_reset_confirm(
    *,
    step_id: int,
    precondition_reason: str | None = None,
    sink: Callable[[str], None] | None = None,
) -> None:
    """
    Block until the operator presses Enter after adjusting the physical scene.

    Used in REAL mode when automatic reorder cannot resolve a failed VLM precondition.
    Does not return ``abort``; the caller should re-run precondition verification.
    """
    reason = (precondition_reason or "").strip()
    lines: list[str] = [
        "=" * 72,
        f"Precondition still not satisfied for subtask step_id={step_id}.",
        "Automatic reorder is exhausted or no further reorder is available.",
    ]
    if reason:
        lines.append(f"VLM / verification note: {truncate(reason, 520)}")
    lines.extend(
        [
            "",
            "Adjust the physical scene as needed (layout, camera aim, devices).",
            "When the environment is reset, press Enter: a fresh frame will be captured "
            "and VLM precondition verification will run again.",
            "The run does not exit here; execution continues after this step passes precondition.",
            "=" * 72,
        ]
    )
    banner = "\n".join(lines)
    log_block(
        f"{_RECOVERY} | INPUT (environment_reset_wait)",
        [
            ("step_id", step_id),
            ("reason_preview", truncate(reason, 240) if reason else None),
        ],
    )
    if sink:
        sink(banner)
    else:
        print(banner, flush=True)
    try:
        input()
    except EOFError:
        logger.warning("%s | stdin EOF during environment reset wait; retrying VLM anyway", _RECOVERY)
    log_block(
        f"{_RECOVERY} | OUTPUT (environment_reset_wait)",
        [("acknowledged", True), ("note", "Enter or EOF")],
    )


def human_confirm(
    prompt: str,
    enabled: bool,
    sink: Callable[[str], None] | None = None,
) -> str:
    """
    If enabled, block on stdin. Returns one of: proceed, abort, skip
    """
    log_block(
        f"{_RECOVERY} | INPUT (human_confirm)",
        [("prompt", truncate(prompt, 400)), ("interactive_enabled", enabled)],
    )
    if not enabled:
        log_block(f"{_RECOVERY} | OUTPUT (human_confirm)", [("choice", "abort"), ("note", "prompt_human_on_failure is false")])
        return "abort"
    if sink:
        sink(prompt)
    try:
        line = input(f"{prompt} [proceed/abort/skip]: ").strip().lower()
    except EOFError:
        log_block(f"{_RECOVERY} | OUTPUT (human_confirm)", [("choice", "abort"), ("note", "EOF on stdin")])
        return "abort"
    if line in ("p", "proceed", "y", "yes"):
        log_block(f"{_RECOVERY} | OUTPUT (human_confirm)", [("choice", "proceed")])
        return "proceed"
    if line in ("s", "skip"):
        log_block(f"{_RECOVERY} | OUTPUT (human_confirm)", [("choice", "skip")])
        return "skip"
    log_block(f"{_RECOVERY} | OUTPUT (human_confirm)", [("choice", "abort"), ("raw_line", line or "")])
    return "abort"
