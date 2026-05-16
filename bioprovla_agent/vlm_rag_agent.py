"""VLM-RAG Verification Agent: precondition and completion via integrations.vlm_rag_backend."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from bioprovla_agent import adapters
from bioprovla_agent.agent_io_log import log_block, truncate
from bioprovla_agent.schemas import (
    ExecutionConfig,
    KnowledgeBaseConfig,
    RunMode,
    SubTask,
    SuggestedAction,
    VerificationResult,
    VerificationType,
)

logger = logging.getLogger(__name__)

_AGENT = "VLM-RAG Verification Agent"


def _emit_vlm_result(result: VerificationResult) -> VerificationResult:
    log_block(
        f"{_AGENT} | OUTPUT",
        [
            ("verification_type", result.verification_type.value),
            ("passed", result.passed),
            ("kb_matched", result.kb_matched),
            ("suggested_action", result.suggested_action.value),
            ("reason", truncate(result.reason, 1200)),
            ("image_paths_used", result.image_paths_used),
            (
                "raw_response_preview",
                truncate(result.raw_response, 800) if result.raw_response else None,
            ),
        ],
    )
    if result.raw_response and logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s | OUTPUT full raw_response:\n%s", _AGENT, result.raw_response)
    return result


def _completion_rag_reason_from_raw(raw: str, how: str) -> str:
    """Human-facing reason for RAG completion: parser tag + VLM goal line (not just 'parsed …')."""
    raw = (raw or "").strip()
    if not raw:
        return how
    m = re.search(
        r"2\.\s*Goal\s+Achievement\s+Verification:\s*(.+?)(?=\n\s*3\.|\n\s*4\.|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        goal = re.sub(r"\s+", " ", m.group(1).strip())
        if goal:
            return f"{how} — {truncate(goal, 900)}"
    tail = raw[:1200].strip()
    return f"{how} — {truncate(tail, 900)}" if tail else how


def _parse_pass_fail_from_completion_text(text: str) -> tuple[bool, str]:
    """Heuristic parse for completion-style VLM output."""
    lower = text.lower()
    m = re.search(r"final decision:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        decision = m.group(1).lower()
        passed = "completed" in decision and "fail" not in decision
        return passed, "parsed Final Decision line"
    if "task fully completed" in lower and "task failed" not in lower[: min(400, len(lower))]:
        return True, "heuristic: task fully completed"
    if "task failed" in lower:
        return False, "heuristic: task failed mentioned"
    return False, "unclear VLM output"


def _parse_precondition_verdict(text: str) -> tuple[bool, str]:
    """Expect VERDICT: READY or VERDICT: NOT_READY (also NOT READY) in precondition output."""
    m = re.search(r"VERDICT:\s*(READY|NOT_READY|NOT\s+READY)\b", text, flags=re.IGNORECASE)
    if m:
        label = re.sub(r"\s+", "_", m.group(1).upper().strip())
        return label == "READY", "parsed VERDICT"
    lower = text.lower()
    if "not_ready" in lower or "not ready" in lower:
        return False, "heuristic not ready"
    if "ready" in lower and "not" not in lower[:120]:
        return True, "heuristic ready"
    return False, "unclear precondition output"


def _vlm_explanation_after_verdict(raw: str) -> str:
    """Text after VERDICT line (or same-line tail); empty if model omitted explanation."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.search(
        r"VERDICT:\s*(?:READY|NOT_READY|NOT\s+READY)\b\s*(.*)$",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        rest = (m.group(1) or "").strip()
        if rest:
            return rest
    lines = [ln.rstrip() for ln in raw.splitlines()]
    non_empty = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty:
        return ""
    _, first = non_empty[0]
    if re.match(r"^VERDICT:\s*(?:READY|NOT_READY|NOT\s+READY)\s*$", first, re.I):
        return "\n".join(ln.strip() for _, ln in non_empty[1:]).strip()
    if not re.search(r"VERDICT:", raw, re.I):
        return raw
    return ""


def _zh_operator_hint(text: str) -> str:
    """Short Mandarin hints for common failure patterns (operator-facing)."""
    if not text:
        return ""
    low = text.lower()
    hints: list[str] = []
    no_cent = any(
        p in low
        for p in (
            "no centrifuge",
            "no visible centrifuge",
            "centrifuge is not visible",
            "centrifuge lid is not visible",
            "cannot see any centrifuge",
            "cannot confirm whether the centrifuge",
            "no indication that the centrifuge",
            "no visible centrifuge or",
            "no equipment",
            "not visible in the image",
            "does not show any",
        )
    )
    if no_cent and ("centrifuge" in low or "equipment" in low or "visible" in low or "image" in low):
        hints.append("提示：画面中可能未清晰看到离心机，请将离心机移入视野或调整相机角度。")
    lid_issue = ("lid" in low or "cover" in low) and any(
        p in low
        for p in (
            "closed",
            "not open",
            "not opened",
            "not fully open",
            "obstructed",
            "not visible",
            "not unlocked",
            "locked",
        )
    )
    if lid_issue:
        hints.append("提示：离心机盖可能未打开、未完全打开或存在遮挡；请先开盖或移开遮挡物。")
    if "hand" in low and ("near" in low or "operator" in low):
        hints.append("提示：检测到手部靠近设备，请注意安全并尽量移开以免误判。")
    return "\n".join(dict.fromkeys(hints))


def _build_precondition_reason(*, how: str, raw: str) -> str:
    expl = _vlm_explanation_after_verdict(raw)
    blob = expl or raw.strip()
    zh = _zh_operator_hint(blob)
    if expl:
        parts = [expl.strip()]
        if zh:
            parts.append(zh)
        return "\n".join(parts)[:2400]
    excerpt = truncate(raw.strip(), 900) if raw.strip() else ""
    if excerpt:
        return f"{how}: model did not add a clear explanation after VERDICT. Excerpt: {excerpt}"
    return f"{how}: empty VLM reply. Check API and prompt."


class VLMRAGVerificationAgent:
    """Uses KB + VLM; mock/dry_run can short-circuit."""

    def __init__(
        self,
        mode: RunMode,
        kb_cfg: KnowledgeBaseConfig,
        fastgpt_module: Any | None = None,
    ) -> None:
        self.mode = mode
        self.kb_cfg = kb_cfg
        self._fg = fastgpt_module
        if fastgpt_module is None and mode in (RunMode.REAL, RunMode.DRY_RUN):
            self._fg = adapters.load_fastgpt_vlm_module()
        self._pre_kb: list[dict[str, Any]] | None = None
        self._pre_kb_path: str | None = None

    def _load_precondition_kb_list(self) -> list[dict[str, Any]]:
        """Load optional precondition-only KB (never ``completion_rag_index_path`` / img_index)."""
        if not self.kb_cfg.rag_precondition_enabled:
            return []
        path = (self.kb_cfg.precondition_rag_index_path or "").strip()
        if not path or self._fg is None:
            return []
        if self._pre_kb is not None and self._pre_kb_path == path:
            return self._pre_kb
        self._pre_kb = self._fg.load_knowledge_base(path)
        self._pre_kb_path = path
        return self._pre_kb

    def _ensure_kb(self) -> None:
        if self._fg is None:
            return
        path = self.kb_cfg.completion_rag_index_path
        if path:
            self._fg.reload_knowledge_base(path)

    def _verify_completion_without_rag(
        self,
        subtask: SubTask,
        image_paths: list[str],
        robot_state_summary: str | None,
    ) -> VerificationResult:
        """Post-step VLM using LLM ``postcondition`` only (``img_index`` / completion RAG disabled)."""
        if self._fg is None:
            return VerificationResult(
                passed=False,
                verification_type=VerificationType.COMPLETION,
                reason="VLM module not loaded",
                raw_response=None,
                suggested_action=SuggestedAction.HUMAN_INTERVENTION,
                kb_matched=False,
                image_paths_used=list(image_paths),
            )
        post = subtask.postcondition
        intro = f"""You are a lab robotics completion verifier (POST-EXECUTION ONLY).
You receive **one** image: the workspace **after** the subtask was attempted (current end state).

Subtask instruction: {subtask.natural_language_instruction}
Required postcondition (camera-verifiable): {post}
Robot state summary: {robot_state_summary or 'unknown'}

Reference-image RAG is OFF: use only the text above and that single image. Do not infer unseen prior motion.

Respond in English with this shape:
1. Final Decision: [Task Fully Completed] OR [Task Failed]
2. Justification: 2-5 sentences comparing the image to the postcondition.

If uncertain, output Task Failed and explain.
"""
        content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
        for p in image_paths:
            b64 = self._fg.encode_image(p)
            if b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        if not any(c.get("type") == "image_url" for c in content):
            return VerificationResult(
                passed=False,
                verification_type=VerificationType.COMPLETION,
                reason="could not encode any images",
                raw_response=None,
                suggested_action=SuggestedAction.RETRY_VERIFICATION,
                kb_matched=False,
                image_paths_used=list(image_paths),
            )
        try:
            resp = self._fg.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{"role": "user", "content": content}],
                max_tokens=900,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            logger.exception("%s | completion (no RAG) VLM call failed: %s", _AGENT, e)
            return VerificationResult(
                passed=False,
                verification_type=VerificationType.COMPLETION,
                reason=str(e),
                raw_response=None,
                suggested_action=SuggestedAction.RETRY_VERIFICATION,
                kb_matched=False,
                image_paths_used=list(image_paths),
            )
        passed, how = _parse_pass_fail_from_completion_text(raw)
        sugg = SuggestedAction.CONTINUE if passed else SuggestedAction.RETRY_EXECUTION
        if how.startswith("unclear"):
            sugg = SuggestedAction.RETRY_VERIFICATION
        reason_out = truncate(raw.strip(), 1200) if raw.strip() else how
        return VerificationResult(
            passed=passed,
            verification_type=VerificationType.COMPLETION,
            reason=reason_out,
            raw_response=raw,
            suggested_action=sugg,
            kb_matched=False,
            image_paths_used=list(image_paths),
        )

    def verify(
        self,
        subtask: SubTask,
        verification_type: VerificationType,
        image_paths: list[str],
        robot_state_summary: str | None,
        _execution_cfg: ExecutionConfig,
    ) -> VerificationResult:
        """Run precondition or completion verification."""
        log_block(
            f"{_AGENT} | INPUT",
            [
                ("run_mode", self.mode.value),
                ("verification_type", verification_type.value),
                ("step_id", subtask.step_id),
                ("action_type", subtask.action_type),
                ("natural_language_instruction", truncate(subtask.natural_language_instruction, 220)),
                ("precondition", truncate(subtask.precondition, 200)),
                ("postcondition", truncate(subtask.postcondition, 200)),
                ("knowledge_base_id", truncate(subtask.knowledge_base_id or "", 120)),
                ("image_path_count", len(image_paths)),
                ("image_paths", list(image_paths)),
                (
                    "robot_state_summary",
                    truncate(robot_state_summary, 160) if robot_state_summary else None,
                ),
                ("kb_completion_rag_index_path", self.kb_cfg.completion_rag_index_path),
                ("mock_kb", self.kb_cfg.mock_kb),
                ("rag_completion_enabled", self.kb_cfg.rag_completion_enabled),
                ("rag_precondition_enabled", self.kb_cfg.rag_precondition_enabled),
                ("precondition_rag_index_path", self.kb_cfg.precondition_rag_index_path),
            ],
        )
        image_paths = list(image_paths)
        if verification_type == VerificationType.COMPLETION and len(image_paths) > 1:
            kept = image_paths[-1]
            image_paths = [kept]
            logger.info(
                "%s | completion | multiple paths supplied; using single after-execution image only: %s",
                _AGENT,
                kept,
            )
        if self.mode == RunMode.REAL and image_paths:
            existing = [p for p in image_paths if Path(p).is_file()]
            if len(existing) < len(image_paths):
                missing = [p for p in image_paths if p not in existing]
                logger.warning(
                    "%s | Discarding %s image path(s) that are not files on disk (showing up to 8): %s",
                    _AGENT,
                    len(missing),
                    missing[:8],
                )
            image_paths = existing

        if self.mode == RunMode.MOCK:
            return _emit_vlm_result(
                VerificationResult(
                    passed=True,
                    verification_type=verification_type,
                    reason="mock_mode bypass",
                    raw_response=None,
                    suggested_action=SuggestedAction.CONTINUE,
                    kb_matched=True,
                    image_paths_used=list(image_paths),
                )
            )

        if self.mode == RunMode.DRY_RUN and not image_paths:
            return _emit_vlm_result(
                VerificationResult(
                    passed=True,
                    verification_type=verification_type,
                    reason="dry_run: no images, simulated pass",
                    raw_response=None,
                    suggested_action=SuggestedAction.CONTINUE,
                    kb_matched=False,
                    image_paths_used=[],
                )
            )

        if self.mode == RunMode.REAL and not image_paths:
            return _emit_vlm_result(
                VerificationResult(
                    passed=False,
                    verification_type=verification_type,
                    reason=(
                        "real mode: no image paths (VLM not called). "
                        "The run did not supply an after-execution snapshot—often because the post snapshot failed "
                        "(robot read errors, or processed observation keys not matching RGB tensors). "
                        "Set execution_config.vlm_scene_image_source to raw_front if OpenCV cameras work but "
                        "motor reads fail, or fix observation image keys."
                    ),
                    raw_response=None,
                    suggested_action=SuggestedAction.RETRY_VERIFICATION,
                    kb_matched=False,
                    image_paths_used=[],
                )
            )

        if self._fg is None:
            return _emit_vlm_result(
                VerificationResult(
                    passed=False,
                    verification_type=verification_type,
                    reason="VLM module not loaded",
                    raw_response=None,
                    suggested_action=SuggestedAction.HUMAN_INTERVENTION,
                    kb_matched=False,
                    image_paths_used=list(image_paths),
                )
            )

        self._ensure_kb()

        if self.kb_cfg.mock_kb or not getattr(self._fg, "KNOWLEDGE_BASE", []):
            if self.mode != RunMode.REAL:
                return _emit_vlm_result(
                    VerificationResult(
                        passed=True,
                        verification_type=verification_type,
                        reason="mock_kb or empty KB in non-real mode",
                        raw_response=None,
                        suggested_action=SuggestedAction.CONTINUE,
                        kb_matched=False,
                        image_paths_used=list(image_paths),
                    )
                )

        task_key = (subtask.knowledge_base_id or subtask.natural_language_instruction or "").strip()
        target_task: dict[str, Any] | None = None
        kb_lookup_label = "none"

        if verification_type == VerificationType.COMPLETION:
            if self.kb_cfg.rag_completion_enabled:
                kb_lookup_label = "completion_index"
                target_task = self._fg.find_task_by_description(task_key) or self._fg.find_task_by_keyword_fallback(
                    task_key
                )
        elif verification_type == VerificationType.PRECONDITION:
            if self.kb_cfg.rag_precondition_enabled:
                pre_kb = self._load_precondition_kb_list()
                if pre_kb:
                    kb_lookup_label = "precondition_rag_index"
                    target_task = self._fg.find_task_by_description_in(
                        task_key, pre_kb
                    ) or self._fg.find_task_by_keyword_fallback_in(task_key, pre_kb)
                elif (self.kb_cfg.precondition_rag_index_path or "").strip():
                    logger.warning(
                        "%s | rag_precondition_enabled but precondition KB empty or failed to load from %s",
                        _AGENT,
                        self.kb_cfg.precondition_rag_index_path,
                    )

        if target_task:
            logger.info(
                "%s | KB lookup (%s) | task_key=%r matched_description=%r index=%s",
                _AGENT,
                kb_lookup_label,
                truncate(task_key, 120),
                truncate(str(target_task.get("task_description", "")), 120),
                target_task.get("index"),
            )
        else:
            logger.info(
                "%s | KB lookup (%s) | task_key=%r matched_description=None",
                _AGENT,
                kb_lookup_label,
                truncate(task_key, 120),
            )

        if verification_type == VerificationType.COMPLETION:
            if not self.kb_cfg.rag_completion_enabled:
                return _emit_vlm_result(
                    self._verify_completion_without_rag(subtask, image_paths, robot_state_summary)
                )
            if not target_task:
                return _emit_vlm_result(
                    VerificationResult(
                        passed=False,
                        verification_type=verification_type,
                        reason="no KB task match for completion verification",
                        raw_response=None,
                        suggested_action=SuggestedAction.HUMAN_INTERVENTION,
                        kb_matched=False,
                        image_paths_used=list(image_paths),
                    )
                )
            raw = self._fg.verify_robotic_arm_motion_with_response(
                target_task["task_description"],
                image_paths,
                completion_criteria=subtask.postcondition,
            )
            if raw is None:
                return _emit_vlm_result(
                    VerificationResult(
                        passed=False,
                        verification_type=verification_type,
                        reason="VLM call failed or no KB match",
                        raw_response=None,
                        suggested_action=SuggestedAction.RETRY_VERIFICATION,
                        kb_matched=target_task is not None,
                        image_paths_used=list(image_paths),
                    )
                )
            passed, how = _parse_pass_fail_from_completion_text(raw)
            sugg = SuggestedAction.CONTINUE if passed else SuggestedAction.RETRY_EXECUTION
            if how.startswith("unclear"):
                sugg = SuggestedAction.RETRY_VERIFICATION
            reason_out = _completion_rag_reason_from_raw(raw, how)
            return _emit_vlm_result(
                VerificationResult(
                    passed=passed,
                    verification_type=verification_type,
                    reason=reason_out,
                    raw_response=raw,
                    suggested_action=sugg,
                    kb_matched=target_task is not None,
                    image_paths_used=list(image_paths),
                )
            )

        # Precondition path: custom prompt, same client
        content: list[dict[str, Any]] = []
        cond = subtask.precondition
        kb_block = ""
        if target_task:
            # Completion RAG references are not pre-start state. Use ``precondition_reference`` only
            # when present; never use completion reference text for readiness.
            pre_ref = str(target_task.get("precondition_reference") or "").strip()
            if pre_ref:
                kb_block = (
                    "\nOptional KB hint (pre-start workspace only; if it conflicts with Required "
                    f"precondition, follow Required precondition): {pre_ref}\n"
                )
        intro = f"""You are a lab robotics safety verifier (PRE-EXECUTION ONLY).
Do NOT judge whether the full manipulation task is completed.
Only decide whether the scene is READY to start this step.

Subtask instruction: {subtask.natural_language_instruction}
Required precondition (this is the ONLY checklist for READY; ignore verb tense of the instruction): {cond}
Robot state summary: {robot_state_summary or 'unknown'}{kb_block}
**How to decide READY vs NOT_READY**
- **READY** means: from the image alone, the **Required precondition** sentence is satisfied (or clearly satisfied with minor uncertainty that does not block starting). Any optional KB hint above is secondary; if it conflicts with **Required precondition**, follow **Required precondition**.
- **NOT_READY** means: something in **Required precondition** is clearly false in the image, or the main equipment is missing, or a serious obstruction makes the step unsafe.

**Critical:** Do **not** infer required lid/cover state from words like "open" or "close" in the instruction. For example, **open_lid** steps usually require the lid **closed/latched before** the robot opens it; **close_lid** steps require the lid **open** before the robot closes it. Always match the **Required precondition** text to the photo.

You MUST answer in this exact shape (English):

Line 1 — exactly one of:
  VERDICT: READY
  VERDICT: NOT_READY

Lines 2+ — 2 to 5 short sentences that justify line 1. You MUST explicitly state:
- Whether the equipment named in **Required precondition** is visible (or say it is not).
- For each lid/cover claim in **Required precondition**: whether the image supports it (open vs closed vs unclear).
- Any obstruction, foreign object, or hand/person that would block this step.

If the centrifuge (or target equipment) is missing or not visible when the precondition requires it visible, say so clearly (e.g. "No centrifuge is visible in the image").

Do not skip the explanation lines. If you cannot decide, still output VERDICT: NOT_READY and explain uncertainty.
"""
        content.append({"type": "text", "text": intro})
        for p in image_paths:
            b64 = self._fg.encode_image(p)
            if b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

        if not any(c.get("type") == "image_url" for c in content):
            if self.mode == RunMode.DRY_RUN:
                return _emit_vlm_result(
                    VerificationResult(
                        passed=True,
                        verification_type=verification_type,
                        reason="dry_run: images unreadable, simulated pass",
                        raw_response=None,
                        suggested_action=SuggestedAction.CONTINUE,
                        kb_matched=target_task is not None,
                        image_paths_used=list(image_paths),
                    )
                )
            return _emit_vlm_result(
                VerificationResult(
                    passed=False,
                    verification_type=verification_type,
                    reason="could not encode any images",
                    raw_response=None,
                    suggested_action=SuggestedAction.RETRY_VERIFICATION,
                    kb_matched=target_task is not None,
                    image_paths_used=list(image_paths),
                )
            )

        try:
            resp = self._fg.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{"role": "user", "content": content}],
                max_tokens=900,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            logger.exception("VLM precondition call failed: %s", e)
            return _emit_vlm_result(
                VerificationResult(
                    passed=False,
                    verification_type=verification_type,
                    reason=str(e),
                    raw_response=None,
                    suggested_action=SuggestedAction.RETRY_VERIFICATION,
                    kb_matched=target_task is not None,
                    image_paths_used=list(image_paths),
                )
            )

        passed, how = _parse_precondition_verdict(raw)
        reason_out = _build_precondition_reason(how=how, raw=raw)
        if passed:
            reason_out = _vlm_explanation_after_verdict(raw) or how
        sugg = SuggestedAction.CONTINUE if passed else SuggestedAction.REORDER
        if how.startswith("unclear"):
            sugg = SuggestedAction.RETRY_VERIFICATION
        if logger.isEnabledFor(logging.INFO):
            logger.info("%s | precondition narrative | %s", _AGENT, truncate(reason_out, 500))
        return _emit_vlm_result(
            VerificationResult(
                passed=passed,
                verification_type=verification_type,
                reason=reason_out,
                raw_response=raw,
                suggested_action=sugg,
                kb_matched=target_task is not None,
                image_paths_used=list(image_paths),
            )
        )
