"""Guiding Decision Agent: central closed-loop scheduler."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from bioprovla_agent.agent_io_log import log_block, truncate
from bioprovla_agent.llm_protocol_agent import TailoredLLMProtocolAgent
from bioprovla_agent.recovery_policy import (
    human_confirm,
    maybe_reorder_for_precondition,
    wait_for_environment_reset_confirm,
)
from bioprovla_agent.reporting import write_markdown_report
from bioprovla_agent.schemas import (
    ExecutionConfig,
    KnowledgeBaseConfig,
    ModelConfig,
    RobotConfig,
    RunMode,
    RunReport,
    SubTask,
    SubTaskRecord,
    SuggestedAction,
    TaskPlan,
    VerificationType,
)
from bioprovla_agent.validation import validate_real_prerequisites
from bioprovla_agent.vla_embodied_agent import VLAEmbodiedAgent
from bioprovla_agent.vlm_rag_agent import VLMRAGVerificationAgent

logger = logging.getLogger(__name__)

_GGUID = "Guiding Decision Agent"


class GuidingDecisionAgent:
    """Orchestrates LLM -> precondition -> VLA -> completion with recovery."""

    def __init__(
        self,
        protocol_text: str,
        model_config: ModelConfig,
        robot_config: RobotConfig,
        execution_config: ExecutionConfig,
        knowledge_base_config: KnowledgeBaseConfig,
    ) -> None:
        self.protocol_text = protocol_text
        self.model_config = model_config
        self.robot_config = robot_config
        self.execution_config = execution_config
        self.knowledge_base_config = knowledge_base_config

        self._run_dir: Path | None = None
        self._llm = TailoredLLMProtocolAgent(execution_config.mode)
        self._vlm = VLMRAGVerificationAgent(execution_config.mode, knowledge_base_config)
        self._vla = VLAEmbodiedAgent(execution_config.mode, robot_config, execution_config)

    def _prepare_run_dir(self) -> Path:
        if self.execution_config.run_dir:
            rd = Path(self.execution_config.run_dir).expanduser().resolve()
        else:
            rd = Path.cwd() / "bioprovla_runs" / time.strftime("%Y%m%d_%H%M%S")
        try:
            rd.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Cannot create run directory {rd}: {e}") from e
        self._run_dir = rd
        return rd

    def _safe_write_reports(self, report: RunReport, run_dir: Path) -> None:
        """Persist Markdown + JSON; log failures without raising."""
        try:
            write_markdown_report(report, run_dir / "report.md")
        except OSError as e:
            logger.error("Failed to write report.md: %s", e)
        try:
            payload = __import__("json").dumps(report.to_dict(), indent=2)
            (run_dir / "report.json").write_text(payload, encoding="utf-8")
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to write report.json: %s", e)

    def _pre_images(self, step: SubTaskRecord) -> list[str]:
        if self.execution_config.mode == RunMode.MOCK:
            return []
        if self._run_dir is None:
            return []
        out = self._run_dir / "pre" / f"step_{step.subtask.step_id}.png"
        return self._vla.capture_scene_images(out)

    def run(self) -> RunReport:
        t_all = time.perf_counter()
        reorder_log: list[str] = []
        human_log: list[str] = []
        failures: list[str] = []

        try:
            run_dir = self._prepare_run_dir()
        except RuntimeError as e:
            logger.error("%s", e)
            failures.append(str(e))
            return RunReport(
                protocol_text=self.protocol_text,
                task_plan=None,
                subtask_records=[],
                reorder_log=reorder_log,
                human_interventions=human_log,
                failure_summaries=failures,
                total_duration_s=time.perf_counter() - t_all,
                overall_success=False,
                run_directory=None,
            )

        if not self.protocol_text.strip():
            msg = "protocol_text is empty or whitespace-only"
            failures.append(msg)
            log_block(f"{_GGUID} | OUTPUT (aborted)", [("reason", msg), ("failures", failures)])
            rep = RunReport(
                protocol_text=self.protocol_text,
                task_plan=None,
                subtask_records=[],
                reorder_log=reorder_log,
                human_interventions=human_log,
                failure_summaries=failures,
                total_duration_s=time.perf_counter() - t_all,
                overall_success=False,
                run_directory=str(run_dir),
            )
            self._safe_write_reports(rep, run_dir)
            return rep

        log_block(
            f"{_GGUID} | INPUT (run)",
            [
                ("execution_mode", self.execution_config.mode.value),
                ("protocol_char_count", len(self.protocol_text)),
                ("protocol_text_preview", truncate(self.protocol_text, 500)),
                ("run_directory", str(run_dir)),
                ("kb_completion_rag_index_path", self.knowledge_base_config.completion_rag_index_path),
                ("kb_mock_kb", self.knowledge_base_config.mock_kb),
                ("kb_rag_completion_enabled", self.knowledge_base_config.rag_completion_enabled),
                ("kb_rag_precondition_enabled", self.knowledge_base_config.rag_precondition_enabled),
                ("kb_precondition_rag_index_path", self.knowledge_base_config.precondition_rag_index_path),
                ("infer_cli_arg_count", len(self.robot_config.infer_cli_args)),
                ("max_parse_retries", self.execution_config.max_parse_retries),
                ("max_vla_retries", self.execution_config.max_vla_retries),
                ("max_precondition_recovery_loops", self.execution_config.max_precondition_recovery_loops),
                ("prompt_human_on_failure", self.execution_config.prompt_human_on_failure),
                ("wait_enter_on_precondition_stall", self.execution_config.wait_enter_on_precondition_stall),
            ],
        )

        preflight = validate_real_prerequisites(
            self.execution_config.mode,
            self.robot_config.infer_cli_args,
            self.protocol_text,
        )
        for msg in preflight:
            failures.append(msg)
        if preflight:
            log_block(
                f"{_GGUID} | OUTPUT (aborted)",
                [("reason", "REAL mode preflight failed"), ("failures", failures)],
            )
            rep = RunReport(
                protocol_text=self.protocol_text,
                task_plan=None,
                subtask_records=[],
                reorder_log=reorder_log,
                human_interventions=human_log,
                failure_summaries=failures,
                total_duration_s=time.perf_counter() - t_all,
                overall_success=False,
                run_directory=str(run_dir),
            )
            self._safe_write_reports(rep, run_dir)
            return rep

        if self.knowledge_base_config.completion_rag_index_path and self._vlm._fg is not None:
            self._vlm._fg.reload_knowledge_base(self.knowledge_base_config.completion_rag_index_path)

        # LLM protocol parse before VLA init so logs show atomic_actions early and we fail fast without loading policy.
        logger.info(
            "%s | Phase LLM: parsing protocol (FastGPT). If keys are unset, this fails in seconds; "
            "configure api_credentials.llm_api_key or FASTGPT_API_KEY.",
            _GGUID,
        )
        plan: TaskPlan | None = None
        parse_err: str | None = None
        for attempt in range(1, self.execution_config.max_parse_retries + 2):
            plan, parse_err = self._llm.parse_to_plan(self.protocol_text)
            if plan is not None:
                break
            logger.warning("parse attempt %s failed: %s", attempt, parse_err)
        if plan is None:
            msg = f"protocol parse failed: {parse_err}"
            failures.append(msg)
            choice = human_confirm(
                msg + " Allow mock empty plan? [never]",
                self.execution_config.prompt_human_on_failure,
            )
            if choice == "proceed" and self.execution_config.mode != RunMode.REAL:
                plan = TaskPlan(
                    reasoning_process={},
                    subtasks=[
                        SubTask(
                            step_id=1,
                            action_type="unknown",
                            target_object="",
                            location_reference="",
                            natural_language_instruction="Human-approved fallback",
                            precondition="ready",
                            postcondition="done",
                        )
                    ],
                    raw_llm_result={},
                )
                human_log.append("human proceed after parse failure (non-real)")
            if plan is None:
                self._vla.shutdown()
                log_block(
                    f"{_GGUID} | OUTPUT (aborted)",
                    [("reason", "protocol parse failed"), ("failures", failures)],
                )
                rep = RunReport(
                    protocol_text=self.protocol_text,
                    task_plan=None,
                    subtask_records=[],
                    reorder_log=reorder_log,
                    human_interventions=human_log,
                    failure_summaries=failures,
                    total_duration_s=time.perf_counter() - t_all,
                    overall_success=False,
                    run_directory=str(run_dir),
                )
                self._safe_write_reports(rep, run_dir)
                return rep

        if not plan.subtasks:
            failures.append("empty subtask list after parse")
            self._vla.shutdown()
            log_block(
                f"{_GGUID} | OUTPUT (aborted)",
                [("reason", "empty subtask list"), ("failures", failures)],
            )
            rep = RunReport(
                protocol_text=self.protocol_text,
                task_plan=plan,
                subtask_records=[],
                reorder_log=reorder_log,
                human_interventions=human_log,
                failure_summaries=failures,
                total_duration_s=time.perf_counter() - t_all,
                overall_success=False,
                run_directory=str(run_dir),
            )
            self._safe_write_reports(rep, run_dir)
            return rep

        logger.info(
            "%s | Phase VLA: protocol parse OK (%s subtask(s)). Next: load robot + policy "
            "(often 30–120s; SmolVLM/HF weight tqdm = still working, not frozen).",
            _GGUID,
            len(plan.subtasks),
        )
        ok_init, err_init = self._vla.initialize()
        if not ok_init:
            failures.append(f"VLA init: {err_init}")
            if self.execution_config.mode == RunMode.REAL:
                log_block(
                    f"{_GGUID} | OUTPUT (aborted)",
                    [("reason", "VLA init failed in REAL mode"), ("error", err_init), ("failures", failures)],
                )
                rep = RunReport(
                    protocol_text=self.protocol_text,
                    task_plan=plan,
                    subtask_records=[],
                    reorder_log=reorder_log,
                    human_interventions=human_log,
                    failure_summaries=failures,
                    total_duration_s=time.perf_counter() - t_all,
                    overall_success=False,
                    run_directory=str(run_dir),
                )
                self._safe_write_reports(rep, run_dir)
                return rep

        records: list[SubTaskRecord] = [SubTaskRecord(subtask=s) for s in plan.subtasks]
        step_ids = [r.subtask.step_id for r in records]
        if len(step_ids) != len(set(step_ids)):
            logger.warning("%s | duplicate step_id in schedule: %s", _GGUID, step_ids)

        reorder_attempts = 0
        pre_recovery_loops = 0
        i = 0

        log_block(
            f"{_GGUID} | SCHEDULE (subtasks)",
            [
                ("subtask_count", len(records)),
                (
                    "steps",
                    [
                        {
                            "step_id": r.subtask.step_id,
                            "action_type": r.subtask.action_type,
                            "instruction": truncate(r.subtask.natural_language_instruction, 100),
                        }
                        for r in records
                    ],
                ),
            ],
        )

        while i < len(records):
            rec = records[i]
            img_dir = run_dir / "exec" / f"step_{rec.subtask.step_id}"

            log_block(
                f"{_GGUID} | SUBTASK (orchestration)",
                [
                    ("queue_index", i),
                    ("step_id", rec.subtask.step_id),
                    ("action_type", rec.subtask.action_type),
                    ("instruction", truncate(rec.subtask.natural_language_instruction, 200)),
                    ("phases", "A: precondition (VLM) -> B: VLA execute -> C: completion (VLM)"),
                ],
            )

            # Step A: precondition
            pre_ok = False
            for _ in range(self.execution_config.max_precondition_verify_retries + 1):
                imgs = self._pre_images(rec)
                if not imgs and self.execution_config.mode == RunMode.MOCK:
                    imgs = []
                pre = self._vlm.verify(
                    rec.subtask,
                    VerificationType.PRECONDITION,
                    imgs,
                    robot_state_summary=None,
                    _execution_cfg=self.execution_config,
                )
                rec.precondition = pre
                rec.precondition_retries += 1
                if pre.passed:
                    pre_ok = True
                    break
                if pre.suggested_action == SuggestedAction.RETRY_VERIFICATION:
                    continue
                break

            if not pre_ok:
                new_plan, rmsg = maybe_reorder_for_precondition(
                    plan, i, reorder_attempts, self.execution_config.max_reorder_attempts
                )
                if new_plan is not None and rmsg:
                    pre_recovery_loops += 1
                    if pre_recovery_loops > self.execution_config.max_precondition_recovery_loops:
                        msg = (
                            f"max_precondition_recovery_loops ({self.execution_config.max_precondition_recovery_loops}) "
                            f"exceeded at step_id={rec.subtask.step_id} (reorder/verify loop)"
                        )
                        failures.append(msg)
                        rec.status = "failed"
                        log_block(f"{_GGUID} | OUTPUT (aborted loop guard)", [("reason", msg)])
                        break
                    reorder_attempts += 1
                    reorder_log.append(rmsg)
                    plan = new_plan
                    records = [SubTaskRecord(subtask=s) for s in plan.subtasks]
                    logger.info("%s | REORDER | %s | new_order=%s", _GGUID, rmsg, [s.step_id for s in plan.subtasks])
                    continue
                stall_reason = rec.precondition.reason if rec.precondition else None
                if (
                    self.execution_config.mode == RunMode.REAL
                    and self.execution_config.wait_enter_on_precondition_stall
                ):
                    wait_for_environment_reset_confirm(
                        step_id=rec.subtask.step_id,
                        precondition_reason=stall_reason,
                    )
                    human_log.append(
                        f"precondition step {rec.subtask.step_id}: "
                        "environment reset acknowledged (Enter), retry VLM from precondition"
                    )
                    reorder_attempts = 0
                    continue
                hc = human_confirm(
                    f"Precondition failed for step {rec.subtask.step_id}",
                    self.execution_config.prompt_human_on_failure,
                )
                human_log.append(f"precondition step {rec.subtask.step_id}: {hc}")
                if hc == "skip":
                    rec.status = "skipped"
                    pre_recovery_loops = 0
                    i += 1
                    continue
                if hc != "proceed":
                    rec.status = "failed"
                    failures.append(f"precondition failed step {rec.subtask.step_id}")
                    break
                pre_ok = True

            if not pre_ok:
                rec.status = "failed"
                failures.append(f"precondition failed step {rec.subtask.step_id}")
                break

            # Step B: VLA (LeRobot ``infer_loop_no_dataset``: keyboard + home)
            exec_ok = False
            ex = None
            for _ in range(self.execution_config.max_vla_retries + 1):
                ex = self._vla.execute(rec.subtask, img_dir)
                rec.execution = ex
                rec.vla_retries += 1
                if ex.success:
                    exec_ok = True
                    break
            if not exec_ok:
                hc = human_confirm(
                    f"VLA failed for step {rec.subtask.step_id}: {ex.error if ex else 'unknown'}",
                    self.execution_config.prompt_human_on_failure,
                )
                human_log.append(f"vla step {rec.subtask.step_id}: {hc}")
                if hc != "proceed":
                    rec.status = "failed"
                    failures.append(f"vla failed step {rec.subtask.step_id}")
                    break
                exec_ok = True

            # Step C: completion (VLM) — if fail, redo same subtask VLA then re-verify
            comp_imgs = list(ex.image_paths) if ex and ex.image_paths else []
            comp_ok = False
            for _ in range(self.execution_config.max_completion_verify_retries + 1):
                comp = self._vlm.verify(
                    rec.subtask,
                    VerificationType.COMPLETION,
                    comp_imgs,
                    robot_state_summary=None,
                    _execution_cfg=self.execution_config,
                )
                rec.completion = comp
                rec.completion_retries += 1
                if comp.passed:
                    comp_ok = True
                    break
                if comp.suggested_action == SuggestedAction.RETRY_VERIFICATION:
                    continue
                break

            if not comp_ok:
                redo = False
                for _ in range(self.execution_config.max_vla_retries):
                    ex2 = self._vla.execute(rec.subtask, img_dir)
                    rec.execution = ex2
                    rec.vla_retries += 1
                    comp2 = self._vlm.verify(
                        rec.subtask,
                        VerificationType.COMPLETION,
                        list(ex2.image_paths),
                        robot_state_summary=None,
                        _execution_cfg=self.execution_config,
                    )
                    rec.completion = comp2
                    rec.completion_retries += 1
                    if comp2.passed:
                        comp_ok = True
                        redo = True
                        break
                if not redo:
                    hc = human_confirm(
                        f"Completion failed for step {rec.subtask.step_id}",
                        self.execution_config.prompt_human_on_failure,
                    )
                    human_log.append(f"completion step {rec.subtask.step_id}: {hc}")
                    if hc != "proceed":
                        rec.status = "failed"
                        failures.append(f"completion failed step {rec.subtask.step_id}")
                        break
                    comp_ok = True

            if comp_ok:
                rec.status = "completed"
                pre_recovery_loops = 0
                i += 1
            else:
                failures.append(f"stuck: step {rec.subtask.step_id} not completed")
                break

        self._vla.shutdown()

        report = RunReport(
            protocol_text=self.protocol_text,
            task_plan=plan,
            subtask_records=records,
            reorder_log=reorder_log,
            human_interventions=human_log,
            failure_summaries=failures,
            total_duration_s=time.perf_counter() - t_all,
            overall_success=len(failures) == 0 and all(r.status == "completed" for r in records),
            run_directory=str(run_dir),
        )
        log_block(
            f"{_GGUID} | OUTPUT (run complete)",
            [
                ("overall_success", report.overall_success),
                ("total_duration_s", round(report.total_duration_s, 4)),
                ("subtask_statuses", [(r.subtask.step_id, r.status) for r in records]),
                ("failure_summaries", failures or None),
                ("reorder_log", reorder_log or None),
                ("human_interventions", human_log or None),
                ("report_json", str(run_dir / "report.json")),
            ],
        )
        self._safe_write_reports(report, run_dir)
        return report
