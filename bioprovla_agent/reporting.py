"""Markdown + JSON report helpers."""

from __future__ import annotations

from pathlib import Path

from bioprovla_agent.schemas import RunReport


def write_markdown_report(report: RunReport, path: Path) -> None:
    """Write a human-readable Markdown summary."""
    lines: list[str] = []
    lines.append("# BioProVLA-Agent Run Report\n")
    lines.append(f"- **Overall success:** {report.overall_success}")
    lines.append(f"- **Total duration (s):** {report.total_duration_s:.2f}")
    lines.append(f"- **Run directory:** {report.run_directory}\n")

    lines.append("## Original protocol\n")
    lines.append(report.protocol_text.strip() + "\n")

    if report.task_plan:
        lines.append("## Parsed subtasks\n")
        for s in report.task_plan.subtasks:
            lines.append(f"### Step {s.step_id} ({s.action_type})\n")
            lines.append(f"- Instruction: {s.natural_language_instruction}")
            lines.append(f"- Precondition: {s.precondition}")
            lines.append(f"- Postcondition: {s.postcondition}\n")

    lines.append("## Per-subtask results\n")
    for r in report.subtask_records:
        lines.append(f"### Step {r.subtask.step_id} — **{r.status}**\n")
        if r.precondition:
            lines.append(f"- Precondition passed: {r.precondition.passed} ({r.precondition.reason})")
        if r.execution:
            lines.append(
                f"- Execution: success={r.execution.success}, reason={r.execution.finish_reason}, "
                f"duration_s={r.execution.duration_s:.2f}"
            )
            if r.execution.image_paths:
                lines.append(f"- Images: {', '.join(r.execution.image_paths)}")
        if r.completion:
            lines.append(f"- Completion passed: {r.completion.passed} ({r.completion.reason})")
        lines.append(
            f"- Retries: pre={r.precondition_retries}, vla={r.vla_retries}, completion={r.completion_retries}\n"
        )

    if report.reorder_log:
        lines.append("## Reorder log\n")
        for e in report.reorder_log:
            lines.append(f"- {e}")
        lines.append("")

    if report.human_interventions:
        lines.append("## Human interventions\n")
        for e in report.human_interventions:
            lines.append(f"- {e}")
        lines.append("")

    if report.failure_summaries:
        lines.append("## Failures\n")
        for e in report.failure_summaries:
            lines.append(f"- {e}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
