#!/usr/bin/env python3
"""CLI entry for BioProVLA-Agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bioprovla_agent.config import dump_example_config, load_run_config
from bioprovla_agent.guiding_decision_agent import GuidingDecisionAgent
from bioprovla_agent.schemas import RunMode, SubTask, VerificationType
from bioprovla_agent.vlm_rag_agent import VLMRAGVerificationAgent


def _run_completion_image_test(
    cfg: dict,
    image_path: Path,
    *,
    step_id: int,
    action_type: str,
    target_object: str,
    location_reference: str,
    instruction: str,
    precondition: str,
    postcondition: str,
    knowledge_base_id: str | None,
) -> int:
    """Call VLM completion verify with one image; prints JSON to stdout."""
    log = logging.getLogger(__name__)
    img = image_path.expanduser().resolve()
    if not img.is_file():
        log.error("Image not found: %s", img)
        return 2

    execution = cfg["execution_config"]
    kb = cfg["knowledge_base_config"]
    if execution.mode == RunMode.MOCK:
        log.error("completion image test needs execution_config.mode real (or dry_run with images); mock bypasses VLM.")
        return 2

    vlm = VLMRAGVerificationAgent(execution.mode, kb)
    sub = SubTask(
        step_id=step_id,
        action_type=action_type,
        target_object=target_object,
        location_reference=location_reference,
        natural_language_instruction=instruction,
        precondition=precondition,
        postcondition=postcondition,
        knowledge_base_id=knowledge_base_id,
    )
    res = vlm.verify(
        sub,
        VerificationType.COMPLETION,
        [str(img)],
        robot_state_summary=None,
        _execution_cfg=execution,
    )
    payload = {
        "passed": res.passed,
        "kb_matched": res.kb_matched,
        "suggested_action": res.suggested_action.value,
        "reason": res.reason,
        "image_paths_used": res.image_paths_used,
        "raw_response": res.raw_response,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if res.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="BioProVLA-Agent closed-loop runner",
        epilog=(
            "Config is always read from the path you pass to --config. "
            "--write-example-config only creates a template file (any path/name you choose); "
            "then run the same file, e.g. --config my_run.json. "
            "configs/bioprovla_example.json is a ready-made template in the repo; use either one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        help="Path to JSON run config (protocol_text, mode, robot, KB, retries, ...)",
    )
    p.add_argument(
        "--write-example-config",
        type=str,
        metavar="OUT_PATH",
        help="Write a starter JSON to OUT_PATH and exit (does not run the loop)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG (e.g. full VLM raw_response on Tailored/VLM agents)",
    )
    p.add_argument(
        "--test-completion-image",
        metavar="PNG_PATH",
        help=(
            "Skip the full loop: run VLM completion verification on this single image only. "
            "Requires --config (same JSON as a real run: API keys + knowledge_base_config). "
            "Use --test-* overrides below to match the subtask you want to simulate."
        ),
    )
    p.add_argument("--test-step-id", type=int, default=1, help="SubTask.step_id for the one-off test")
    p.add_argument("--test-action-type", default="open_lid", help="SubTask.action_type")
    p.add_argument("--test-target-object", default="Centrifugal engine room cover", help="SubTask.target_object")
    p.add_argument("--test-location-reference", default="lab bench", help="SubTask.location_reference")
    p.add_argument(
        "--test-instruction",
        default="Open the centrifugal engine room cover c",
        help="SubTask.natural_language_instruction",
    )
    p.add_argument(
        "--test-precondition",
        default="Centrifuge visible; engine room cover appears fully closed and latched.",
        help="SubTask.precondition",
    )
    p.add_argument(
        "--test-postcondition",
        default="Centrifuge visible; engine room cover appears open; interior chamber accessible.",
        help="SubTask.postcondition (used as completion criteria even when RAG is enabled)",
    )
    p.add_argument(
        "--test-knowledge-base-id",
        default="",
        help="SubTask.knowledge_base_id for KB lookup (empty = None)",
    )
    args = p.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("bioprovla_agent").setLevel(level)

    if args.write_example_config:
        dump_example_config(args.write_example_config)
        print(f"Wrote example config to {args.write_example_config}")
        return 0

    if not args.config:
        p.error("--config is required unless --write-example-config is used")

    try:
        cfg = load_run_config(args.config)
    except ValueError as e:
        logging.getLogger(__name__).error("%s", e)
        return 2

    if args.test_completion_image:
        kb_id = (args.test_knowledge_base_id or "").strip() or None
        return _run_completion_image_test(
            cfg,
            Path(args.test_completion_image),
            step_id=args.test_step_id,
            action_type=args.test_action_type,
            target_object=args.test_target_object,
            location_reference=args.test_location_reference,
            instruction=args.test_instruction,
            precondition=args.test_precondition,
            postcondition=args.test_postcondition,
            knowledge_base_id=kb_id,
        )

    agent = GuidingDecisionAgent(
        protocol_text=cfg["protocol_text"],
        model_config=cfg["model_config"],
        robot_config=cfg["robot_config"],
        execution_config=cfg["execution_config"],
        knowledge_base_config=cfg["knowledge_base_config"],
    )
    report = agent.run()
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
