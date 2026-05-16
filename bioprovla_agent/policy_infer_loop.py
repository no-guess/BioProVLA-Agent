"""
Twin of LeRobot ``infer_loop_no_dataset`` (keyboard + home detection) for diffing / experiments.

**Production path:** ``VLAEmbodiedAgent`` calls LeRobot's ``infer_loop_no_dataset`` from
``lerobot.scripts.lerobot_infer_multitask`` so behavior matches ``lerobot-infer-multitask`` exactly.
This file is optional; keep it in sync when LeRobot changes the upstream loop.

**Keyboard** (via ``lerobot.utils.control_utils.init_keyboard_listener`` ``events`` dict):

- **Right arrow**: sets ``exit_early`` → loop ends with ``finish_reason="manual_stop"`` (caller may
  treat as skip-to-verification / next subtask signal; see LeRobot multitask script).
- **Left arrow**: sets ``rerecord_episode`` and ``exit_early`` → same early stop; caller checks
  ``rerecord_episode`` to redo the current subtask.
- **Esc**: stop-like early exit.

**Auto home**: After ``min_task_time_s`` (floored to 8s like LeRobot), joint vector from
``*.pos`` keys is compared to ``home_state``; if within ``home_threshold`` for ``home_stable_time_s``
continuously, the loop ends with ``finish_reason="auto_home"``.

**BioProVLA orchestration** (implemented in ``guiding_decision_agent``): After this loop returns,
the guiding agent runs **VLM completion** on saved frames. If completion fails, it **re-runs VLA**
on the **same** subtask; if it passes, it advances to the **next** subtask. Semantic success is not
inferred from ``finish_reason`` alone — use the VLM completion step.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

logger = logging.getLogger(__name__)


def extract_joint_state_from_obs(obs: dict[str, Any]) -> np.ndarray | None:
    """Sorted ``*.pos`` motor keys → float32 vector (same logic as LeRobot multitask infer)."""
    motor_keys = sorted(
        key for key, value in obs.items() if key.endswith(".pos") and isinstance(value, (int, float))
    )
    if not motor_keys:
        return None
    return np.array([float(obs[k]) for k in motor_keys], dtype=np.float32)


def wait_for_home_or_manual_switch(
    robot: Robot,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    events: dict[str, Any],
    home_state: np.ndarray | None,
    fps: int,
    home_threshold: float,
    home_stable_time_s: float,
    max_wait_home_s: float,
) -> str:
    """
    Wait between subtasks (optional helper; mirrors LeRobot ``_wait_for_home_or_manual_switch``).

    Returns one of: ``"auto_next"``, ``"manual_next"``, ``"redo_current"``, ``"stop"``.
    """
    start_t = time.perf_counter()
    stable_t = 0.0
    while (time.perf_counter() - start_t) < max_wait_home_s:
        loop_t = time.perf_counter()

        if events["stop_recording"]:
            return "stop"
        if events["rerecord_episode"]:
            events["rerecord_episode"] = False
            events["exit_early"] = False
            return "redo_current"
        if events["exit_early"]:
            events["exit_early"] = False
            return "manual_next"

        if home_state is not None:
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            state = extract_joint_state_from_obs(obs_processed)
            if state is not None and state.shape == home_state.shape:
                max_err = float(np.max(np.abs(state - home_state)))
                if max_err <= home_threshold:
                    stable_t += time.perf_counter() - loop_t
                    if stable_t >= home_stable_time_s:
                        return "auto_next"
                else:
                    stable_t = 0.0

        precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_t), 0.0))

    return "auto_next"


def infer_policy_loop(
    robot: Robot,
    events: dict[str, Any],
    fps: int,
    task: str,
    features: dict[str, Any],
    rename_map: dict[str, str],
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    control_time_s: int | float,
    max_consecutive_read_failures: int,
    min_task_time_s: float,
    task_check_interval_s: float,
    home_state: np.ndarray | None,
    home_threshold: float,
    home_stable_time_s: float,
    dataset_proxy: Any,
    display_data: bool,
    display_compressed_images: bool,
) -> dict[str, Any]:
    """
    Policy-only inference loop with optional auto-home stop (same contract as LeRobot
    ``infer_loop_no_dataset``). ``dataset_proxy`` must expose ``.features`` for ``build_dataset_frame``.
    """
    del features, task_check_interval_s, rename_map
    del teleop_action_processor

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    effective_min_task_time_s = max(float(min_task_time_s), 8.0)
    stable_home_t = 0.0
    consecutive_read_failures = 0
    finish_reason = "timeout"
    elapsed_s = 0.0
    while elapsed_s < float(control_time_s):
        loop_t = time.perf_counter()

        if events["stop_recording"] or events["exit_early"]:
            events["exit_early"] = False
            finish_reason = "manual_stop"
            break

        try:
            obs = robot.get_observation()
            consecutive_read_failures = 0
        except Exception as e:
            consecutive_read_failures += 1
            logger.warning(
                "Robot observation read failed (%s/%s): %s",
                consecutive_read_failures,
                max_consecutive_read_failures,
                e,
            )
            if consecutive_read_failures >= max_consecutive_read_failures:
                raise ConnectionError(
                    f"Too many consecutive robot read failures ({consecutive_read_failures})."
                ) from e
            precise_sleep(0.05)
            elapsed_s = time.perf_counter() - start_t
            continue
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_proxy.features, obs_processed, prefix=OBS_STR)

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        act_processed_policy: RobotAction = make_robot_action(action_values, dataset_proxy.features)
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=act_processed_policy,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - loop_t
        elapsed_s = time.perf_counter() - start_t

        if home_state is not None and elapsed_s >= effective_min_task_time_s:
            state = extract_joint_state_from_obs(obs_processed)
            if state is not None and state.shape == home_state.shape:
                max_err = float(np.max(np.abs(state - home_state)))
                if max_err <= float(home_threshold):
                    stable_home_t += dt_s
                    if stable_home_t >= float(home_stable_time_s):
                        finish_reason = "auto_home"
                        break
                else:
                    stable_home_t = 0.0

        sleep_time_s = 1 / fps - dt_s
        precise_sleep(max(sleep_time_s, 0.0))

    return {"elapsed_s": elapsed_s, "finish_reason": finish_reason}


# Back-compat alias for code/tests that expect the LeRobot name.
infer_loop_no_dataset = infer_policy_loop
