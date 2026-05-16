#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run policy inference on a real robot without dataset recording.

This script is designed for fast online testing:
- no frame buffering to dataset
- no episode saving/finalization overhead
- sequential multi-task execution with one policy load
"""

import json
import logging
import time
import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.factory import make_pre_post_processors, make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.scripts.lerobot_record import record_loop


@dataclass
class MultiTaskSequenceConfig:
    # Inline JSON string for tasks.
    # Supports:
    # - ["task_a", "task_b"]  -> one group with ordered tasks
    # - [["g1_t1", "g1_t2"], ["g2_t1"]] -> grouped tasks
    # - {"group1": ["..."], "group2": ["..."]} -> grouped tasks from dict values
    inline_json: str | None = None
    # External JSON file path, same schema as inline_json.
    source_json_path: str | Path | None = None
    # Fallback task when neither inline_json nor source_json_path are provided.
    fallback_single_task: str | None = None
    # Run each group this many times.
    repeat_rounds: int = 1


@dataclass
class InferConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    tasks: MultiTaskSequenceConfig = field(default_factory=MultiTaskSequenceConfig)
    rename_map: dict[str, str] = field(default_factory=dict)
    fps: int = 30
    # Enable a safe default speed cap if robot config doesn't provide one.
    enforce_safe_default_speed: bool = False
    # Max per-step joint target delta in robot native units (degrees when use_degrees=True).
    safe_max_relative_target: float = 5.0
    task_time_s: int | float = 120
    min_task_time_s: float = 8.0
    task_check_interval_s: float = 1.0
    max_consecutive_read_failures: int = 10
    reset_time_s: int | float = 0
    # Subtask auto-advance when robot returns near initial joint state.
    home_threshold: float = 4.0
    home_stable_time_s: float = 0.5
    max_wait_home_s: float = 60.0
    # Between combo rounds, require right-arrow to continue.
    require_right_key_for_next_combo: bool = True
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        # Keep CLI behavior consistent with lerobot-record.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("A policy is required. Please set --policy.path=...")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@dataclass
class _InMemoryDatasetMeta:
    features: dict[str, Any]
    stats: dict[str, Any]


@dataclass
class _NoSaveDataset:
    fps: int
    features: dict[str, Any]

    def add_frame(self, frame: dict[str, Any]) -> None:
        # Keep record_loop execution path unchanged while skipping all persistence.
        del frame


def _extract_state_from_obs(obs: dict[str, Any]) -> np.ndarray | None:
    motor_keys = sorted(
        key for key, value in obs.items() if key.endswith(".pos") and isinstance(value, (int, float))
    )
    if not motor_keys:
        return None
    return np.array([float(obs[k]) for k in motor_keys], dtype=np.float32)


def _wait_for_home_or_manual_switch(
    robot: Robot,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    events: dict,
    home_state: np.ndarray | None,
    fps: int,
    home_threshold: float,
    home_stable_time_s: float,
    max_wait_home_s: float,
) -> str:
    """
    Wait between subtasks.
    Returns one of: "auto_next", "manual_next", "redo_current", "stop".
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
            state = _extract_state_from_obs(obs_processed)
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


def _parse_task_payload(payload: Any) -> list[list[str]]:
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return [payload]
        if all(isinstance(item, list) for item in payload):
            parsed = []
            for group in payload:
                if not all(isinstance(task, str) for task in group):
                    raise ValueError("Each task group must contain only task strings.")
                parsed.append(group)
            return parsed
    if isinstance(payload, dict):
        parsed = []
        for _, value in payload.items():
            if isinstance(value, list) and all(isinstance(task, str) for task in value):
                parsed.append(value)
            else:
                raise ValueError("Dict payload values must be list[str].")
        return parsed
    raise ValueError("Task payload must be list[str], list[list[str]], or dict[str, list[str]].")


def resolve_task_groups(tasks_cfg: MultiTaskSequenceConfig) -> list[list[str]]:
    def _decode_tasks_payload(payload_str: str) -> Any:
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError:
            # Accept Python-literal style payloads (single quotes) for CLI convenience.
            return ast.literal_eval(payload_str)

    if tasks_cfg.inline_json:
        payload = _decode_tasks_payload(tasks_cfg.inline_json)
        groups = _parse_task_payload(payload)
    elif tasks_cfg.source_json_path:
        task_path = Path(tasks_cfg.source_json_path).expanduser().resolve()
        payload = _decode_tasks_payload(task_path.read_text(encoding="utf-8"))
        groups = _parse_task_payload(payload)
    elif tasks_cfg.fallback_single_task:
        groups = [[tasks_cfg.fallback_single_task]]
    else:
        raise ValueError(
            "No tasks provided. Set one of: --tasks.inline_json, --tasks.source_json_path, "
            "--tasks.fallback_single_task."
        )

    non_empty_groups = [group for group in groups if group]
    if not non_empty_groups:
        raise ValueError("Task groups are empty.")
    return non_empty_groups


def infer_loop_no_dataset(
    robot: Robot,
    events: dict,
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
    dataset_proxy: _NoSaveDataset,
    display_data: bool,
    display_compressed_images: bool,
) -> dict[str, Any]:
    del features, task_check_interval_s, rename_map

    del teleop_action_processor  # policy-only path
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
            logging.warning(
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

        # No dataset persistence in this script.
        # Keep execution aligned with record loop inference path only.

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=act_processed_policy,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - loop_t
        elapsed_s = time.perf_counter() - start_t

        if home_state is not None and elapsed_s >= effective_min_task_time_s:
            state = _extract_state_from_obs(obs_processed)
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


@parser.wrap()
def infer(cfg: InferConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="inference_multitask", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    task_groups = resolve_task_groups(cfg.tasks)
    combo_repeat_count = max(1, cfg.tasks.repeat_rounds)
    logging.info(
        "Resolved %s task groups, repeat combos %s times.",
        len(task_groups),
        combo_repeat_count,
    )

    if (
        cfg.enforce_safe_default_speed
        and hasattr(cfg.robot, "max_relative_target")
        and getattr(cfg.robot, "max_relative_target") is None
    ):
        setattr(cfg.robot, "max_relative_target", cfg.safe_max_relative_target)
        logging.info(
            "Applied default speed cap: robot.max_relative_target=%.3f "
            "(override with --robot.max_relative_target=... or disable with --enforce_safe_default_speed=false)",
            cfg.safe_max_relative_target,
        )

    robot = make_robot_from_config(cfg.robot)
    listener = None
    try:
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )
        ds_meta = _InMemoryDatasetMeta(features=features, stats={})
        policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
        logging.info(
            "Control rate: %s Hz | max_relative_target=%s",
            cfg.fps,
            getattr(cfg.robot, "max_relative_target", None),
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        robot.connect()
        listener, events = init_keyboard_listener()
        home_state = None
        for attempt in range(3):
            try:
                home_state = _extract_state_from_obs(robot_observation_processor(robot.get_observation()))
                break
            except Exception as e:
                logging.warning("Initial home-state read failed (attempt %s/3): %s", attempt + 1, e)
                time.sleep(0.2)
        if home_state is None:
            logging.warning("Auto-home detection disabled for this run due to robot read failure at startup.")
        else:
            logging.info("Home-state auto advance enabled: True")

        dataset_proxy = _NoSaveDataset(fps=cfg.fps, features=features)

        for combo_idx in range(1, combo_repeat_count + 1):
            logging.info("Starting combo round %s/%s", combo_idx, combo_repeat_count)
            rerun_current_combo = True
            while rerun_current_combo:
                rerun_current_combo = False
                for group_idx, group in enumerate(task_groups, start=1):
                    log_say(
                        f"Start task group {group_idx}/{len(task_groups)} in combo {combo_idx}/{combo_repeat_count}",
                        cfg.play_sounds,
                    )
                    task_idx = 1
                    while task_idx <= len(group):
                        task = group[task_idx - 1]
                        if events["stop_recording"]:
                            break
                        logging.info(
                            "Combo %s/%s | group %s/%s | task %s/%s: %s",
                            combo_idx,
                            combo_repeat_count,
                            group_idx,
                            len(task_groups),
                            task_idx,
                            len(group),
                            task,
                        )
                        log_say(f"Task {task_idx}: {task}", cfg.play_sounds)
                        events["exit_early"] = False
                        result = infer_loop_no_dataset(
                            robot=robot,
                            events=events,
                            fps=cfg.fps,
                            task=task,
                            features=features,
                            rename_map=cfg.rename_map,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            control_time_s=cfg.task_time_s,
                        max_consecutive_read_failures=cfg.max_consecutive_read_failures,
                            min_task_time_s=cfg.min_task_time_s,
                            task_check_interval_s=cfg.task_check_interval_s,
                            home_state=home_state,
                            home_threshold=cfg.home_threshold,
                            home_stable_time_s=cfg.home_stable_time_s,
                            dataset_proxy=dataset_proxy,
                            display_data=cfg.display_data,
                            display_compressed_images=display_compressed_images,
                        )
                        logging.info(
                            "Task finished: reason=%s elapsed=%.2fs",
                            result["finish_reason"],
                            result["elapsed_s"],
                        )
                        if result["finish_reason"] == "manual_stop":
                            # Left arrow (rerecord_episode) means redo current task;
                            # right arrow means proceed to next task.
                            if events["rerecord_episode"]:
                                events["rerecord_episode"] = False
                                logging.info("Left-arrow detected: redo current subtask.")
                                continue
                            logging.info("Right-arrow detected during task: switch to next subtask.")
                            task_idx += 1
                            continue
                        if cfg.reset_time_s > 0 and task_idx < len(group):
                            log_say("Reset environment", cfg.play_sounds)
                            time.sleep(cfg.reset_time_s)
                        task_idx += 1
                    if events["stop_recording"]:
                        break
                    logging.info(
                        "Completed task group %s/%s for combo %s/%s",
                        group_idx,
                        len(task_groups),
                        combo_idx,
                        combo_repeat_count,
                    )
                if events["stop_recording"]:
                    break
                if rerun_current_combo:
                    break
                log_say(f"Combo round {combo_idx}/{combo_repeat_count} completed", cfg.play_sounds)
                if cfg.require_right_key_for_next_combo and combo_idx < combo_repeat_count:
                    log_say(
                        "Press RIGHT arrow to start next combo round, or LEFT arrow to rerun this combo.",
                        cfg.play_sounds,
                    )
                    while True:
                        if events["stop_recording"]:
                            break
                        if events["rerecord_episode"]:
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            rerun_current_combo = True
                            break
                        if events["exit_early"]:
                            events["exit_early"] = False
                            break
                        precise_sleep(0.05)
                    if events["stop_recording"]:
                        break
                    if rerun_current_combo:
                        log_say(
                            f"Rerun combo round {combo_idx}/{combo_repeat_count}",
                            cfg.play_sounds,
                        )
                        continue
            if events["stop_recording"]:
                break
    finally:
        log_say("Stop inference", cfg.play_sounds, blocking=True)
        if robot.is_connected:
            try:
                robot.disconnect()
            except Exception as e:
                logging.warning("Robot disconnect failed during cleanup (ignored): %s", e)
        if not is_headless() and listener:
            listener.stop()
        log_say("Exiting", cfg.play_sounds)


def main():
    register_third_party_plugins()
    infer()


if __name__ == "__main__":
    main()
