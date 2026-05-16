"""VLA Embodied Agent: LeRobot policy execution via ``lerobot_infer_multitask.infer_loop_no_dataset``."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from bioprovla_agent import adapters
from bioprovla_agent.agent_io_log import log_block, truncate
from bioprovla_agent.schemas import ExecutionConfig, ExecutionResult, RobotConfig, RunMode, SubTask

logger = logging.getLogger(__name__)

_AGENT = "VLA Embodied Agent"


def _notify_operator_vla_start(subtask: SubTask) -> None:
    """Print and log which subtask the real robot arm is about to run (VLA policy loop)."""
    inst = (subtask.natural_language_instruction or "").strip()
    banner = (
        "\n"
        + "=" * 72
        + "\n"
        + "ROBOT ARM (VLA) IS NOW EXECUTING THIS SUBTASK\n"
        + f"  step_id:       {subtask.step_id}\n"
        + f"  action_type:   {subtask.action_type}\n"
        + f"  instruction: {inst}\n"
        + "=" * 72
        + "\n"
    )
    logger.info(
        "%s | ARM EXECUTING | step_id=%s | action_type=%s | instruction=%s",
        _AGENT,
        subtask.step_id,
        subtask.action_type,
        truncate(inst, 400),
    )
    print(banner, flush=True)


class VLAEmbodiedAgent:
    """
    Loads robot + policy once (real mode). Executes one language instruction per call.

    The control loop is LeRobot's ``infer_loop_no_dataset`` (keyboard + auto-home), same as
    ``lerobot-infer-multitask``. A twin implementation exists in ``policy_infer_loop`` for reference.
    Semantic **completion** (next vs repeat subtask) is decided by ``GuidingDecisionAgent`` via VLM.
    """

    def __init__(self, mode: RunMode, robot_cfg: RobotConfig, execution_cfg: ExecutionConfig) -> None:
        self.mode = mode
        self.robot_cfg = robot_cfg
        self.execution_cfg = execution_cfg
        self._session: dict[str, Any] | None = None

    def initialize(self) -> tuple[bool, str | None]:
        """Prepare LeRobot session for REAL mode."""
        log_block(
            f"{_AGENT} | INPUT (initialize)",
            [
                ("run_mode", self.mode.value),
                ("infer_cli_arg_count", len(self.robot_cfg.infer_cli_args)),
                ("infer_cli_args", list(self.robot_cfg.infer_cli_args)),
                ("vla_task_time_s", self.execution_cfg.vla_task_time_s),
                ("vla_fps", self.execution_cfg.vla_fps),
                ("vlm_scene_image_source", getattr(self.execution_cfg, "vlm_scene_image_source", "processed")),
                ("vlm_raw_camera_name", getattr(self.execution_cfg, "vlm_raw_camera_name", "front")),
            ],
        )
        if self.mode != RunMode.REAL:
            self._session = {}
            log_block(
                f"{_AGENT} | OUTPUT (initialize)",
                [("success", True), ("note", "non-REAL mode: no robot connection")],
            )
            return True, None
        if not self.robot_cfg.infer_cli_args:
            log_block(
                f"{_AGENT} | OUTPUT (initialize)",
                [("success", False), ("error", "robot_config.infer_cli_args is empty for REAL mode")],
            )
            return False, "robot_config.infer_cli_args is empty for REAL mode"
        try:
            adapters.ensure_lerobot_src_on_path()
            from lerobot.datasets.feature_utils import combine_feature_dicts  # type: ignore
            from lerobot.datasets.pipeline_features import (  # type: ignore
                aggregate_pipeline_dataset_features,
                create_initial_features,
            )
            from lerobot.policies.factory import make_policy, make_pre_post_processors  # type: ignore
            from lerobot.processor import make_default_processors  # type: ignore
            from lerobot.robots import make_robot_from_config  # type: ignore
            from lerobot.scripts.lerobot_infer_multitask import (  # type: ignore
                _InMemoryDatasetMeta,
                _NoSaveDataset,
            )
            from lerobot.utils.import_utils import register_third_party_plugins  # type: ignore
            from lerobot.utils.utils import init_logging  # type: ignore

            from lerobot.processor.rename_processor import rename_stats  # type: ignore

            register_third_party_plugins()
            init_logging()

            cfg = adapters.build_infer_config_from_cli_args(self.robot_cfg.infer_cli_args)
            logger.info("%s | InferConfig parsed from infer_cli_args.", _AGENT)

            if cfg.display_data:
                from lerobot.utils.visualization_utils import init_rerun  # type: ignore

                logger.info(
                    "%s | display_data=true: starting Rerun (extra startup; wgpu messages are often harmless).",
                    _AGENT,
                )
                init_rerun(
                    session_name="bioprovla_infer",
                    ip=cfg.display_ip,
                    port=cfg.display_port,
                )
            display_compressed_images = (
                True
                if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
                else bool(cfg.display_compressed_images)
            )

            logger.info("%s | Building robot driver (cameras/serial not opened until connect).", _AGENT)
            robot = make_robot_from_config(cfg.robot)
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
            logger.info(
                "%s | Loading policy weights (VLM backbone + policy head). Expect tens of seconds; "
                "HF/PyTorch 'Loading weights' tqdm means progress, not an error.",
                _AGENT,
            )
            policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
            logger.info("%s | Policy object ready; building preprocessor/postprocessor.", _AGENT)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.rename_map},
                },
            )

            logger.info("%s | Connecting robot (USB/cameras). Hangs here if port or camera index is wrong.", _AGENT)
            robot.connect()
            from lerobot.scripts.lerobot_infer_multitask import init_keyboard_listener  # type: ignore

            listener, events = init_keyboard_listener()

            _extract_state = adapters.import_infer_helpers()["_extract_state_from_obs"]
            home_state = None
            for attempt in range(3):
                try:
                    home_state = _extract_state(robot_observation_processor(robot.get_observation()))
                    break
                except Exception as e:
                    logger.warning("home-state read attempt %s failed: %s", attempt + 1, e)
                    time.sleep(0.2)

            dataset_proxy = _NoSaveDataset(fps=cfg.fps, features=features)

            self._session = {
                "cfg": cfg,
                "robot": robot,
                "listener": listener,
                "events": events,
                "teleop_action_processor": teleop_action_processor,
                "robot_action_processor": robot_action_processor,
                "robot_observation_processor": robot_observation_processor,
                "features": features,
                "policy": policy,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "home_state": home_state,
                "dataset_proxy": dataset_proxy,
                "display_data": bool(cfg.display_data),
                "display_compressed_images": display_compressed_images,
            }
            log_block(
                f"{_AGENT} | OUTPUT (initialize)",
                [
                    ("success", True),
                    ("robot_type", getattr(robot, "robot_type", None)),
                    ("fps", getattr(cfg, "fps", None)),
                    ("task_time_s", getattr(cfg, "task_time_s", None)),
                    ("home_state_available", home_state is not None),
                ],
            )
            return True, None
        except Exception as e:
            logger.exception("VLA initialize failed")
            self._session = None
            log_block(
                f"{_AGENT} | OUTPUT (initialize)",
                [("success", False), ("error", str(e))],
            )
            return False, str(e)

    def _snapshot_for_vlm(self, out_file: Path) -> bool:
        """
        One PNG for VLM paths. Matches ``execution_config.vlm_scene_image_source``:

        - ``raw_front``: OpenCV frame from ``robot.cameras`` (no motor read / no rename_map).
        - ``processed``: LeRobot observation processor; if that fails, raw camera fallback
          so completion still gets a frame when the bus is flaky but cameras work.
        """
        if not self._session:
            return False
        robot = self._session["robot"]
        proc = self._session["robot_observation_processor"]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        src = getattr(self.execution_cfg, "vlm_scene_image_source", "processed")
        cam_name = getattr(self.execution_cfg, "vlm_raw_camera_name", "front") or "front"
        if src == "raw_front":
            return bool(adapters.snapshot_raw_robot_camera(robot, cam_name, out_file))
        if adapters.snapshot_robot_image(robot, proc, out_file):
            return True
        if adapters.snapshot_raw_robot_camera(robot, cam_name, out_file):
            logger.info("%s | snapshot | processed failed; saved raw camera %r", _AGENT, cam_name)
            return True
        return False

    def capture_scene_images(self, out_file: Path) -> list[str]:
        """Save one current camera frame for precondition checks (REAL mode only)."""
        if self.mode != RunMode.REAL or not self._session:
            logger.info("%s | capture_scene_images | skipped (mode=%s session=%s)", _AGENT, self.mode.value, bool(self._session))
            return []
        src = getattr(self.execution_cfg, "vlm_scene_image_source", "processed")
        cam_name = getattr(self.execution_cfg, "vlm_raw_camera_name", "front") or "front"
        if self._snapshot_for_vlm(out_file):
            logger.info(
                "%s | capture_scene_images | saved %s (%s)",
                _AGENT,
                out_file,
                f"raw camera {cam_name!r}" if src == "raw_front" else "processed or fallback raw",
            )
            return [str(out_file)]
        logger.warning("%s | capture_scene_images | failed %s", _AGENT, out_file)
        return []

    def shutdown(self) -> None:
        """Disconnect robot and clear session."""
        if self._session is None:
            logger.info("%s | shutdown | no session", _AGENT)
            return
        if self.mode != RunMode.REAL:
            self._session = None
            logger.info("%s | shutdown | non-REAL mode (cleared placeholder session)", _AGENT)
            return
        robot = self._session.get("robot")
        listener = self._session.get("listener")
        try:
            from lerobot.utils.control_utils import is_headless  # type: ignore

            if robot is not None and getattr(robot, "is_connected", False):
                try:
                    robot.disconnect()
                except Exception as e:
                    logger.warning("robot.disconnect failed: %s", e)
            if listener and not is_headless():
                listener.stop()
        finally:
            self._session = None
        log_block(f"{_AGENT} | OUTPUT (shutdown)", [("status", "REAL session released")])

    def execute(
        self,
        subtask: SubTask,
        run_images_dir: Path | None,
    ) -> ExecutionResult:
        """Run VLA for a single instruction."""
        t0 = time.perf_counter()
        log_block(
            f"{_AGENT} | INPUT (execute)",
            [
                ("run_mode", self.mode.value),
                ("step_id", subtask.step_id),
                ("action_type", subtask.action_type),
                ("natural_language_instruction", truncate(subtask.natural_language_instruction, 240)),
                ("run_images_dir", str(run_images_dir) if run_images_dir else None),
                ("save_images", self.execution_cfg.save_images),
            ],
        )
        if self.mode in (RunMode.MOCK, RunMode.DRY_RUN):
            logger.info(
                "%s | %s execute (no physical arm motion) | step_id=%s | action_type=%s | %s",
                _AGENT,
                self.mode.value,
                subtask.step_id,
                subtask.action_type,
                truncate(subtask.natural_language_instruction, 240),
            )
            print(
                f"[{self.mode.value.upper()}] Simulated VLA step_id={subtask.step_id} "
                f"action={subtask.action_type}: {truncate(subtask.natural_language_instruction, 120)}",
                flush=True,
            )
            paths: list[str] = []
            if run_images_dir is not None and self.execution_cfg.save_images:
                run_images_dir.mkdir(parents=True, exist_ok=True)
                dummy = run_images_dir / f"mock_step_{subtask.step_id}.txt"
                dummy.write_text("mock image placeholder\n", encoding="utf-8")
                paths.append(str(dummy))
            out = ExecutionResult(
                success=True,
                duration_s=time.perf_counter() - t0,
                finish_reason="mock",
                image_paths=paths,
                robot_state_before=None,
                robot_state_after=None,
                error=None,
            )
            log_block(
                f"{_AGENT} | OUTPUT (execute)",
                [
                    ("success", out.success),
                    ("finish_reason", out.finish_reason),
                    ("duration_s", round(out.duration_s, 4)),
                    ("image_paths", out.image_paths),
                    ("error", out.error),
                ],
            )
            return out

        if not self._session:
            out = ExecutionResult(
                success=False,
                duration_s=time.perf_counter() - t0,
                finish_reason="not_initialized",
                image_paths=[],
                robot_state_before=None,
                robot_state_after=None,
                error="VLA session not initialized",
            )
            log_block(
                f"{_AGENT} | OUTPUT (execute)",
                [
                    ("success", out.success),
                    ("finish_reason", out.finish_reason),
                    ("duration_s", round(out.duration_s, 4)),
                    ("error", out.error),
                ],
            )
            return out

        s = self._session
        cfg = s["cfg"]
        robot = s["robot"]
        events = s["events"]
        robot_observation_processor = s["robot_observation_processor"]
        # After-execution snapshot only is returned for completion VLM; pre PNG may still be saved for debugging.
        image_paths: list[str] = []

        if run_images_dir is not None and self.execution_cfg.save_images:
            run_images_dir.mkdir(parents=True, exist_ok=True)
            pre_path = run_images_dir / f"step_{subtask.step_id}_pre.png"
            self._snapshot_for_vlm(pre_path)

        events["stop_recording"] = False
        events["exit_early"] = False
        events["rerecord_episode"] = False

        try:
            from lerobot.scripts.lerobot_infer_multitask import infer_loop_no_dataset  # type: ignore

            _notify_operator_vla_start(subtask)

            ctrl_s = max(1.0, float(self.execution_cfg.vla_task_time_s or cfg.task_time_s))
            fps_use = max(1, int(self.execution_cfg.vla_fps or cfg.fps))
            result = infer_loop_no_dataset(
                robot=robot,
                events=events,
                fps=fps_use,
                task=subtask.natural_language_instruction,
                features=s["features"],
                rename_map=cfg.rename_map,
                teleop_action_processor=s["teleop_action_processor"],
                robot_action_processor=s["robot_action_processor"],
                robot_observation_processor=robot_observation_processor,
                policy=s["policy"],
                preprocessor=s["preprocessor"],
                postprocessor=s["postprocessor"],
                control_time_s=ctrl_s,
                max_consecutive_read_failures=cfg.max_consecutive_read_failures,
                min_task_time_s=cfg.min_task_time_s,
                task_check_interval_s=cfg.task_check_interval_s,
                home_state=s["home_state"],
                home_threshold=cfg.home_threshold,
                home_stable_time_s=cfg.home_stable_time_s,
                dataset_proxy=s["dataset_proxy"],
                display_data=bool(s.get("display_data", False)),
                display_compressed_images=bool(s.get("display_compressed_images", False)),
            )
        except Exception as e:
            logger.exception("infer_loop_no_dataset failed")
            out = ExecutionResult(
                success=False,
                duration_s=time.perf_counter() - t0,
                finish_reason="exception",
                image_paths=list(image_paths),
                robot_state_before=None,
                robot_state_after=None,
                error=str(e),
            )
            log_block(
                f"{_AGENT} | OUTPUT (execute)",
                [
                    ("success", out.success),
                    ("finish_reason", out.finish_reason),
                    ("duration_s", round(out.duration_s, 4)),
                    ("image_paths", out.image_paths),
                    ("error", out.error),
                ],
            )
            return out

        if run_images_dir is not None and self.execution_cfg.save_images:
            post_path = run_images_dir / f"step_{subtask.step_id}_post.png"
            if self._snapshot_for_vlm(post_path):
                image_paths = [str(post_path)]

        fr = str(result.get("finish_reason", "unknown"))
        ok = fr not in ("exception",)
        out = ExecutionResult(
            success=ok,
            duration_s=time.perf_counter() - t0,
            finish_reason=fr,
            image_paths=image_paths,
            robot_state_before=None,
            robot_state_after=None,
            error=None if ok else fr,
        )
        log_block(
            f"{_AGENT} | OUTPUT (execute)",
            [
                ("success", out.success),
                ("finish_reason", out.finish_reason),
                ("duration_s", round(out.duration_s, 4)),
                ("image_paths", out.image_paths),
                ("error", out.error),
            ],
        )
        return out
