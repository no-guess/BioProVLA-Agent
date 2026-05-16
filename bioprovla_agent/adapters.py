"""Adapters: bundled LLM/VLM integrations and LeRobot infer script."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger(__name__)


def repo_root() -> Path:
    """Workspace root: parent of the bioprovla_agent package."""
    return Path(__file__).resolve().parent.parent


def load_llm_api_module() -> ModuleType:
    """Load canonical LLM protocol parser (bioprovla_agent.integrations.llm_protocol_parser)."""
    from bioprovla_agent.integrations import llm_protocol_parser

    return llm_protocol_parser


def load_fastgpt_vlm_module() -> ModuleType:
    """Load canonical VLM-RAG backend (bioprovla_agent.integrations.vlm_rag_backend)."""
    from bioprovla_agent.integrations import vlm_rag_backend

    return vlm_rag_backend


def get_parse_protocol() -> Callable[[str], dict[str, Any]]:
    mod = load_llm_api_module()
    return mod.parse_protocol  # type: ignore[no-any-return]


def ensure_lerobot_src_on_path() -> Path:
    """Return LeRobot src root and prepend it to sys.path if needed."""
    src = repo_root() / "lerobot-main" / "src"
    s = str(src)
    if src.is_dir() and s not in sys.path:
        sys.path.insert(0, s)
    return src


def import_infer_helpers() -> dict[str, Any]:
    """
    Import symbols from ``lerobot.scripts.lerobot_infer_multitask`` for programmatic runs.

    REAL VLA execution uses LeRobot's ``infer_loop_no_dataset`` here (same binary behavior as
    ``lerobot-infer-multitask``). A maintained twin lives in ``bioprovla_agent.policy_infer_loop``
    for diffing / future migration without editing the LeRobot script.
    """
    ensure_lerobot_src_on_path()
    from lerobot.scripts.lerobot_infer_multitask import (  # type: ignore
        InferConfig,
        _extract_state_from_obs,
        infer_loop_no_dataset,
        init_keyboard_listener,
    )
    from lerobot.configs import parser as lerobot_parser  # type: ignore

    return {
        "InferConfig": InferConfig,
        "_extract_state_from_obs": _extract_state_from_obs,
        "infer_loop_no_dataset": infer_loop_no_dataset,
        "init_keyboard_listener": init_keyboard_listener,
        "parser": lerobot_parser,
    }


def build_infer_config_from_cli_args(cli_args: list[str]) -> Any:
    """
    Parse LeRobot-style CLI args into InferConfig.

    LeRobot's ``InferConfig.__post_init__`` reads ``--policy.path=...`` via
    ``lerobot.configs.parser.get_path_arg``, which defaults to ``sys.argv``.
    The official ``lerobot-infer-multitask`` entrypoint also strips ``--policy.*``
    from the argv fragment passed to ``draccus.parse`` (those flags are not
    draccus-native). We mirror both behaviors here so JSON ``infer_cli_args``
    matches shell usage.
    """
    import draccus

    helpers = import_infer_helpers()
    InferConfig = helpers["InferConfig"]
    lerobot_parser = helpers["parser"]

    argv_tail = list(cli_args)
    saved_argv = sys.argv[:]
    prog = saved_argv[0] if saved_argv else "python"
    try:
        sys.argv = [prog, *argv_tail]
        path_fields = InferConfig.__get_path_fields__()
        filtered = lerobot_parser.filter_path_args(path_fields, argv_tail)
        return draccus.parse(config_class=InferConfig, args=filtered)
    finally:
        sys.argv = saved_argv


def _observation_tensor_to_hwc_uint8(arr: Any) -> Any:
    """Convert torch/tensor/numpy camera frame to HWC uint8 RGB, or None if unsupported."""
    import numpy as np

    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    a = np.asarray(arr)
    while a.ndim > 3:
        a = a[0]
    if a.ndim != 3:
        return None
    # CHW (3, H, W) — common after LeRobot processors
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] != 3:
        return None
    if np.issubdtype(a.dtype, np.floating):
        mx = float(np.max(a)) if a.size else 0.0
        if mx <= 1.0 + 1e-3:
            a = np.clip(a * 255.0, 0.0, 255.0)
        else:
            a = np.clip(a, 0.0, 255.0)
        a = np.round(a).astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def save_obs_image(obs: dict[str, Any], out_path: Path, image_keys: tuple[str, ...] | None = None) -> bool:
    """
    Save first available camera image from observation dict to PNG.

    After LeRobot ``rename_map``, keys may be ``observation.images.camera1`` etc.
    instead of ``front`` / ``handeye``; we fall back to any ``observation.images.*`` key.
    Supports CHW float tensors from ``robot_observation_processor``.
    """
    preferred: list[str] = (
        list(image_keys)
        if image_keys
        else [
            "observation.images.front",
            "observation.images.handeye",
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.image",
            "observation.images.cam_high",
            "front",
            "handeye",
            "image",
        ]
    )
    seen = set(preferred)
    extras = sorted(
        k
        for k in obs
        if isinstance(k, str) and k.startswith("observation.images.") and k not in seen
    )
    keys = preferred + extras
    for k in keys:
        if k not in obs:
            continue
        try:
            hwc = _observation_tensor_to_hwc_uint8(obs[k])
            if hwc is None:
                continue
            from PIL import Image

            img = Image.fromarray(hwc)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return True
        except Exception as e:
            logger.warning("save_obs_image failed for key %s: %s", k, e)
    sample_keys = [repr(x) for x in list(obs.keys())[:24]]
    logger.warning(
        "save_obs_image: no usable RGB frame (tried %s keys). Observation key sample: %s",
        len(keys),
        sample_keys,
    )
    return False


def snapshot_robot_image(robot: Any, robot_observation_processor: Any, out_path: Path) -> bool:
    """Grab one processed observation and save RGB."""
    try:
        obs = robot.get_observation()
        processed = robot_observation_processor(obs)
        return save_obs_image(processed, out_path)
    except Exception as e:
        logger.warning("snapshot_robot_image failed: %s", e)
        return False


def snapshot_raw_robot_camera(
    robot: Any,
    camera_key: str,
    out_path: Path,
    *,
    max_age_ms: int = 2000,
) -> bool:
    """
    Save one frame from ``robot.cameras[camera_key].read_latest()`` (driver RGB, no policy pipeline).

    OpenCV cameras in LeRobot already convert BGR→RGB before buffering.
    """
    import numpy as np
    from PIL import Image

    try:
        cams = getattr(robot, "cameras", None)
        if not isinstance(cams, dict) or camera_key not in cams:
            logger.warning(
                "snapshot_raw_robot_camera: missing camera key %r (available: %s)",
                camera_key,
                sorted(cams.keys()) if isinstance(cams, dict) else type(cams).__name__,
            )
            return False
        frame = cams[camera_key].read_latest(max_age_ms=max_age_ms)
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            logger.warning(
                "snapshot_raw_robot_camera: unexpected frame shape %s dtype=%s",
                getattr(arr, "shape", None),
                getattr(arr, "dtype", None),
            )
            return False
        if not np.issubdtype(arr.dtype, np.integer):
            hwc = _observation_tensor_to_hwc_uint8(arr)
            if hwc is None:
                return False
            arr = hwc
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_path)
        return True
    except Exception as e:
        logger.warning("snapshot_raw_robot_camera failed: %s", e)
        return False


def wait_brief(seconds: float) -> None:
    time.sleep(max(0.0, float(seconds)))
