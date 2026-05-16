"""
VLM + RAG knowledge base for robotic motion verification (canonical implementation).

Canonical VLM-RAG backend for BioProVLA. Default KB index under
Multimodal_detection_system/BioRobo-MVKB/img_index (override with BIOPROVLA_KB_PATH).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Directory containing bioprovla_agent/ (workspace root for this project)."""
    return Path(__file__).resolve().parent.parent.parent


# ==================== Configuration & Initialization ====================
_DEFAULT_KB = _repo_root() / "Multimodal_detection_system" / "BioRobo-MVKB" / "img_index"
KB_FILE_PATH = os.environ.get("BIOPROVLA_KB_PATH", str(_DEFAULT_KB))

DEFAULT_VLM_BASE_URL = "https://cloud.fastgpt.cn/api/v1"
_VLM_API_KEY = os.environ.get("FASTGPT_VLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
_VLM_BASE_URL = os.environ.get("FASTGPT_VLM_BASE_URL", DEFAULT_VLM_BASE_URL)

_openai_client: Any | None = None


def configure_api(api_key: str, base_url: str | None = None) -> None:
    """Set VLM client credentials for this process (used when loading ``api_credentials`` from JSON)."""
    global _VLM_API_KEY, _VLM_BASE_URL, _openai_client
    _VLM_API_KEY = api_key or ""
    _VLM_BASE_URL = (base_url or "").strip() or DEFAULT_VLM_BASE_URL
    _openai_client = None


def _ensure_openai_client() -> Any:
    """Lazily construct OpenAI client so importing this module does not require ``openai``."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "BioProVLA VLM needs the 'openai' package for real/dry_run API calls. "
                "Install with: pip install openai"
            ) from e
        _openai_client = OpenAI(api_key=_VLM_API_KEY or "missing-key", base_url=_VLM_BASE_URL)
    return _openai_client


class _LazyOpenAIClient:
    """Proxy so ``client.chat.completions.create`` works without importing openai at import time."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_ensure_openai_client(), name)


client = _LazyOpenAIClient()


def load_knowledge_base(file_path: str) -> List[Dict]:
    """Load task knowledge base from external JSON file.

    Each task may include success/failure reference images for completion RAG and optional
    ``precondition_reference`` (pre-start workspace hints for precondition checks only).
    Completion criteria are supplied by the LLM-generated ``SubTask.postcondition``.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load knowledge base from %s: %s", file_path, e)
        return []


KNOWLEDGE_BASE: List[Dict] = load_knowledge_base(KB_FILE_PATH)


def reload_knowledge_base(file_path: Optional[str] = None) -> List[Dict]:
    """Reload global KNOWLEDGE_BASE from disk (used by BioProVLA and tests)."""
    global KNOWLEDGE_BASE, KB_FILE_PATH
    path = file_path or KB_FILE_PATH
    KB_FILE_PATH = path
    KNOWLEDGE_BASE = load_knowledge_base(path)
    return KNOWLEDGE_BASE


def encode_image(image_path: str) -> Optional[str]:
    """Encode local image to Base64 string, handle file errors."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning("Image file not found, skipped: %s", image_path)
        return None
    except Exception as e:
        logger.warning("Failed to read image %s: %s", image_path, e)
        return None


def parse_image_paths(paths_str: str) -> List[str]:
    """Parse semicolon-separated paths, filter empty/invalid entries."""
    if not paths_str or paths_str.strip() == "":
        return []
    return [path.strip() for path in paths_str.split(";") if path.strip()]


def find_task_by_description_in(task_desc: str, kb: List[Dict]) -> Optional[Dict]:
    """Find exact matching task in a given knowledge-base list."""
    if not kb or not task_desc.strip():
        return None
    t = task_desc.strip().lower()
    for task in kb:
        if str(task.get("task_description", "")).strip().lower() == t:
            return task
    return None


def find_task_by_keyword_fallback_in(task_desc: str, kb: List[Dict]) -> Optional[Dict]:
    """Best-effort KB match by token overlap within ``kb``."""
    if not kb or not task_desc.strip():
        return None
    query_tokens = set(task_desc.lower().replace(",", " ").split())
    best: Optional[Dict] = None
    best_score = 0
    for task in kb:
        td = str(task.get("task_description", "")).lower().replace(",", " ")
        cand_tokens = set(td.split())
        score = len(query_tokens & cand_tokens)
        if score > best_score:
            best_score = score
            best = task
    return best if best_score > 0 else None


def find_task_by_description(task_desc: str) -> Optional[Dict]:
    """Find exact matching task from the default completion knowledge base."""
    return find_task_by_description_in(task_desc, KNOWLEDGE_BASE)


def find_task_by_keyword_fallback(task_desc: str) -> Optional[Dict]:
    """Best-effort KB match on the default completion knowledge base."""
    return find_task_by_keyword_fallback_in(task_desc, KNOWLEDGE_BASE)


def verify_robotic_arm_motion_with_response(
    task_description: str,
    motion_sequence_image_paths: List[str],
    completion_criteria: str | None = None,
    kb: Optional[List[Dict]] = None,
) -> Optional[str]:
    """
    Verify subtask completion from workspace imagery; returns assistant text or None on failure.

    Args:
        task_description: Matches ``task_description`` in the KB entry.
        motion_sequence_image_paths: After-execution workspace image(s). If more than one path is
            passed, only the **last** is used (BioProVLA completion policy: single current frame).
        completion_criteria: LLM-generated ``SubTask.postcondition``. This is the only textual
            completion checklist; KB entries provide references only and do not need ``prompt``.
        kb: Optional KB list (same schema as ``img_index``). Defaults to global ``KNOWLEDGE_BASE``.
    """
    use_kb = kb if kb is not None else KNOWLEDGE_BASE
    if not use_kb:
        logger.error("Knowledge base is not loaded.")
        return None
    paths = [p for p in motion_sequence_image_paths if p]
    if not paths:
        logger.error("No completion workspace images provided.")
        return None
    if len(paths) > 1:
        logger.warning(
            "verify_robotic_arm_motion_with_response: %s paths supplied; using last path only for completion",
            len(paths),
        )
        paths = paths[-1:]
    target_task = find_task_by_description_in(task_description, use_kb) or find_task_by_keyword_fallback_in(
        task_description, use_kb
    )
    if not target_task:
        logger.error("No matching task found for description: %r", task_description)
        return None
    criteria = (completion_criteria or "").strip()
    if not criteria:
        logger.warning(
            "No LLM completion criteria supplied for %r; falling back to task description.",
            task_description,
        )
        criteria = task_description
    request_content = build_message_content(target_task, paths, criteria)
    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": request_content}],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("VLM API request failed: %s", e)
        return None


def build_message_content(
    task: Dict,
    scene_image_paths: List[str],
    completion_criteria: str,
) -> List[Dict[str, Any]]:
    """
    Build interleaved text-image prompt for **single-frame** completion verification (after subtask).

    ``scene_image_paths`` must contain exactly one path (callers may pass several; upstream keeps the last).
    """
    content: List[Dict[str, Any]] = []
    n = len(scene_image_paths)
    if n != 1:
        logger.warning("build_message_content: expected 1 scene image, got %s", n)

    system_prompt = f"""
    You are a professional biotech laboratory robotics **scene completion** verifier.
    You will receive **exactly one** photograph: the robot workspace **immediately after** the reported
    subtask was attempted (current end state / “what the cameras see now”).

    --- TASK DEFINITION ---
    RAG-Matched Task: {task['task_description']}
    Required Postcondition (from LLM; this is the only completion checklist): {completion_criteria}

    --- EVALUATION RULES (MUST FOLLOW) ---
    1. Base your verdict **only** on this single image and the Required Postcondition.
       Do **not** infer unseen prior motion, and do **not** claim the arm did something that is not
       visible in the image.
    2. If the image **clearly** shows the goal state described in the Required Postcondition (e.g. lid open/closed,
       tube removed as required), output **Task Fully Completed**.
    3. If the goal state is not visible, ambiguous, contradicts the Required Postcondition, or only partially met,
       output **Task Failed**.
    4. Success and Failure reference images below are **style guides only**; your verdict must match
       what you see in the single observed workspace image, not the references alone.
    """
    content.append({"type": "text", "text": system_prompt})

    success_paths = parse_image_paths(task["success_examples"].get("front_camera", ""))
    if success_paths:
        content.append({"type": "text", "text": "--- REFERENCE: SUCCESS EXAMPLES (CORRECT TASK COMPLETION) ---"})
        for idx, img_path in enumerate(success_paths, 1):
            base64_img = encode_image(img_path)
            if base64_img:
                content.append(
                    {
                        "type": "text",
                        "text": f"[SUCCESS REFERENCE {idx}] This image shows a correct state for the target task.",
                    }
                )
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                )

    failure_paths = parse_image_paths(task["failure_examples"].get("front_camera", ""))
    if failure_paths:
        content.append({"type": "text", "text": "--- REFERENCE: FAILURE EXAMPLES (INCORRECT/FAILED TASK STATES) ---"})
        for idx, img_path in enumerate(failure_paths, 1):
            base64_img = encode_image(img_path)
            if base64_img:
                content.append(
                    {
                        "type": "text",
                        "text": f"[FAILURE REFERENCE {idx}] This image shows an incorrect/failed state for the target task.",
                    }
                )
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                )

    content.append(
        {
            "type": "text",
            "text": "--- OBSERVED WORKSPACE (AFTER SUBTASK — SINGLE IMAGE) ---",
        }
    )
    content.append(
        {
            "type": "text",
            "text": "The next image is the only evidence of the workspace state after the subtask. Judge completion from it.",
        }
    )

    for img_path in scene_image_paths:
        base64_img = encode_image(img_path)
        if base64_img:
            content.append(
                {
                    "type": "text",
                    "text": "[OBSERVED WORKSPACE — END STATE]",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})

    final_query = """
    --- FINAL EVALUATION INSTRUCTION ---
    Based on the single observed workspace image, reference examples, and Required Postcondition, output your analysis strictly following the format below:

    1.  Final Decision: [Task Fully Completed / Task Failed]
    2.  Goal Achievement Verification: [Does this image show the task goal is met? Cite visible evidence only]
    3.  Visual analysis: [2–5 sentences describing what in the image supports or contradicts the criteria]
    4.  Issue (If Task Failed): [Why the visible end state does not satisfy the criteria]
    """
    content.append({"type": "text", "text": final_query})

    return content


def verify_robotic_arm_motion(
    task_description: str,
    motion_sequence_image_paths: List[str],
    completion_criteria: str | None = None,
) -> None:
    """
    CLI-style entry: verify full robotic arm motion sequence and print results.
    """
    target_task = find_task_by_description(task_description) or find_task_by_keyword_fallback(
        task_description
    )
    if not KNOWLEDGE_BASE or not motion_sequence_image_paths or not target_task:
        if not KNOWLEDGE_BASE:
            logger.error("Knowledge base is not loaded. Exiting.")
        elif not motion_sequence_image_paths:
            logger.error("No workspace images provided. Exiting.")
        else:
            logger.error("No matching task found for description: %r. Exiting.", task_description)
        return

    print("=" * 70)
    print(f"Loaded Task: {target_task['task_description']}")
    print(f"Task Index: {target_task['index']}")
    print(f"Workspace images (completion): {len(motion_sequence_image_paths)}")
    print("=" * 70)

    text = verify_robotic_arm_motion_with_response(
        task_description,
        motion_sequence_image_paths,
        completion_criteria=completion_criteria or task_description,
    )
    if text:
        print("\n" + "=" * 30 + " ANALYSIS RESULT " + "=" * 30)
        print(text)
        print("=" * 72)


if __name__ == "__main__":
    root = _repo_root()
    TARGET_TASK = "Place the 15 mL centrifuge tube into the centrifuge tube rack"
    MOTION_SEQUENCE_IMAGES = [
        str(root / "Multimodal_detection_system/BioRobo-MVKB/test/insert_15ml_centri_tube_rack_success/test1.png"),
        str(root / "Multimodal_detection_system/BioRobo-MVKB/test/insert_15ml_centri_tube_rack_success/test2.png"),
        str(root / "Multimodal_detection_system/BioRobo-MVKB/test/insert_15ml_centri_tube_rack_success/test3.png"),
    ]
    verify_robotic_arm_motion(TARGET_TASK, MOTION_SEQUENCE_IMAGES)
