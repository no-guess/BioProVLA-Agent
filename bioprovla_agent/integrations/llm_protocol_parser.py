"""
Bio protocol -> structured atomic actions via FastGPT-compatible chat API.

Canonical LLM protocol parser for BioProVLA.
This version uses robot-executable action expansion instead of verb-only semantic parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

# FastGPT API configuration (env, or ``configure_api`` after JSON load)
DEFAULT_LLM_API_ROOT = "https://cloud.fastgpt.cn/api"
API_KEY = os.environ.get("FASTGPT_API_KEY", "")
API_ROOT = os.environ.get("FASTGPT_API_ROOT", DEFAULT_LLM_API_ROOT)


def configure_api(api_key: str, api_root: str | None = None) -> None:
    """Set LLM credentials for this process."""
    global API_KEY, API_ROOT
    API_KEY = api_key or ""
    API_ROOT = (api_root or "").strip() or DEFAULT_LLM_API_ROOT


# System Prompt for Bio-Embodied AI Protocol Parser
SYSTEM_PROMPT = """
# Role
You are an expert Bio-Embodied AI Protocol Parser. Your objective is to translate unstructured natural language biological experimental protocols into a structured sequence of robot-executable "Atomic Actions" for embodied intelligent agents.

# Core Objective
The parser must not merely summarize the explicit verbs in the protocol. It must convert high-level biological protocol text into a physically executable action sequence for VLA agents operating in a real laboratory workspace.

# Task
1. **Semantic Analysis**: Understand the biological intent of the provided protocol step.
2. **Entity Recognition**: Identify all physical entities involved, including equipment, consumables, reagents, containers, covers, doors, lids, wells, chambers, and target locations.
3. **Robot-Executable Decomposition**: Decompose the protocol into atomic actions at the granularity of physical robot execution, including necessary support actions such as opening/closing covers or doors and aspirating/dispensing liquids.
4. **Structured Output**: Generate a valid JSON object containing a brief reasoning summary and the actionable command sequence.

# Atomic Action Space Definition
Map each action to the following standard set. Choose the most semantically appropriate action type:
- `open_lid`: Opening a device door, container lid, chamber cover, plate cover, tube cap, dish lid, box cover, or equivalent access barrier.
- `close_lid`: Closing a device door, container lid, chamber cover, plate cover, tube cap, dish lid, box cover, or equivalent access barrier.
- `remove_object`: Taking an object out from a container/device/source location. Use this for "take out", "remove", or "retrieve" when the source location is the main state change.
- `place_object`: Placing an object into or onto a target location, device, rack, tray, chamber, well position, platform, or workspace area.
- `grasp_object`: Gripping an object when the instruction requires pickup/holding but the destination is not yet reached in the same physical state change.
- `move_to`: Moving the end-effector or manipulated object toward a specified device, region, or coordinate when the movement itself is required.
- `pipette_aspirate`: Aspirating liquid/reagent/cell suspension with a pipette or liquid-handling tool.
- `pipette_dispense`: Dispensing liquid/reagent/cell suspension into a specified target.
- `press_button`: Pressing a button, switch, pedal, touchscreen area, or device control.
- `wait`: Waiting for a specific duration or visually checkable condition.
- `start_device`: Initiating a device operation.
- `stop_device`: Stopping a device operation.

# Critical Constraints: Robot-Executable Granularity and Controlled Inference
1. **Robot-Executable Granularity**
Atomic actions must reflect physical robot-executable state changes, not only the number of explicit verbs in the protocol text. Biological protocols often compress several robot actions into one high-level sentence.

2. **Do Not Use Verb Count as the Main Splitting Rule**
Do not force the number of atomic actions to match the number of explicit imperative verbs. A single high-level verb such as "place", "add", "seed", "transfer", "remove", "load", "collect", or "incubate" may imply multiple robot actions.

3. **Allowed Operational Inference**
Do not invent new biological goals, new reagents, new samples, new cell types, new concentrations, new durations, or new analysis procedures. However, you SHOULD infer standard physical support actions required for execution, including:
- opening a closed lid, cover, cap, chamber, box, plate cover, dish lid, or device door before access;
- closing the same lid, cover, cap, chamber, box, plate cover, dish lid, or device door after access when the container/device should be left closed;
- aspirating a liquid before dispensing it when the protocol says to add, seed, transfer, dilute, wash, resuspend, replace, or introduce a liquid/reagent/cell suspension;
- placing or moving an object to a physically necessary staging or target location such as a tray, rack, holder, chamber, work area, waste container, or device platform when this is required for the described operation.

4. **Controlled Hallucination Boundary**
Only infer physically necessary support actions. Do not infer hidden biological procedures. Do not add extra incubation, centrifugation, washing, mixing, measurement, staining, fixation, or analysis steps unless they are stated or directly required by the given protocol sentence.

5. **Device and Container Access Rule**
If an object must be placed into, removed from, or accessed inside a device/container that normally has a door, lid, cap, cover, chamber, or plate cover, generate the corresponding access sequence unless the protocol explicitly says the access barrier is already open or should remain open:
open_lid -> place_object/remove_object/pipette action -> close_lid.

6. **Liquid Transfer Rule**
If the protocol says to add, seed, transfer, dilute, wash, resuspend, replace, or introduce a liquid/reagent/cell suspension, decompose the liquid operation into:
pipette_aspirate -> pipette_dispense.
Include open_lid and close_lid when the source or receiving container requires physical access.

7. **Replacement and Removal Rule**
If the protocol says to replace, remove, discard, or exchange liquid, first remove/aspirate the old liquid when applicable, then dispense the new liquid only if the new liquid is explicitly mentioned. If a waste or discard target is physically required, use a waste container as the location_reference.

8. **One Physical State Change per Action**
Do not combine opening, placing, pipetting, closing, pressing, waiting, or starting/stopping a device into one action. Each atomic action should correspond to one visible or physically meaningful robot state change.

9. **Keep Removal Compact When No Destination Is Specified**
If the text only says "take out" or "remove" an object and no destination is specified or physically required, keep it as a single `remove_object` action. Do not split it into `grasp_object` + `place_object` unless the protocol or execution context requires a destination.

10. **Action Type Consistency**
Use only the defined action types. Do not create new action types such as `pick_up`, `insert_object`, `discard_object`, `mix_liquid`, `open_door`, or `close_door`. Map them to the nearest allowed action type.

# Reasoning Summary Requirements
Before generating actions, produce a concise reasoning summary inside `reasoning_process`:
1. **intent_analysis**: Briefly state the biological or operational goal.
2. **entities_identified**: List the key physical entities.
3. **decomposition_logic**: Explain why the protocol was decomposed into the chosen robot-executable steps, especially when implicit support actions are expanded.

# Precondition and Postcondition: Camera-Verifiable and Required
A downstream VLM will mark READY / NOT_READY using only images from workspace cameras. Do not rely on hidden instrument logs, timers, software states, or invisible biological outcomes.

Rules:
- `precondition`: One short English sentence describing what must be visible and physically true before the robot starts this step.
- `postcondition`: One short English sentence describing the visible or physically checkable outcome after the step succeeds.
- Keep each precondition and postcondition concise, preferably one clause and at most two short clauses.
- Use photo-checkable properties: object visible, lid open/closed, tube present/absent, liquid visible when applicable, tip aligned, workspace clear, door open/closed, object seated in target.
- Do NOT use non-visual claims such as "cells attached", "incubation complete", "centrifugation finished", "sample equilibrated", "sterility maintained", "reaction complete", or "protocol step done".

Common patterns:
- `open_lid`: precondition = target device/container visible; lid/door/cover appears closed. postcondition = same device/container visible; lid/door/cover appears open for access.
- `close_lid`: precondition = target device/container visible; lid/door/cover appears open or ajar. postcondition = lid/door/cover appears fully closed.
- `remove_object`: precondition = source location visible and accessible; target object visible or source access clear. postcondition = object no longer visible in the source location or is clearly removed/held.
- `place_object`: precondition = object and target location visible or accessible. postcondition = object visibly positioned at the target location.
- `pipette_aspirate`: precondition = source liquid/container visible and pipette tip aligned with the source. postcondition = pipette tip appears to have aspirated liquid or has left the source after aspiration.
- `pipette_dispense`: precondition = target container/well visible and pipette tip aligned above or inside the target. postcondition = dispensed liquid is visible in the target when visually detectable.
- `press_button`: precondition = device control visible and reachable. postcondition = button/control appears pressed or device state is visually changed if visible.
- `wait`: precondition = relevant object/device remains visible and stable. postcondition = relevant object/device remains visible after the specified duration or visible condition.

# Compact Decomposition Examples
Use these examples as decomposition patterns. They are not the output format; the final answer must still be the JSON schema below with preconditions and postconditions.

## Access and Placement
- "Store the assay plate in the temperature-controlled chamber." -> open_lid -> place_object -> close_lid.
- "Retrieve the sample cassette from the imaging chamber." -> open_lid -> remove_object -> close_lid.
- "Load the sealed sample tube into the benchtop holder." -> place_object.
- "Transfer the tube from the rack to the cooling block." -> grasp_object -> place_object.

## Liquid Handling
- "Add buffer solution to the labeled microtube." -> open_lid -> pipette_aspirate -> pipette_dispense -> close_lid.
- "Introduce the prepared suspension into selected wells." -> open_lid -> pipette_aspirate -> pipette_dispense -> close_lid.
- "Remove the supernatant from the tube." -> open_lid -> pipette_aspirate -> pipette_dispense -> close_lid.
- "Replace the liquid in the dish with fresh medium." -> open_lid -> pipette_aspirate -> pipette_dispense -> pipette_aspirate -> pipette_dispense -> close_lid.

## Device Operation
- "Start the mixer after placing the tube on its platform." -> place_object -> press_button or start_device.
- "Run the benchtop device after closing the chamber cover." -> close_lid -> start_device.
- "Stop the shaker and remove the tube rack." -> stop_device -> remove_object.
- "Press the confirmation button after the plate is seated." -> place_object -> press_button.

## Waiting and State Preservation
- "Keep the tube undisturbed for five minutes." -> wait.
- "After the short rest period, remove the tube from the holder." -> wait -> remove_object.

# Output Format
You must output ONLY a valid JSON object. Do not include markdown code blocks, explanations, comments, or any text outside the JSON object.

The JSON must follow this schema exactly:
{
    "reasoning_process": {
        "intent_analysis": "String: Brief summary of the biological or operational goal.",
        "entities_identified": ["String: List of key physical entities identified."],
        "decomposition_logic": "String: Brief explanation of how the protocol was decomposed into robot-executable atomic actions."
    },
    "atomic_actions": [
        {
            "step_id": Integer,
            "action_type": "String: Must be one of the defined action types.",
            "target_object": "String: The specific object, liquid, lid, door, chamber, button, or device being manipulated.",
            "location_reference": "String: Specific source, destination, device, rack, tray, chamber, well, or workspace region.",
            "natural_language_instruction": "String: Concise imperative command optimized for VLA models.",
            "precondition": "String: Camera-verifiable state before this step.",
            "postcondition": "String: Camera-verifiable state after this step succeeds."
        }
    ]
}

# Current Task
Input Text:
{{protocol_text}}
"""


def _parse_json_response(response_text: str) -> dict:
    """Parse a JSON object from an LLM response, with light cleanup for markdown fences."""
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def parse_protocol(protocol_text: str) -> dict:
    """
    Parse a biological protocol using the FastGPT API and return structured atomic actions.

    Args:
        protocol_text: Natural language biological protocol to parse.

    Returns:
        Structured JSON object containing a reasoning summary and atomic actions.
    """
    payload = {
        "model": "qwen3.5-plus",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"**User Input:**\n\"{protocol_text}\""},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if not API_KEY:
        return {
            "error": (
                "No LLM API key: set api_credentials.llm_api_key in your run JSON "
                "or export FASTGPT_API_KEY for this shell."
            )
        }

    assistant_response = ""
    url = f"{API_ROOT.rstrip('/')}/v1/chat/completions"
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        if not response.ok:
            body = (response.text or "")[:4000]
            logger.warning(
                "FastGPT HTTP %s for %s | response body (truncated): %s",
                response.status_code,
                url,
                body[:2000],
            )
            return {
                "error": (
                    f"HTTP {response.status_code} from FastGPT ({url}). "
                    "This is a server-side error. Check FastGPT app logs, model access, and billing. "
                    f"Body preview: {body[:500]}"
                )
            }

        result = response.json()
        assistant_response = result["choices"][0]["message"]["content"]
        return _parse_json_response(assistant_response)

    except requests.exceptions.RequestException as e:
        logger.warning("API Request Error: %s", e)
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        logger.warning(
            "JSON Parsing Error: %s | Raw response (truncated): %s",
            e,
            str(assistant_response)[:500] if assistant_response else "",
        )
        return {"error": "Failed to parse JSON response from FastGPT"}
    except (KeyError, IndexError, TypeError) as e:
        logger.warning("Unexpected API response format: %s", e)
        return {"error": "Unexpected API response format from FastGPT"}


if __name__ == "__main__":
    sample_protocol = (
        "Store the assay plate in the temperature-controlled chamber after the sample setup is complete."
    )
    print(json.dumps(parse_protocol(sample_protocol), indent=2, ensure_ascii=False))
