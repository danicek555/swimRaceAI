"""swimRaceAI — vlm."""

import base64
import json
import os
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from .config import *  # noqa: F401,F403


_VLM_REF_CACHE: dict[str, str] | None = None


def reset_vlm_reference_cache() -> None:
    """Zahodit cache few-shot referenci (po zmene souboru refs)."""
    global _VLM_REF_CACHE
    _VLM_REF_CACHE = None


def vlm_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SWIM_VLM_API_KEY")
    return key.strip() if key else None



def encode_image_jpeg_b64(
    bgr: np.ndarray,
    max_side: int = VLM_JPEG_MAX_SIDE,
    allow_upscale: bool = False,
) -> str:
    """Resize and JPEG-encode a crop for the vision LM.

    allow_upscale=True enlarges small crops up to max_side: a tight 150px
    foot+block region carries more usable detail at 512px than a wide crop
    with four lanes squeezed into the same pixels.
    """
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if not allow_upscale:
        scale = min(1.0, scale)
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode crop as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")



# --- NEPOUZIVANE (zadne volani v projektu) — ponechano zakomentovane pro pripadne oziveni ---
# def encode_image_file_b64(path: Path, max_side: int = VLM_JPEG_MAX_SIDE) -> str | None:
#     """Load a reference image from disk and JPEG-encode it for the LM."""
#     if not path.is_file():
#         return None
#     img = cv2.imread(str(path))
#     if img is None:
#         return None
#     return encode_image_jpeg_b64(img, max_side=max_side)
#
#
# --- konec nepouzivaneho bloku ---

def vlm_reference_b64() -> dict[str, str]:
    """Cached base64 for few-shot ON / LEFT example photos (clean, no overlays)."""
    global _VLM_REF_CACHE
    if _VLM_REF_CACHE is not None:
        return _VLM_REF_CACHE
    refs: dict[str, str] = {}
    for key, path in (
        ("ON_BLOCK", VLM_REF_ON),
        ("ON_BLOCK_HARD", VLM_REF_ON_HARD),
        ("ON_BLOCK_WEDGE", VLM_REF_ON_WEDGE),
        ("LEFT_BLOCK", VLM_REF_LEFT),
        ("LEFT_BLOCK_HARD", VLM_REF_LEFT_HARD),
    ):
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        refs[key] = encode_image_jpeg_b64(img)
    _VLM_REF_CACHE = refs
    return refs



def vlm_ssl_context() -> ssl.SSLContext:
    """Use certifi CA bundle when present (fixes macOS CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()



def parse_json_bool(value: object, default: bool = False) -> bool:
    """Parse JSON bools safely (bool('false') is True in Python — never use that)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", "none"):
            return False
    return default



def _foot_on(value: object) -> bool:
    """True if a foot field means still contacting the block."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("on", "contact", "touching", "yes", "true", "1", "rear", "front"):
        return True
    if s in ("off", "clear", "none", "no", "false", "0", "air", "water"):
        return False
    return False



def label_from_vlm_json(obj: dict) -> str:
    """Map structured LM evidence to ON_BLOCK / LEFT_BLOCK.

    Leave only when BOTH feet are clear. Any toe still touching => ON.
    Ambiguous answers default to ON (early LEFT underestimates RT).
    """
    rear = obj.get("rear_foot", obj.get("left_foot", obj.get("foot")))
    front = obj.get("front_foot", obj.get("right_foot"))
    any_toe = obj.get("any_toe_contact", obj.get("contact"))
    both_clear = obj.get("both_feet_clear")
    platform_empty = obj.get("platform_empty")
    gap = str(obj.get("gap", "")).strip().lower()

    rear_on = _foot_on(rear) if rear is not None else None
    front_on = _foot_on(front) if front is not None else None

    if rear_on is True or front_on is True:
        return "ON_BLOCK"
    if parse_json_bool(any_toe, default=False):
        return "ON_BLOCK"
    if both_clear is not None and not parse_json_bool(both_clear, False):
        return "ON_BLOCK"
    if gap in ("none", "tiny"):
        return "ON_BLOCK"

    if rear_on is False and front_on is False:
        return "LEFT_BLOCK"
    if parse_json_bool(both_clear, False):
        return "LEFT_BLOCK"
    if parse_json_bool(platform_empty, False) and not parse_json_bool(any_toe, False):
        return "LEFT_BLOCK"
    if gap == "clear" and rear_on is not True and front_on is not True:
        return "LEFT_BLOCK"

    foot = str(obj.get("foot", "")).strip().lower()
    if foot in ("rear", "front", "on"):
        return "ON_BLOCK"
    if foot in ("none", "off", "in_water") and parse_json_bool(any_toe, False) is False:
        if parse_json_bool(platform_empty, False) or gap == "clear":
            return "LEFT_BLOCK"

    return "ON_BLOCK"



def vlm_token_limit_kwargs(n: int = 120) -> dict[str, int]:
    """
    Newer OpenAI models (gpt-5*, o-series) reject max_tokens and require
    max_completion_tokens. Older vision models still want max_tokens.
    """
    name = VLM_MODEL.lower()
    if name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}



def ask_vlm_on_or_left(crop_bgr: np.ndarray) -> tuple[str | None, str]:
    """
    Ask a vision LM for structured foot-vs-block evidence.

    Returns (label, raw_reply). Raises RuntimeError('vlm_auth') on HTTP 401.
    """
    key = vlm_api_key()
    if not key:
        return None, ""

    # Upscale small tight crops: more usable pixels on the one foot that matters.
    b64 = encode_image_jpeg_b64(crop_bgr, allow_upscale=True)
    refs = vlm_reference_b64()

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Side-view crop of ONE swim start. Decide if the MAIN swimmer "
                "still has ANY foot contact with the starting block "
                "(flat red top OR rear red wedge).\n\n"
                "Balanced rules:\n"
                "- ON if rear foot is still on the rear WEDGE, even when the "
                "front foot has already lifted and the body is diving.\n"
                "- ON if front toes are still pressed on the flat red top "
                "(no air under the contact point).\n"
                "- LEFT only when BOTH feet are clear of top AND wedge "
                "(visible gap under the last foot, or platform empty).\n"
                "- Front foot in the air alone does NOT mean LEFT while the "
                "rear foot remains on the wedge.\n"
                "- Foot past the front edge with a clear gap under the toes, "
                "AND rear foot also clear = LEFT.\n"
                "- Hands do not count. Ignore other lanes.\n"
                "- The photo is cropped on ONE target block: the LARGE "
                "platform at the CENTER/BOTTOM of the image. Regions outside "
                "the target block and its swimmer may be DARKENED — darkened "
                "areas are OTHER lanes, never judge them.\n"
                "- PARALLAX: a flying swimmer's foot can visually overlap a "
                "FARTHER lane's block (smaller, higher in the image). That is "
                "NOT contact. Contact counts ONLY on the target block.\n\n"
                "Reply JSON only:\n"
                '{"rear_foot":"on"|"off","front_foot":"on"|"off",'
                '"any_toe_contact":true|false,"both_feet_clear":true|false,'
                '"gap":"none"|"tiny"|"clear"}'
            ),
        }
    ]
    if "ON_BLOCK" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": "EXAMPLE → ON: foot still contacting red top/wedge.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "ON_BLOCK_WEDGE" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → ON (rear wedge): front foot may already be in "
                        "the air, but rear foot is STILL on the slanted red wedge. "
                        "rear_foot=on, both_feet_clear=false. This is ON, not LEFT."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK_WEDGE']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "ON_BLOCK_HARD" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → ON (last toes): diving, one leg up, but front "
                        "toes still pressed on the front red edge with no gap."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK_HARD']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "LEFT_BLOCK" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → LEFT: both feet clear / empty platform. "
                        "rear_foot=off, front_foot=off, both_feet_clear=true."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['LEFT_BLOCK']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "LEFT_BLOCK_HARD" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → LEFT (both clear): diving with the front foot "
                        "near the edge BUT airborne with a gap, AND the rear foot "
                        "also off the wedge. Only then LEFT."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['LEFT_BLOCK_HARD']}",
                        "detail": "high",
                    },
                },
            ]
        )
    user_content.extend(
        [
            {
                "type": "text",
                "text": (
                    "QUERY: Check the MAIN swimmer's REAR foot on the wedge AND "
                    "FRONT foot on the top of the TARGET block (the large "
                    "platform at the center/bottom). If EITHER still touches, "
                    "answer ON. LEFT only if BOTH are clear of the TARGET block. "
                    "A foot overlapping a farther lane's block is parallax, not "
                    "contact. JSON only."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
    )


    payload = {
        "model": VLM_MODEL,
        "temperature": 0,
        **vlm_token_limit_kwargs(120),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a World Aquatics-style swim-start timer. Output JSON only. "
                    "ON if either foot still contacts the red top OR rear wedge. "
                    "A lifted front foot with rear foot still on the wedge is ON. "
                    "LEFT only when both feet are clear. Hands do not count."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    }
    req = urllib.request.Request(
        f"{VLM_API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90, context=vlm_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 401:
            raise RuntimeError("vlm_auth") from err
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        print(f"  Vision LM request failed: HTTP {err.code} {err.reason} {detail}")
        return None, ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        print(f"  Vision LM request failed: {err}")
        return None, ""

    try:
        text_out = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        return None, ""

    raw = text_out
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start_j = raw.find("{")
    end_j = raw.rfind("}")
    if start_j < 0 or end_j < start_j:
        upper = text_out.upper()
        if "LEFT" in upper and "ON" not in upper:
            return "LEFT_BLOCK", text_out
        if "ON" in upper:
            return "ON_BLOCK", text_out
        return None, text_out
    try:
        obj = json.loads(raw[start_j : end_j + 1])
    except json.JSONDecodeError:
        return None, text_out
    if not isinstance(obj, dict):
        return None, text_out
    return label_from_vlm_json(obj), text_out




def label_from_hand_vlm_json(obj: dict) -> str:
    """Map LM JSON to HANDS_AIR / HANDS_WATER / HANDS_UNKNOWN."""
    vis = str(obj.get("visibility", "")).strip().lower()
    blocker = parse_json_bool(obj.get("blocker"), default=False)
    hands_visible = obj.get("hands_visible")
    only_splash = parse_json_bool(obj.get("only_splash"), default=False)

    # Person/object covering the dive lane → cannot judge this frame.
    if blocker or vis in ("occluded", "blocked", "hidden"):
        return "HANDS_UNKNOWN"
    if (
        hands_visible is not None
        and not parse_json_bool(hands_visible, True)
        and not only_splash
        and not parse_json_bool(obj.get("splash"), False)
    ):
        return "HANDS_UNKNOWN"

    hands = str(obj.get("hands", "")).strip().lower()
    wet = parse_json_bool(obj.get("fingers_wet"), default=False)
    splash = parse_json_bool(obj.get("splash"), default=False)
    contact = parse_json_bool(obj.get("water_contact"), default=False)
    body_vis = obj.get("body_visible", obj.get("swimmer_visible"))

    if only_splash:
        return "HANDS_WATER"
    # Body gone above water with splash elsewhere already handled; if body
    # not visible but not occluded → treated as entered.
    if body_vis is not None and not parse_json_bool(body_vis, True) and not blocker:
        if splash or contact or wet or only_splash or hands in ("water", "entry"):
            return "HANDS_WATER"
        # Ambiguous disappearance without splash → unknown, not AIR.
        return "HANDS_UNKNOWN"
    if hands in ("water", "in_water", "wet", "entry", "submerged"):
        return "HANDS_WATER"
    if wet or contact or splash:
        return "HANDS_WATER"
    if hands in ("air", "above", "flying", "flight"):
        return "HANDS_AIR"
    return "HANDS_AIR"



def ask_vlm_hands_air_or_water(crop_bgr: np.ndarray) -> tuple[str | None, str]:
    """Ask vision LM: AIR / WATER / UNKNOWN (occluded)."""
    key = vlm_api_key()
    if not key:
        return None, ""

    b64 = encode_image_jpeg_b64(crop_bgr, allow_upscale=True)
    refs: dict[str, str] = {}
    for key_name, path in (
        ("HANDS_AIR", VLM_REF_HANDS_AIR),
        ("HANDS_WATER", VLM_REF_HANDS_WATER),
        ("HANDS_WATER_SPLASH", VLM_REF_HANDS_WATER_SPLASH),
    ):
        if path.is_file():
            img = cv2.imread(str(path))
            if img is not None:
                refs[key_name] = encode_image_jpeg_b64(img, allow_upscale=True)

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Crop of ONE tracked swimmer's dive (side view).\n"
                "First: can you SEE this swimmer's hands vs the water?\n"
                "Then: have the hands started water entry?\n\n"
                "Visibility:\n"
                "- visibility=occluded / blocker=true if a person, hat, pole, "
                "or object covers the dive so you cannot see hands+water. "
                "Do NOT guess AIR/WATER when blocked.\n"
                "- visibility=clear when hands (or entry splash) are visible.\n"
                "- visibility=partial if only partly visible.\n\n"
                "Entry (only if not occluded):\n"
                "- hands=air: clear air gap under fingertips, no entry foam.\n"
                "- hands=water: hands in water OR white splash at wrists, even "
                "if torso still above water; OR mostly splash and body gone.\n"
                "- only_splash=true when little body is visible and foam remains.\n\n"
                "Reply JSON only:\n"
                '{"visibility":"clear"|"partial"|"occluded","blocker":true|false,'
                '"hands_visible":true|false,'
                '"hands":"air"|"water","fingers_wet":true|false,'
                '"splash":true|false,"water_contact":true|false,'
                '"body_visible":true|false,"only_splash":true|false}'
            ),
        }
    ]
    if "HANDS_AIR" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → clear AIR: air gap under hands, no entry splash."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['HANDS_AIR']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "HANDS_WATER" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → clear WATER: hands/forearms in water with "
                        "splash; torso may still be above."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['HANDS_WATER']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "HANDS_WATER_SPLASH" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → WATER via splash only: body mostly gone, "
                        "white water remains. Not occluded — already entered."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['HANDS_WATER_SPLASH']}",
                        "detail": "high",
                    },
                },
            ]
        )
    user_content.extend(
        [
            {
                "type": "text",
                "text": (
                    "QUERY: If a deck person/hat blocks the hands, set "
                    "visibility=occluded and blocker=true (do not guess). "
                    "Otherwise: still in flight, or hand entry/splash started? "
                    "JSON only."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
    )

    payload = {
        "model": VLM_MODEL,
        "temperature": 0,
        **vlm_token_limit_kwargs(140),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You time swim dive HAND ENTRY. Output JSON only. "
                    "If a person/object blocks the view of hands vs water, "
                    "mark occluded/blocker — never invent contact. "
                    "When clear: AIR = gap under hands; WATER = hands in water "
                    "or entry splash or splash-only after entry."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    }
    req = urllib.request.Request(
        f"{VLM_API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90, context=vlm_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 401:
            raise RuntimeError("vlm_auth") from err
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        print(f"  Hand-entry LM request failed: HTTP {err.code} {err.reason} {detail}")
        return None, ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        print(f"  Hand-entry LM request failed: {err}")
        return None, ""

    try:
        text_out = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        return None, ""

    raw = text_out
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start_j = raw.find("{")
    end_j = raw.rfind("}")
    if start_j < 0 or end_j < start_j:
        upper = text_out.upper()
        if "OCCLUDED" in upper or "BLOCKER" in upper:
            return "HANDS_UNKNOWN", text_out
        if "WATER" in upper:
            return "HANDS_WATER", text_out
        if "AIR" in upper:
            return "HANDS_AIR", text_out
        return None, text_out
    try:
        obj = json.loads(raw[start_j : end_j + 1])
    except json.JSONDecodeError:
        return None, text_out
    if not isinstance(obj, dict):
        return None, text_out
    return label_from_hand_vlm_json(obj), text_out



