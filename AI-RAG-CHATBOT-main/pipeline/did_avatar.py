import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import base64
import tempfile
import os
import requests
from loguru import logger
from config import settings

# ── CHANGELOG (Session 11c) ───────────────────────────────────────────────────
# [FIXED] file.io blocked — switched to D-ID's own /audios upload endpoint
# [CHANGED] Audio delivery: file.io → D-ID /audios API → use returned URL
# ──────────────────────────────────────────────────────────────────────────────

DID_BASE_URL       = "https://api.d-id.com"
DEFAULT_AVATAR_URL = "https://create-images-results.d-id.com/DefaultPresenters/Noelle_f/image.png"


def _auth_headers() -> dict:
    encoded = base64.b64encode(settings.DID_API_KEY.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _upload_audio_to_did(mp3_bytes: bytes) -> str | None:
    """
    Upload MP3 directly to D-ID's /audios endpoint.
    Returns the audio URL hosted on D-ID's CDN, or None on failure.
    """
    try:
        resp = requests.post(
            f"{DID_BASE_URL}/audios",
            headers=_auth_headers(),
            files={"audio": ("audio.mp3", mp3_bytes, "audio/mpeg")},
            timeout=30,
        )
        if resp.status_code == 201:
            url = resp.json().get("url")
            logger.info("[D-ID] Audio uploaded to D-ID CDN → {}", url)
            return url
        else:
            logger.error("[D-ID] Audio upload failed: {} - {}", resp.status_code, resp.text)
            return None
    except Exception as e:
        logger.error("[D-ID] Audio upload error: {}", e)
        return None


def _create_talk(audio_url: str) -> str | None:
    """Submit audio URL to D-ID. Returns talk_id or None."""
    headers = _auth_headers()
    headers["Content-Type"] = "application/json"

    payload = {
        "source_url": DEFAULT_AVATAR_URL,
        "script": {
            "type": "audio",
            "audio_url": audio_url,
        },
        "config": {
            "fluent": True,
            "pad_audio": 0.0,
        },
    }

    try:
        resp = requests.post(
            f"{DID_BASE_URL}/talks",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 201:
            talk_id = resp.json().get("id")
            logger.info("[D-ID] Talk created → {}", talk_id)
            return talk_id
        else:
            logger.error("[D-ID] Create failed: {} - {}", resp.status_code, resp.text)
            return None
    except Exception as e:
        logger.error("[D-ID] Request error: {}", e)
        return None


def _poll_video(talk_id: str, max_wait: int = 60) -> bytes | None:
    """Poll D-ID every 2s until video is ready. Returns MP4 bytes or None."""
    elapsed = 0
    while elapsed < max_wait:
        try:
            resp = requests.get(
                f"{DID_BASE_URL}/talks/{talk_id}",
                headers=_auth_headers(),
                timeout=15,
            )

            if resp.status_code != 200:
                logger.error("[D-ID] Poll error: {}", resp.status_code)
                return None

            data   = resp.json()
            status = data.get("status")

            if status == "done":
                video_url = data.get("result_url")
                logger.info("[D-ID] Video ready → {}", video_url)
                return requests.get(video_url, timeout=30).content

            elif status == "error":
                logger.error("[D-ID] Talk failed: {}", data)
                return None

            else:
                logger.debug("[D-ID] Status: {} ({}/{}s)", status, elapsed, max_wait)
                time.sleep(2)
                elapsed += 2

        except Exception as e:
            logger.error("[D-ID] Poll exception: {}", e)
            return None

    logger.error("[D-ID] Timeout after {}s", max_wait)
    return None


def generate_avatar_video(mp3_bytes: bytes) -> bytes | None:
    """
    Full pipeline:
        gTTS MP3 bytes → D-ID /audios upload → D-ID /talks → MP4 video bytes
    """
    if not mp3_bytes:
        logger.warning("[D-ID] Empty audio received, skipping.")
        return None

    if not settings.DID_API_KEY:
        logger.error("[D-ID] DID_API_KEY not set in .env!")
        return None

    # Step 1 — upload audio to D-ID CDN
    audio_url = _upload_audio_to_did(mp3_bytes)
    if not audio_url:
        return None

    # Step 2 — create talk with audio URL
    talk_id = _create_talk(audio_url)
    if not talk_id:
        return None

    # Step 3 — poll until video ready
    return _poll_video(talk_id)
