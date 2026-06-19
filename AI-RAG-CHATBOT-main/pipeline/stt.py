import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
from loguru import logger
from config import settings

# ── CHANGELOG (Session 10) ────────────────────────────────────────────────────
# [NO CHANGE] This file was already on the correct cloud stack.
#             Groq Whisper (whisper-large-v3) is the target STT engine.
#             No local dependencies (faster-whisper / ffmpeg) needed.
# ──────────────────────────────────────────────────────────────────────────────


def transcribe_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe raw audio bytes using Groq Whisper API.

    Accepts any audio format Groq supports (wav, mp3, webm, ogg, flac, m4a).
    The buffer is named 'audio.wav' so Groq detects the format correctly when
    the caller passes WAV bytes (standard from Streamlit's audio recorder).

    Returns:
        Transcribed text string, or "" on failure / empty input.
    """
    if not audio_bytes:
        logger.warning("transcribe_bytes: received empty audio bytes, skipping.")
        return ""

    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)

        # Groq SDK expects a file-like object with a .name attribute
        audio_buffer      = io.BytesIO(audio_bytes)
        audio_buffer.name = "audio.wav"

        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_buffer,
            language="en",
            response_format="text",
        )

        # Groq returns plain str when response_format="text"
        if isinstance(transcription, str):
            transcript = transcription.strip()
        else:
            transcript = transcription.text.strip()

        logger.info("Groq Whisper transcribed: '{}'", transcript[:80])
        return transcript

    except Exception as e:
        logger.error("Groq Whisper transcription failed: {}", e)
        return ""


if __name__ == "__main__":
    print("Groq Whisper STT ready")
    print("Model: whisper-large-v3")