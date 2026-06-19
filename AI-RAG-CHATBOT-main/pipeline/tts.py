import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import re
from loguru import logger
from config import settings
from groq import Groq


def _preprocess_text(text: str) -> str:
    """Fix common TTS misreadings before speaking."""
    text = re.sub(r'(?<!\d)(\d+)[.)]\s+', lambda m: f"{m.group(1)}. ", text)
    text = re.sub(r':\s+', '. ', text)
    text = re.sub(r';\s+', ', ', text)
    text = re.sub(r'(\d+)\.(\d+)', lambda m: f"{m.group(1)} point {m.group(2)}", text)
    text = re.sub(r'\s-\s', ', ', text)
    text = text.replace('—', ', ')
    text = text.replace('–', ', ')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'linkedin\.com\S*', '', text)
    text = re.sub(r'\+?\d{1,3}[\s-]?\d{10}', '', text)
    text = text.replace('→', 'to')
    text = text.replace('•', '')
    text = text.replace('●', '')
    text = text.replace('▪', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _speak_groq(text: str) -> bytes:
    """Groq Orpheus TTS — primary engine. Returns WAV bytes."""
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        input=text,
        response_format="wav",
    )
    return response.read()


def _speak_gtts(text: str) -> bytes:
    """gTTS fallback — Indian English accent. Returns MP3 bytes."""
    from gtts import gTTS
    tts = gTTS(text=text, lang="en-in", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


def speak(text: str, speed: str = None) -> bytes:
    """
    Convert text to speech. gTTS only (permanent for now).
    """
    if not text or not text.strip():
        logger.warning("speak(): empty text received, skipping.")
        return b""

    text = _preprocess_text(text)

    if not text.strip():
        return b""

    try:
        audio_bytes = _speak_gtts(text)
        if audio_bytes:
            logger.info("gTTS: {} bytes", len(audio_bytes))
            return audio_bytes
    except Exception as e:
        logger.error("gTTS failed: {}", e)

    return b""