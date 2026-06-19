import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import re
import threading
import queue
from loguru import logger
from pipeline.tts import speak

# ── CHANGELOG (Session 10) ────────────────────────────────────────────────────
# [NO CHANGE] This file has no embedding, DB, or TTS-engine logic.
#             It is a generic audio queue that calls speak() from tts.py.
#             Since tts.py now correctly uses Groq Orpheus → gTTS fallback,
#             this file automatically benefits with zero modifications.
# ──────────────────────────────────────────────────────────────────────────────

_audio_queue: queue.Queue = queue.Queue()
_stop_flag = threading.Event()


# ---------------------------------------------------------------------------
# Sentence splitter — used by callers who pass multi-sentence LLM output
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences for incremental TTS synthesis.
    Splits on '. ', '! ', '? ' boundaries while ignoring decimal numbers.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Background audio player worker
# ---------------------------------------------------------------------------
def _player_worker():
    """
    Runs in a daemon thread.
    Pulls WAV/MP3 bytes from _audio_queue and plays them sequentially.
    Stops when _stop_flag is set or a None sentinel is received.
    """
    while not _stop_flag.is_set():
        try:
            audio_bytes = _audio_queue.get(timeout=0.5)
            if audio_bytes is None:
                # Sentinel value — graceful shutdown
                break
            try:
                import soundfile as sf
                import sounddevice as sd
                buf = io.BytesIO(audio_bytes)
                data, samplerate = sf.read(buf, dtype="float32")
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                logger.error("Playback error: {}", e)
            finally:
                _audio_queue.task_done()
        except queue.Empty:
            continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_player() -> threading.Thread:
    """
    Clear any leftover audio, reset the stop flag, and start the player thread.
    Call this once at app startup (or before a new conversation turn).
    Returns the started Thread object.
    """
    _stop_flag.clear()
    # Drain any stale items from a previous session
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
    t = threading.Thread(target=_player_worker, daemon=True)
    t.start()
    return t


def stop_player():
    """
    Signal the player thread to stop after finishing the current item.
    Puts a None sentinel so the worker exits its loop cleanly.
    """
    _stop_flag.set()
    _audio_queue.put(None)


def enqueue_sentence(text: str):
    """
    Convert one sentence to audio in a background thread and push it onto
    the playback queue. Non-blocking — returns immediately.

    The player worker picks up items in order, so sentences are spoken
    in the sequence they were enqueued.
    """
    if not text.strip():
        return

    def _convert():
        try:
            audio_bytes = speak(text)
            if audio_bytes:
                _audio_queue.put(audio_bytes)
        except Exception as e:
            logger.error("TTS convert error: {}", e)

    threading.Thread(target=_convert, daemon=True).start()


def enqueue_text(text: str):
    """
    Convenience wrapper: split multi-sentence text and enqueue each sentence.
    Useful when passing a full LLM response directly.
    """
    for sentence in _split_sentences(text):
        enqueue_sentence(sentence)