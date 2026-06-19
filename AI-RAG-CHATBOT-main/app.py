import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import base64
import threading
import queue
import io
import time
from typing import List
import streamlit as st
import streamlit.components.v1 as components
from loguru import logger
from pipeline.knowledge_base import ingest_document, is_indexed, get_qdrant_client, get_collection
from pipeline.llm import answer_query_stream, answer_query, clear_cache
from pipeline.tts import speak
from pipeline.stt import transcribe_bytes
from pipeline.did_avatar import generate_avatar_video

# ─── Supported Extensions ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "doc", "xlsx", "xls", "xlsm", "ods", "csv", "tsv",
    "pptx", "ppt", "txt", "md", "html", "htm", "xml", "json", "jsonl",
    "yaml", "yml", "eml", "msg", "py", "js", "ts", "java", "cs", "cpp",
    "sql", "log", "rtf", "epub",
]

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA · AI Research Intelligence Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font: Space Grotesk for display, Inter for body ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --aria-ink:        #0f0f14;
    --aria-surface:    #16161f;
    --aria-panel:      #1c1c28;
    --aria-card:       #22222f;
    --aria-border:     rgba(120,118,180,0.18);
    --aria-border-hov: rgba(120,118,180,0.38);
    --aria-purple:     #7c72d8;
    --aria-purple-dim: rgba(124,114,216,0.12);
    --aria-teal:       #2ac8a0;
    --aria-teal-dim:   rgba(42,200,160,0.10);
    --aria-coral:      #e87050;
    --aria-gold:       #f0ab40;
    --aria-text:       #e8e6f2;
    --aria-muted:      #8c8aaa;
    --aria-subtle:     #3a3850;
    --aria-radius:     14px;
    --aria-radius-lg:  20px;
}

/* ── Base Streamlit overrides ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--aria-ink) !important;
    color: var(--aria-text) !important;
}
.main { background: var(--aria-ink) !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { display: none; }
footer { display: none !important; }

/* ── Main layout grid ── */
.aria-shell {
    display: grid;
    grid-template-columns: 320px 1fr;
    grid-template-rows: 72px 1fr;
    min-height: 100vh;
    gap: 0;
}
.aria-topbar {
    grid-column: 1 / -1;
    grid-row: 1;
    display: flex;
    align-items: center;
    padding: 0 28px;
    background: var(--aria-surface);
    border-bottom: 1px solid var(--aria-border);
    gap: 16px;
    z-index: 10;
}
.aria-sidebar {
    grid-column: 1;
    grid-row: 2;
    background: var(--aria-panel);
    border-right: 1px solid var(--aria-border);
    padding: 24px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.aria-main {
    grid-column: 2;
    grid-row: 2;
    display: flex;
    flex-direction: column;
    height: calc(100vh - 72px);
    overflow: hidden;
}

/* ── Topbar ── */
.aria-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: -0.4px;
    color: var(--aria-text);
    display: flex;
    align-items: center;
    gap: 10px;
}
.aria-logo-hex {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--aria-purple), var(--aria-teal));
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: flex; align-items: center; justify-content: center;
}
.aria-status-pill {
    display: flex; align-items: center; gap: 6px;
    background: var(--aria-teal-dim);
    border: 1px solid rgba(42,200,160,0.25);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--aria-teal);
    font-weight: 500;
}
.aria-dot { width:6px; height:6px; border-radius:50%; background:var(--aria-teal); }
.aria-dot.pulse { animation: dot-pulse 2s ease-in-out infinite; }
@keyframes dot-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }

/* ── Sidebar sections ── */
.aria-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--aria-muted);
    margin-bottom: 10px;
}
.aria-upload-zone {
    border: 1.5px dashed var(--aria-border-hov);
    border-radius: var(--aria-radius);
    padding: 20px 16px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    background: var(--aria-purple-dim);
}
.aria-upload-zone:hover { border-color: var(--aria-purple); background: rgba(124,114,216,.18); }
.aria-upload-icon { font-size: 28px; margin-bottom: 8px; opacity: .8; }
.aria-upload-text { font-size: 13px; color: var(--aria-muted); line-height: 1.5; }
.aria-doc-card {
    background: var(--aria-card);
    border: 1px solid var(--aria-border);
    border-radius: var(--aria-radius);
    padding: 14px 16px;
    display: flex; align-items: center; gap: 12px;
}
.aria-doc-icon {
    width: 36px; height: 36px;
    background: var(--aria-purple-dim);
    border: 1px solid rgba(124,114,216,.25);
    border-radius: 10px;
    display:flex; align-items:center; justify-content:center;
    font-size: 16px; flex-shrink: 0;
}
.aria-doc-name { font-size: 13px; font-weight: 500; color: var(--aria-text); word-break:break-all; }
.aria-doc-meta { font-size: 11px; color: var(--aria-muted); margin-top: 2px; }
.aria-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 16px; border-radius: 10px;
    font-size: 13px; font-weight: 500;
    cursor: pointer; border: none;
    transition: all .15s;
}
.aria-btn-ghost {
    background: transparent;
    border: 1px solid var(--aria-border-hov);
    color: var(--aria-muted);
}
.aria-btn-ghost:hover { background: var(--aria-subtle); color: var(--aria-text); }
.aria-btn-danger {
    background: rgba(232,112,80,.10);
    border: 1px solid rgba(232,112,80,.25);
    color: var(--aria-coral);
}
.aria-btn-danger:hover { background: rgba(232,112,80,.20); }

/* ── Chat area ── */
.aria-chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 32px 6%;
    display: flex;
    flex-direction: column;
    gap: 28px;
    scrollbar-width: thin;
    scrollbar-color: var(--aria-subtle) transparent;
}
.aria-chat-scroll::-webkit-scrollbar { width: 5px; }
.aria-chat-scroll::-webkit-scrollbar-track { background: transparent; }
.aria-chat-scroll::-webkit-scrollbar-thumb { background: var(--aria-subtle); border-radius: 99px; }

/* ── Message bubbles ── */
.aria-msg-user {
    display: flex; justify-content: flex-end;
}
.aria-bubble-user {
    max-width: 68%;
    background: linear-gradient(135deg, #3a3265, #2d2758);
    border: 1px solid rgba(124,114,216,.3);
    border-radius: 20px 20px 6px 20px;
    padding: 14px 18px;
    font-size: 14.5px;
    line-height: 1.65;
    color: var(--aria-text);
}

/* ── Split answer row: avatar (left) + text (right) ── */
.aria-answer-row {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 20px;
    align-items: stretch;
}
@media (max-width: 900px) {
    .aria-answer-row { grid-template-columns: 1fr; }
}

/* ── Avatar panel ── */
.aria-avatar-panel {
    background: var(--aria-panel);
    border: 1px solid var(--aria-border);
    border-radius: var(--aria-radius);
    padding: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    min-height: 220px;
}
.aria-avatar-panel video {
    width: 100%;
    border-radius: 12px;
    display: block;
}
.aria-avatar-label {
    font-size: 10px; color: var(--aria-muted);
    text-transform: uppercase; letter-spacing: .8px;
    font-weight: 600;
}

/* ── Idle / generating avatar placeholder (used while no video yet) ── */
.aria-avatar-idle-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
}
.aria-avatar-ring {
    width: 88px; height: 88px;
    border-radius: 50%;
    border: 2px solid var(--aria-purple);
    padding: 5px;
    background: var(--aria-card);
    display: flex; align-items: center; justify-content: center;
    position: relative;
}
.aria-avatar-ring.thinking {
    animation: ring-spin 3s linear infinite;
    border-color: transparent;
    background: conic-gradient(var(--aria-purple) 0deg, var(--aria-teal) 120deg, transparent 200deg);
}
@keyframes ring-spin { to { transform: rotate(360deg); } }
.aria-avatar-inner {
    width: 74px; height: 74px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2a2638, #1a1628);
    border: 1.5px solid var(--aria-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    position: relative; z-index: 1;
    overflow: hidden;
}
.aria-avatar-glow {
    position: absolute; inset: -6px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,114,216,.3) 0%, transparent 70%);
    animation: glow-pulse 2.5s ease-in-out infinite;
}
@keyframes glow-pulse { 0%,100%{opacity:.5} 50%{opacity:1} }
.aria-avatar-status-text {
    font-size: 12px; color: var(--aria-muted); text-align: center; line-height: 1.5;
}

/* ── Bot text bubble (right side) ── */
.aria-msg-bot { display: flex; flex-direction: column; }
.aria-bubble-bot {
    background: var(--aria-card);
    border: 1px solid var(--aria-border);
    border-radius: 6px 20px 20px 20px;
    padding: 16px 20px;
    font-size: 14.5px;
    line-height: 1.7;
    color: var(--aria-text);
    position: relative;
    height: 100%;
}
.aria-bubble-bot code {
    background: rgba(124,114,216,.15);
    border: 1px solid rgba(124,114,216,.2);
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 13px;
    color: #b0aaee;
}
.aria-bubble-bot pre {
    background: #12121a;
    border: 1px solid var(--aria-border);
    border-radius: 10px;
    padding: 14px 16px;
    overflow-x: auto;
    margin: 10px 0;
}

/* ── Thinking animation (inside text panel) ── */
.aria-thinking {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px;
    background: var(--aria-card);
    border: 1px solid var(--aria-border);
    border-radius: 6px 20px 20px 20px;
    width: fit-content;
}
.aria-thinking-label {
    font-size: 12px;
    color: var(--aria-muted);
    font-weight: 500;
    letter-spacing: .3px;
}
.aria-thinking-dots { display: flex; gap: 5px; }
.aria-thinking-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--aria-purple);
    animation: dot-bounce .9s ease-in-out infinite;
}
.aria-thinking-dot:nth-child(2) { animation-delay: .15s; background: var(--aria-teal); }
.aria-thinking-dot:nth-child(3) { animation-delay: .30s; background: var(--aria-purple); opacity:.6; }
@keyframes dot-bounce { 0%,80%,100%{transform:scale(1)} 40%{transform:scale(1.4)} }

/* ── Source chips ── */
.aria-sources { margin-top: 14px; border-top: 1px solid var(--aria-border); padding-top: 12px; }
.aria-sources-label { font-size: 11px; font-weight: 600; letter-spacing: .8px; text-transform: uppercase; color: var(--aria-muted); margin-bottom: 8px; }
.aria-source-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--aria-teal-dim);
    border: 1px solid rgba(42,200,160,.2);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 11.5px;
    color: var(--aria-teal);
    margin: 3px 3px 0 0;
    cursor: pointer;
}

/* ── Input bar ── */
.aria-input-bar {
    border-top: 1px solid var(--aria-border);
    background: var(--aria-surface);
    padding: 18px 10%;
    display: flex; flex-direction: column; gap: 12px;
}

/* ── Voice input bar ── */
.aria-voice-row {
    display: flex; align-items: center; gap: 12px;
}
.aria-voice-hint { font-size: 12px; color: var(--aria-muted); }

/* ── Welcome state ── */
.aria-welcome {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 20px; padding: 48px;
    text-align: center;
}
.aria-welcome-hex {
    width: 80px; height: 80px;
    background: linear-gradient(135deg, var(--aria-purple-dim), var(--aria-teal-dim));
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display:flex; align-items:center; justify-content:center;
    font-size: 36px;
    border: 1px solid var(--aria-border);
    position:relative;
}
.aria-welcome-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px; font-weight: 600;
    color: var(--aria-text); letter-spacing: -.4px;
}
.aria-welcome-sub { font-size: 14.5px; color: var(--aria-muted); max-width: 380px; line-height: 1.65; }

/* ── Streamlit element overrides ── */
.stFileUploader > div { background: transparent !important; border: none !important; }
.stFileUploader label { color: var(--aria-muted) !important; font-size: 13px !important; }
.stAudio audio { width: 100%; border-radius: 10px; }
div[data-testid="stChatInput"] { background: var(--aria-card) !important; border: 1px solid var(--aria-border) !important; border-radius: var(--aria-radius) !important; }
div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--aria-text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
div[data-testid="stChatInput"]:focus-within { border-color: var(--aria-purple) !important; }
.stButton button {
    background: transparent !important;
    border: 1px solid var(--aria-border-hov) !important;
    color: var(--aria-muted) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all .15s !important;
}
.stButton button:hover {
    background: var(--aria-subtle) !important;
    color: var(--aria-text) !important;
    border-color: var(--aria-purple) !important;
}
.stSpinner > div { border-top-color: var(--aria-purple) !important; }
.stSuccess { background: var(--aria-teal-dim) !important; border: 1px solid rgba(42,200,160,.25) !important; border-radius: 10px !important; color: var(--aria-teal) !important; }
.stWarning { background: rgba(240,171,64,.08) !important; border: 1px solid rgba(240,171,64,.2) !important; border-radius: 10px !important; color: var(--aria-gold) !important; }
.stError   { background: rgba(232,112,80,.08) !important; border: 1px solid rgba(232,112,80,.2) !important; border-radius: 10px !important; color: var(--aria-coral) !important; }
.stInfo    { background: var(--aria-purple-dim) !important; border: 1px solid rgba(124,114,216,.2) !important; border-radius: 10px !important; color: var(--aria-purple) !important; }
div[data-testid="column"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── JS Audio Queue ──────────────────────────────────────────────────────────
components.html("""
<script>
  parent._vra_audioQueue   = [];
  parent._vra_audioPlaying = false;
  parent._vra_playNext = function() {
    if (parent._vra_audioQueue.length === 0) { parent._vra_audioPlaying = false; return; }
    parent._vra_audioPlaying = true;
    var b64 = parent._vra_audioQueue.shift();
    var audio = new Audio("data:audio/mp3;base64," + b64);
    audio.onended = parent._vra_playNext;
    audio.onerror = parent._vra_playNext;
    audio.play().catch(parent._vra_playNext);
  };
  parent.enqueueAudio = function(b64) {
    parent._vra_audioQueue.push(b64);
    if (!parent._vra_audioPlaying) parent._vra_playNext();
  };
  parent.clearAudioQueue = function() {
    parent._vra_audioQueue   = [];
    parent._vra_audioPlaying = false;
  };
</script>
""", height=0)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

# ─── Helpers ──────────────────────────────────────────────────────────────────
def delete_current_doc():
    current = st.session_state.get("current_doc")
    if not current:
        return
    try:
        client = get_qdrant_client()
        client.delete_collection(get_collection(current))
    except Exception as e:
        logger.warning("Could not delete Qdrant collection: {}", e)
    try:
        doc_path = Path("data/docs") / current
        if doc_path.exists():
            doc_path.unlink()
    except Exception as e:
        logger.warning("Could not delete document file: {}", e)
    clear_cache(current)
    st.session_state.current_doc = None
    st.session_state.messages = []

def _strip_markdown(text: str) -> str:
    text = re.sub(r'\*{1,3}|_{1,3}', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def _is_heading(text: str) -> bool:
    t = text.strip()
    if re.match(r'^#{1,6}\s+', t):
        return True
    if re.match(r'^\*{1,2}[^*]+\*{1,2}$', t):
        return True
    words = t.split()
    if len(words) <= 8 and not re.search(r'[.!?,]$', t):
        upper_count = sum(1 for w in words if w and w[0].isupper())
        if upper_count >= len(words) * 0.6:
            return True
    return False

def tts_worker(tts_queue: queue.Queue, result_queue: queue.Queue):
    while True:
        item = tts_queue.get()
        if item is None:
            result_queue.put(None)
            break
        idx, text = item
        audio = speak(text)
        result_queue.put((idx, audio))
        tts_queue.task_done()

MIN_PHRASE_CHARS = 15

def _flush_phrase(phrase: str, phrase_idx: int, tts_queue: queue.Queue) -> int:
    phrase = phrase.strip()
    if not phrase:
        return phrase_idx
    clean = _strip_markdown(phrase)
    if not clean.strip():
        return phrase_idx
    if _is_heading(clean):
        tts_queue.put((phrase_idx, clean))
        tts_queue.put((phrase_idx + 1, "..."))
        return phrase_idx + 2
    if len(clean) >= MIN_PHRASE_CHARS:
        tts_queue.put((phrase_idx, clean))
        return phrase_idx + 1
    return phrase_idx

def _push_audio_to_browser(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    components.html(f"<script>parent.enqueueAudio('{b64}')</script>", height=0)

def get_file_emoji(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        '.pdf': '📕', '.docx': '📘', '.doc': '📘',
        '.xlsx': '📗', '.xls': '📗', '.csv': '📊',
        '.pptx': '📙', '.ppt': '📙', '.txt': '📄',
        '.md': '📝', '.json': '🔧', '.py': '🐍',
        '.js': '🟨', '.html': '🌐', '.epub': '📚',
    }
    return mapping.get(ext, '📄')

def _avatar_idle_html(label: str = "Waiting for a question") -> str:
    """Idle avatar placeholder shown when there's no video yet (no spinning ring)."""
    return f"""
    <div class="aria-avatar-panel">
        <div class="aria-avatar-idle-wrap">
            <div class="aria-avatar-wrap" style="position:relative;">
                <div class="aria-avatar-glow"></div>
                <div class="aria-avatar-ring">
                    <div class="aria-avatar-inner">⬡</div>
                </div>
            </div>
            <div class="aria-avatar-status-text">{label}</div>
        </div>
    </div>"""

def _avatar_thinking_html(label: str = "ARIA is thinking…") -> str:
    """Spinning avatar shown while the answer is being generated / TTS'd / rendered."""
    return f"""
    <div class="aria-avatar-panel">
        <div class="aria-avatar-idle-wrap">
            <div class="aria-avatar-wrap" style="position:relative;">
                <div class="aria-avatar-glow"></div>
                <div class="aria-avatar-ring thinking">
                    <div class="aria-avatar-inner">⬡</div>
                </div>
            </div>
            <div class="aria-avatar-status-text">{label}</div>
        </div>
    </div>"""

def _avatar_video_html(video_b64: str) -> str:
    """Avatar panel showing the generated talking-head video, autoplaying."""
    return f"""
    <div class="aria-avatar-panel">
        <div class="aria-avatar-label">⬡ ARIA Avatar</div>
        <video autoplay playsinline controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    </div>"""

# ─── Layout: Left sidebar + main column ──────────────────────────────────────
sidebar_col, main_col = st.columns([1, 2.8], gap="small")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with sidebar_col:
    # ── Topbar (within sidebar) ──────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:0 0 24px 0;border-bottom:1px solid var(--aria-border);margin-bottom:24px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,var(--aria-purple),var(--aria-teal));clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);flex-shrink:0;"></div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:16px;color:var(--aria-text);letter-spacing:-.3px;">ARIA</div>
            <div style="font-size:10px;color:var(--aria-muted);letter-spacing:.8px;text-transform:uppercase;">AI Research Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status indicator ──────────────────────────────────────────────
    current_doc = st.session_state.get("current_doc")
    if current_doc:
        status_html = f"""
        <div style="display:flex;align-items:center;gap:8px;background:var(--aria-teal-dim);border:1px solid rgba(42,200,160,.22);border-radius:10px;padding:10px 12px;margin-bottom:4px;">
            <div style="width:7px;height:7px;border-radius:50%;background:var(--aria-teal);flex-shrink:0;"></div>
            <div style="font-size:12px;color:var(--aria-teal);font-weight:500;">Knowledge base active</div>
        </div>"""
    else:
        status_html = """
        <div style="display:flex;align-items:center;gap:8px;background:rgba(240,171,64,.08);border:1px solid rgba(240,171,64,.2);border-radius:10px;padding:10px 12px;margin-bottom:4px;">
            <div style="width:7px;height:7px;border-radius:50%;background:var(--aria-gold);flex-shrink:0;"></div>
            <div style="font-size:12px;color:var(--aria-gold);font-weight:500;">No document loaded</div>
        </div>"""
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Document Upload ───────────────────────────────────────────────
    st.markdown('<div class="aria-section-label">📂 Knowledge Source</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload document",
        type=SUPPORTED_EXTENSIONS,
        label_visibility="collapsed",
    )

    if uploaded_file:
        save_path = Path("data/docs") / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if st.session_state.get("current_doc") != uploaded_file.name:
            if st.session_state.get("current_doc"):
                clear_cache(st.session_state["current_doc"])
            st.session_state.current_doc = uploaded_file.name
            st.session_state.messages = []

        if is_indexed(uploaded_file.name):
            emoji = get_file_emoji(uploaded_file.name)
            size_kb = len(uploaded_file.getvalue()) // 1024
            st.markdown(f"""
            <div class="aria-doc-card">
                <div class="aria-doc-icon">{emoji}</div>
                <div>
                    <div class="aria-doc-name">{uploaded_file.name}</div>
                    <div class="aria-doc-meta">Indexed · {size_kb} KB</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Indexing document..."):
                count = ingest_document(str(save_path))
            if count > 0:
                emoji = get_file_emoji(uploaded_file.name)
                st.markdown(f"""
                <div class="aria-doc-card">
                    <div class="aria-doc-icon">{emoji}</div>
                    <div>
                        <div class="aria-doc-name">{uploaded_file.name}</div>
                        <div class="aria-doc-meta">✓ {count} chunks indexed</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.error("Could not extract text from document.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────
    st.markdown('<div class="aria-section-label">⚙ Session</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("✕ End & Delete", use_container_width=True):
            delete_current_doc()
            st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Voice Input ───────────────────────────────────────────────────
    st.markdown('<div class="aria-section-label">🎙 Voice Input</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:var(--aria-muted);margin-bottom:8px;">Record a question to ask ARIA</div>', unsafe_allow_html=True)
    audio_input = st.audio_input("Voice", label_visibility="collapsed")

    # ── Conversation stats ─────────────────────────────────────────────
    if st.session_state.messages:
        msg_count = len(st.session_state.messages)
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.markdown(f"""
        <div style="margin-top:24px;padding:14px 16px;background:var(--aria-card);border:1px solid var(--aria-border);border-radius:var(--aria-radius);">
            <div class="aria-section-label" style="margin-bottom:12px;">📊 Session Stats</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="text-align:center;padding:10px;background:var(--aria-purple-dim);border-radius:8px;border:1px solid rgba(124,114,216,.15);">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:var(--aria-purple);">{user_msgs}</div>
                    <div style="font-size:10px;color:var(--aria-muted);margin-top:2px;">Questions</div>
                </div>
                <div style="text-align:center;padding:10px;background:var(--aria-teal-dim);border-radius:8px;border:1px solid rgba(42,200,160,.15);">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:var(--aria-teal);">{user_msgs}</div>
                    <div style="font-size:10px;color:var(--aria-muted);margin-top:2px;">Answers</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
with main_col:
    # ── Top ribbon ───────────────────────────────────────────────────
    doc_label = f"<span style='color:var(--aria-teal);'>⬡ {current_doc}</span>" if current_doc else "<span style='color:var(--aria-muted);'>No document loaded</span>"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--aria-border);background:var(--aria-surface);">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;color:var(--aria-text);">Chat</div>
            <div style="width:1px;height:16px;background:var(--aria-border);"></div>
            <div style="font-size:12.5px;">{doc_label}</div>
        </div>
        <div style="font-size:11px;color:var(--aria-muted);">Voice · Text · Avatar</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # Chat history or Welcome screen
    # ─────────────────────────────────────────────────────────────────
    if not st.session_state.messages:
        # ── Welcome (capability grid removed) ───────────────────────
        st.markdown("""
        <div class="aria-welcome">
            <div class="aria-welcome-hex">⬡</div>
            <div class="aria-welcome-title">Hello, I'm ARIA</div>
            <div class="aria-welcome-sub">
                Your AI Research Intelligence Assistant. Upload a document
                from the sidebar, then ask me anything — by voice or text.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render past messages ───────────────────────────────────
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="aria-msg-user">
                    <div class="aria-bubble-user">{msg["content"]}</div>
                </div>
                <div style="height:4px"></div>
                """, unsafe_allow_html=True)
            else:
                video_b64 = msg.get("video_b64")
                avatar_html = _avatar_video_html(video_b64) if video_b64 else _avatar_idle_html("No avatar for this answer")
                st.markdown(f"""
                <div class="aria-answer-row">
                    {avatar_html}
                    <div class="aria-msg-bot">
                        <div class="aria-bubble-bot">{msg["content"]}</div>
                    </div>
                </div>
                <div style="height:16px"></div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # Voice transcription
    # ─────────────────────────────────────────────────────────────────
    voice_query = ""
    if audio_input:
        with st.spinner("Transcribing..."):
            audio_bytes = bytes(audio_input.read())
            voice_query = transcribe_bytes(audio_bytes)
        if voice_query:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;background:var(--aria-purple-dim);border:1px solid rgba(124,114,216,.25);border-radius:10px;padding:10px 14px;margin:8px 0;font-size:13px;color:var(--aria-purple);">
                <span>🎙</span> <em>"{voice_query}"</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Could not transcribe audio. Please try again.")

    # ── Chat text input ─────────────────────────────────────────────
    text_query = st.chat_input("Ask ARIA anything about your document…")

    # ─────────────────────────────────────────────────────────────────
    # Process query
    # ─────────────────────────────────────────────────────────────────
    query = voice_query or text_query
    current_doc = st.session_state.get("current_doc")

    if query:
        if not current_doc:
            st.markdown("""
            <div style="background:rgba(240,171,64,.08);border:1px solid rgba(240,171,64,.25);border-radius:12px;padding:14px 18px;font-size:13.5px;color:var(--aria-gold);margin-top:8px;">
                ⚠ Please upload a document from the sidebar before asking questions.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": query})

            # Render user bubble
            st.markdown(f"""
            <div class="aria-msg-user">
                <div class="aria-bubble-user">{query}</div>
            </div>
            <div style="height:12px"></div>
            """, unsafe_allow_html=True)

            # ── Combined answer row placeholder: avatar (left) + text (right) ──
            # The avatar auto-appears here by default the moment a question
            # is asked — no button or command needed.
            answer_row_placeholder = st.empty()
            answer_row_placeholder.markdown(f"""
            <div class="aria-answer-row">
                {_avatar_thinking_html("ARIA is thinking…")}
                <div class="aria-msg-bot">
                    <div class="aria-thinking">
                        <div class="aria-thinking-label">ARIA is thinking</div>
                        <div class="aria-thinking-dots">
                            <div class="aria-thinking-dot"></div>
                            <div class="aria-thinking-dot"></div>
                            <div class="aria-thinking-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Streaming answer ────────────────────────────────────
            full_answer = ""
            buffer      = ""
            phrase_idx  = 0
            all_audio_chunks = []

            tts_queue    = queue.Queue()
            result_queue = queue.Queue()
            tts_thread   = threading.Thread(target=tts_worker, args=(tts_queue, result_queue), daemon=True)
            tts_thread.start()

            def _drain_ready_audio():
                while True:
                    try:
                        item = result_queue.get_nowait()
                        if item is None:
                            return True
                        idx, audio_bytes = item
                        if audio_bytes:
                            _push_audio_to_browser(audio_bytes)
                            all_audio_chunks.append(audio_bytes)
                    except queue.Empty:
                        break
                return False

            tts_done = False

            for token in answer_query_stream(query, st.session_state.messages, current_doc):
                full_answer += token
                buffer      += token

                # Once we have some content, switch from "thinking" to
                # streaming text, with the avatar still active alongside it.
                if len(full_answer) > 3:
                    answer_row_placeholder.markdown(f"""
                    <div class="aria-answer-row">
                        {_avatar_thinking_html("ARIA is speaking…")}
                        <div class="aria-msg-bot">
                            <div class="aria-bubble-bot">{full_answer}<span style="display:inline-block;width:2px;height:14px;background:var(--aria-purple);margin-left:2px;animation:cursor-blink .8s ease-in-out infinite;vertical-align:middle;border-radius:1px;"></span></div>
                        </div>
                    </div>
                    <style>@keyframes cursor-blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}</style>
                    """, unsafe_allow_html=True)

                if "\n" in buffer:
                    parts = buffer.split("\n")
                    for part in parts[:-1]:
                        if part.strip():
                            phrase_idx = _flush_phrase(part, phrase_idx, tts_queue)
                    buffer = parts[-1]
                elif re.search(r'[.!?]\s*$', buffer.strip()):
                    phrase_idx = _flush_phrase(buffer, phrase_idx, tts_queue)
                    buffer     = ""
                elif re.search(r',\s*$', buffer.strip()):
                    if len(buffer.strip()) >= MIN_PHRASE_CHARS:
                        phrase_idx = _flush_phrase(buffer, phrase_idx, tts_queue)
                        buffer     = ""

                if not tts_done:
                    tts_done = _drain_ready_audio()

            if buffer.strip():
                phrase_idx = _flush_phrase(buffer, phrase_idx, tts_queue)

            tts_queue.put(None)

            # ── Drain remaining audio ────────────────────────────────
            if not tts_done:
                while True:
                    item = result_queue.get()
                    if item is None:
                        break
                    idx, audio_bytes = item
                    if audio_bytes:
                        _push_audio_to_browser(audio_bytes)
                        all_audio_chunks.append(audio_bytes)

            tts_thread.join()

            # ── Auto-generate the avatar video for this answer ──────
            # No command/button required — this always runs once the
            # full answer + its audio are ready.
            video_b64 = None
            if all_audio_chunks:
                answer_row_placeholder.markdown(f"""
                <div class="aria-answer-row">
                    {_avatar_thinking_html("Generating ARIA avatar…")}
                    <div class="aria-msg-bot">
                        <div class="aria-bubble-bot">{full_answer}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                combined_mp3 = b"".join(all_audio_chunks)
                try:
                    video_bytes = generate_avatar_video(combined_mp3)
                except Exception as e:
                    logger.warning("Avatar generation failed: {}", e)
                    video_bytes = None

                if video_bytes:
                    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

            # ── Final render: avatar (left, video if available) + text (right) ──
            final_avatar_html = _avatar_video_html(video_b64) if video_b64 else _avatar_idle_html("Avatar unavailable for this answer")
            answer_row_placeholder.markdown(f"""
            <div class="aria-answer-row">
                {final_avatar_html}
                <div class="aria-msg-bot">
                    <div class="aria-bubble-bot">{full_answer}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Sources ──────────────────────────────────────────────
            result  = answer_query(query, st.session_state.messages, current_doc)
            sources = result.get("sources", [])
            if sources:
                chips = ""
                for src in sources:
                    chips += f'<span class="aria-source-chip">📄 p.{src["page"]} · {round(src["score"]*100)}%</span>'
                details_html = ""
                for src in sources:
                    snippet = src["content"][:180] + ("…" if len(src["content"]) > 180 else "")
                    details_html += f"""
                    <div style="background:var(--aria-card);border:1px solid var(--aria-border);border-radius:10px;padding:12px 14px;margin-top:8px;">
                        <div style="font-size:11px;font-weight:600;color:var(--aria-teal);margin-bottom:4px;">📄 {src['source']} · Page {src['page']} · Score {round(src['score']*100)}%</div>
                        <div style="font-size:12.5px;color:var(--aria-muted);line-height:1.6;">{snippet}</div>
                    </div>"""
                with st.expander(f"📚 {len(sources)} source{'s' if len(sources)>1 else ''} cited"):
                    st.markdown(details_html, unsafe_allow_html=True)

            # ── Save to history (video stored as base64 so it replays) ──
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "video_b64": video_b64,
            })