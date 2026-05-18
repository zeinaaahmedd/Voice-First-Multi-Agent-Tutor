"""
app.py  –  Streamlit UI for the Voice-First Multi-Agent Tutor (بصيرة)

Run:
    streamlit run app.py

Dependencies (add to requirements if missing):
    pip install streamlit audio-recorder-streamlit
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import ffmpeg_utils
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# ── force UTF-8 so Arabic prints correctly on Windows ────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_FFMPEG = ffmpeg_utils.FFMPEG  # usable ffmpeg.exe path (PATH already patched)

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="بصيرة — المعلم الصوتي",
    page_icon="🎓",
    layout="centered",
)

# ── RTL + minimal style fixes ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Right-to-left for all chat bubbles */
        .stChatMessage, .stChatMessage p, .stChatMessage div {
            direction: rtl;
            text-align: right;
        }
        /* Keep audio player LTR */
        .stAudio { direction: ltr; }
        /* Tighten the recorder button area */
        .recorder-row { display: flex; align-items: center; gap: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── cached orchestrator (loaded once per server process) ─────────────────────
@st.cache_resource(show_spinner="جاري تحميل النظام، ارجع لحظة…")
def load_orchestrator():
    # Add project root to path so local modules resolve correctly
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change CWD so relative file loads inside agents work
    os.chdir(project_root)

    from orchestrator import TutorOrchestrator
    return TutorOrchestrator(enable_tts=True)


# ── session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None          # orchestrator assigns on first turn

if "messages" not in st.session_state:
    st.session_state["messages"] = []              # list of {"role", "text", "audio"}

if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = None     # dedup guard — avoids double-processing


# ── header ────────────────────────────────────────────────────────────────────
st.title("🎓 بصيرة — المعلم التعليمي الصوتي")
st.caption("نظام تعليمي صوتي بالعربي المصري | اضغط على الميكروفون وابدأ السؤال")
st.divider()


# ── chat history display ──────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    role_label = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role_label):
        st.markdown(msg["text"])
        if msg.get("audio") and Path(msg["audio"]).exists():
            st.audio(msg["audio"], format="audio/wav", autoplay=False)


# ── recording row ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 🎙️ سجّل سؤالك")

col_mic, col_hint = st.columns([1, 5])
with col_mic:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e53935",
        neutral_color="#1976d2",
        icon_size="2x",
        pause_threshold=2.5,   # stop after 2.5 s of silence
    )
with col_hint:
    st.markdown(
        "<small style='color:grey'>اضغط للتسجيل ← وقف التسجيل تلقائياً بعد ثانيتين من الصمت</small>",
        unsafe_allow_html=True,
    )


# ── process new recording ─────────────────────────────────────────────────────
if audio_bytes:
    # Dedup: ignore if this exact recording was already processed
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if audio_hash != st.session_state["last_audio_hash"]:
        st.session_state["last_audio_hash"] = audio_hash

        # The browser records in WebM/Opus (Chrome) or MP4/AAC (Safari).
        # Save the raw bytes without assuming an extension, then let ffmpeg
        # transcode to 16 kHz mono s16le WAV that whisperx expects.
        tmp_dir  = Path(tempfile.gettempdir())
        raw_path = tmp_dir / "basira_raw"
        wav_path = tmp_dir / "basira_input.wav"
        raw_path.write_bytes(audio_bytes)
        try:
            subprocess.run(
                [
                    _FFMPEG, "-y", "-i", str(raw_path),
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as conv_err:
            st.error("فشل تحويل الصوت — " + conv_err.stderr.decode(errors="replace")[-300:])
            st.stop()
        tmp_path = str(wav_path)

        with st.spinner("بصيرة بتفكر…"):
            try:
                orchestrator = load_orchestrator()
                result = orchestrator.process_audio(
                    audio_path=tmp_path,
                    session_id=st.session_state["session_id"],
                )
            except Exception as exc:
                st.error(f"حصل خطأ: {exc}")
                st.code(traceback.format_exc(), language="python")
                st.stop()

        # Persist the session ID so the orchestrator keeps context
        st.session_state["session_id"] = result["session_id"]

        # Append user turn
        transcription = result.get("transcription", "").strip()
        if transcription:
            st.session_state["messages"].append({
                "role": "user",
                "text": transcription,
                "audio": None,
            })

        # Append assistant turn
        response_text = result.get("response_text", "")
        audio_output  = result.get("audio_output")
        st.session_state["messages"].append({
            "role": "assistant",
            "text": response_text,
            "audio": audio_output,
        })

        # Rerun to refresh the chat display with the new messages
        st.rerun()


# ── text fallback input (useful when microphone isn't available) ──────────────
with st.expander("⌨️ أو اكتب سؤالك نصياً"):
    text_input = st.text_input(
        "اكتب هنا:",
        key="text_input_box",
        label_visibility="collapsed",
        placeholder="مثال: انا عايز افهم النظام البيئي",
    )
    send_btn = st.button("إرسال", use_container_width=True)

    if send_btn and text_input.strip():
        with st.spinner("بصيرة بتفكر…"):
            try:
                orchestrator = load_orchestrator()
                result = orchestrator.process_text(
                    text=text_input.strip(),
                    session_id=st.session_state["session_id"],
                )
            except Exception as exc:
                st.error(f"حصل خطأ: {exc}")
                st.code(traceback.format_exc(), language="python")
                st.stop()

        st.session_state["session_id"] = result["session_id"]

        st.session_state["messages"].append({
            "role": "user",
            "text": text_input.strip(),
            "audio": None,
        })
        st.session_state["messages"].append({
            "role": "assistant",
            "text": result.get("response_text", ""),
            "audio": result.get("audio_output"),
        })
        st.rerun()


# ── sidebar: session info + reset ─────────────────────────────────────────────
with st.sidebar:
    st.header("معلومات الجلسة")

    sid = st.session_state.get("session_id")
    if sid:
        st.success(f"جلسة نشطة ✅")
        st.code(sid[:8] + "…", language=None)
    else:
        st.info("لم تبدأ جلسة بعد")

    turn_count = len([m for m in st.session_state["messages"] if m["role"] == "user"])
    st.metric("عدد الأسئلة", turn_count)

    st.divider()
    st.markdown("**الأوامر الصوتية:**")
    st.markdown(
        "- سؤال عادي → تدريس\n"
        "- 'لخص' → ملخص المحاضرة\n"
        "- 'كويز' → أسئلة اختبار"
    )

    st.divider()
    if st.button("🔄 بدء جلسة جديدة", use_container_width=True):
        for key in ("session_id", "messages", "last_audio_hash"):
            st.session_state[key] = None if key != "messages" else []
        st.rerun()
