"""
BASIRA – Arabic Voice Tutor
Phase 1 · Audio Ingestion UI (Gradio)

Accessibility-first design for visually impaired learners:
  • Giant centered press-to-speak button
  • Hover audio cue (place your guide audio file at assets/hover_cue.mp3)
  • High-contrast, large-typography layout
  • Keyboard accessible (Space / Enter to record)
  • Screen-reader friendly aria labels

Run:
    pip install gradio>=4.0 numpy soundfile
    python app.py
"""

import os
import tempfile
import numpy as np
import gradio as gr
from pathlib import Path
from audio_ingestion import ingest_audio   # Phase 1 module (same directory)

# ── Assets ──────────────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent / "assets"
HOVER_AUDIO = ASSETS_DIR / "hover_cue.mp3"          # <── drop your audio file here
HOVER_AUDIO_SRC = str(HOVER_AUDIO) if HOVER_AUDIO.exists() else None

# ── Core callback ────────────────────────────────────────────────────────────
def process_voice(audio_tuple):
    """
    Gradio Audio component (type='numpy') returns (sample_rate, np.ndarray).
    We pass it through the ingestion pipeline and return a status message.
    """
    if audio_tuple is None:
        return "⚠️  لم يتم التقاط صوت. حاول مرة أخرى.", None

    sample_rate, audio_array = audio_tuple

    # Ingest → validated numpy array ready for WhisperX
    audio_np, meta = ingest_audio(audio_array, sample_rate)

    status = (
        f"✅  تم استلام الصوت بنجاح\n"
        f"المدة: {meta['duration_sec']:.1f} ثانية  |  "
        f"معدل الأخذ: {meta['sample_rate']} Hz  |  "
        f"الشكل: {meta['shape']}"
    )
    return status, audio_np   # audio_np passed to downstream STT agent


# ── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Google Font: Tajawal (Arabic-friendly) + Syne (display) ── */
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&family=Syne:wght@700;800&display=swap');

:root {
    --bg:          #0a0a0f;
    --surface:     #12121a;
    --border:      #1e1e2e;
    --accent:      #00e5a0;
    --accent-dim:  #00b87c;
    --accent-glow: rgba(0, 229, 160, 0.28);
    --text-hi:     #f0f0f5;
    --text-lo:     #6b6b80;
    --danger:      #ff4d6d;
    --radius:      24px;
    --btn-size:    340px;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text-hi) !important;
    font-family: 'Tajawal', sans-serif !important;
    min-height: 100vh;
}

/* ── Page scaffold ── */
.basira-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 40px 24px;
    gap: 0;
    position: relative;
    overflow: hidden;
}

/* Ambient background rings */
.basira-wrap::before {
    content: '';
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 700px; height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,229,160,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.basira-header {
    text-align: center;
    margin-bottom: 56px;
    position: relative;
    z-index: 1;
}

.basira-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.4rem, 5vw, 3.6rem) !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    color: var(--text-hi) !important;
    line-height: 1.1 !important;
}

.basira-title span {
    color: var(--accent);
}

.basira-subtitle {
    font-size: 1.15rem !important;
    color: var(--text-lo) !important;
    margin-top: 10px !important;
    font-weight: 400 !important;
    direction: rtl;
}

/* ── Central button zone ── */
.btn-zone {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 32px;
}

/* Pulse rings behind button */
.pulse-ring {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    border: 2px solid var(--accent);
    opacity: 0;
    pointer-events: none;
}
.pulse-ring:nth-child(1) { width: 380px; height: 380px; animation: pulse 2.8s ease-out infinite; }
.pulse-ring:nth-child(2) { width: 460px; height: 460px; animation: pulse 2.8s ease-out infinite 0.7s; }
.pulse-ring:nth-child(3) { width: 540px; height: 540px; animation: pulse 2.8s ease-out infinite 1.4s; }

@keyframes pulse {
    0%   { opacity: 0.6; transform: translate(-50%, -50%) scale(0.85); }
    100% { opacity: 0;   transform: translate(-50%, -50%) scale(1.2); }
}

/* ── THE BUTTON ── */
#speak-btn {
    width:  var(--btn-size) !important;
    height: var(--btn-size) !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 38% 35%, #1a2e28, #0d1f1a 70%) !important;
    border: 3px solid var(--accent) !important;
    box-shadow:
        0 0 0 1px rgba(0,229,160,0.12),
        0 0 60px var(--accent-glow),
        inset 0 1px 0 rgba(0,229,160,0.15) !important;
    cursor: pointer !important;
    transition:
        transform 0.18s cubic-bezier(.34,1.56,.64,1),
        box-shadow 0.18s ease,
        background 0.18s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    position: relative !important;
    outline: none !important;
    -webkit-tap-highlight-color: transparent !important;
}

/* Recording state */
#speak-btn.recording {
    background: radial-gradient(circle at 38% 35%, #2e1a1a, #1f0d0d 70%) !important;
    border-color: var(--danger) !important;
    box-shadow:
        0 0 0 1px rgba(255,77,109,0.18),
        0 0 80px rgba(255,77,109,0.35),
        inset 0 1px 0 rgba(255,77,109,0.15) !important;
    animation: record-throb 1s ease-in-out infinite !important;
}

@keyframes record-throb {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.025); }
}

#speak-btn:hover:not(.recording) {
    transform: scale(1.06) !important;
    box-shadow:
        0 0 0 1px rgba(0,229,160,0.22),
        0 0 90px rgba(0,229,160,0.45),
        inset 0 1px 0 rgba(0,229,160,0.2) !important;
}

#speak-btn:focus-visible {
    outline: 4px solid var(--accent) !important;
    outline-offset: 8px !important;
}

#speak-btn:active:not(.recording) {
    transform: scale(0.97) !important;
}

/* ── Mic icon inside button ── */
.mic-icon {
    font-size: 72px !important;
    line-height: 1 !important;
    transition: transform 0.2s ease !important;
    filter: drop-shadow(0 0 16px var(--accent)) !important;
    user-select: none !important;
}

.recording .mic-icon {
    filter: drop-shadow(0 0 20px var(--danger)) !important;
    animation: mic-bounce 0.6s ease-in-out infinite alternate !important;
}

@keyframes mic-bounce {
    from { transform: scale(1);    }
    to   { transform: scale(1.1);  }
}

.btn-label {
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: color 0.2s !important;
    user-select: none !important;
    direction: rtl !important;
}

.recording .btn-label {
    color: var(--danger) !important;
}

/* ── Status bar ── */
.status-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px 28px !important;
    font-size: 1rem !important;
    color: var(--text-lo) !important;
    text-align: center !important;
    direction: rtl !important;
    min-width: 420px !important;
    max-width: 560px !important;
    min-height: 72px !important;
    line-height: 1.6 !important;
    transition: color 0.3s, border-color 0.3s !important;
    position: relative;
    z-index: 1;
}

.status-box.ok  { color: var(--accent) !important; border-color: var(--accent-dim) !important; }
.status-box.err { color: var(--danger) !important; border-color: var(--danger) !important; }

/* ── Gradio Audio widget (hidden visually but accessible) ── */
#gradio-audio-input {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

/* ── Footer ── */
.basira-footer {
    position: fixed;
    bottom: 24px;
    font-size: 0.78rem;
    color: var(--text-lo);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    z-index: 1;
}
"""

# ── JS hover-audio + button orchestration ────────────────────────────────────
HOVER_AUDIO_JS = f"""
<script>
(function () {{
    // ── Hover audio ──────────────────────────────────────────────────────────
    const HOVER_AUDIO_SRC = {repr(HOVER_AUDIO_SRC or '')};
    let hoverAudio = null;

    if (HOVER_AUDIO_SRC) {{
        hoverAudio = new Audio(HOVER_AUDIO_SRC);
        hoverAudio.preload = 'auto';
    }}

    // ── State ────────────────────────────────────────────────────────────────
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let stream = null;

    function getBtn()    {{ return document.getElementById('speak-btn'); }}
    function getMicIcon(){{ return document.querySelector('.mic-icon'); }}
    function getBtnLabel(){{ return document.querySelector('.btn-label'); }}

    // ── Hover: play guide audio ──────────────────────────────────────────────
    document.addEventListener('mouseover', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && btn.contains(e.target) && hoverAudio && !isRecording) {{
            hoverAudio.currentTime = 0;
            hoverAudio.play().catch(() => {{}});
        }}
    }});

    document.addEventListener('mouseout', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && btn.contains(e.target) && hoverAudio) {{
            hoverAudio.pause();
            hoverAudio.currentTime = 0;
        }}
    }});

    // ── Button click / keyboard ──────────────────────────────────────────────
    document.addEventListener('click', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && btn.contains(e.target)) toggleRecording();
    }});

    document.addEventListener('keydown', function(e) {{
        if ((e.code === 'Space' || e.code === 'Enter') && !e.repeat) {{
            const focused = document.activeElement;
            if (focused && focused.id === 'speak-btn') {{
                e.preventDefault();
                toggleRecording();
            }}
        }}
    }});

    // ── Recording logic ──────────────────────────────────────────────────────
    async function toggleRecording() {{
        if (!isRecording) {{
            await startRecording();
        }} else {{
            stopRecording();
        }}
    }}

    async function startRecording() {{
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {{
                if (e.data.size > 0) audioChunks.push(e.data);
            }};

            mediaRecorder.onstop = handleRecordingStop;
            mediaRecorder.start(100);
            isRecording = true;
            setRecordingState(true);
        }} catch (err) {{
            console.error('Microphone error:', err);
            updateStatus('❌  تعذّر الوصول إلى الميكروفون. تأكد من منح الإذن.', 'err');
        }}
    }}

    function stopRecording() {{
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
            mediaRecorder.stop();
        }}
        if (stream) {{
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }}
        isRecording = false;
        setRecordingState(false);
    }}

    async function handleRecordingStop() {{
        const blob = new Blob(audioChunks, {{ type: 'audio/webm' }});

        // Push to Gradio's hidden audio input via FileReader → DataTransfer trick
        const file = new File([blob], 'recording.webm', {{ type: 'audio/webm' }});

        // Find Gradio's upload input (inside the hidden component)
        const inputs = document.querySelectorAll('input[type="file"]');
        for (const input of inputs) {{
            if (input.accept && input.accept.includes('audio')) {{
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                break;
            }}
        }}

        // Also expose a global for direct pipeline access
        window.BASIRA_LAST_AUDIO_BLOB = blob;
        window.BASIRA_LAST_AUDIO_URL  = URL.createObjectURL(blob);

        updateStatus('⏳  جارٍ معالجة الصوت…', '');
    }}

    // ── UI helpers ───────────────────────────────────────────────────────────
    function setRecordingState(recording) {{
        const btn = getBtn();
        const icon = getMicIcon();
        const label = getBtnLabel();
        if (!btn) return;

        if (recording) {{
            btn.classList.add('recording');
            btn.setAttribute('aria-label', 'إيقاف التسجيل');
            if (icon)  icon.textContent  = '⏹';
            if (label) label.textContent = 'اضغط للإيقاف';
            updateStatus('🔴  جارٍ التسجيل…', '');
        }} else {{
            btn.classList.remove('recording');
            btn.setAttribute('aria-label', 'اضغط للتحدث');
            if (icon)  icon.textContent  = '🎙️';
            if (label) label.textContent = 'اضغط للتحدث';
        }}
    }}

    function updateStatus(msg, cls) {{
        const box = document.querySelector('.status-box');
        if (!box) return;
        box.textContent = msg;
        box.className = 'status-box ' + (cls || '');
    }}

    // ── Init ─────────────────────────────────────────────────────────────────
    // Wait for DOM to settle then wire up the button aria
    const observer = new MutationObserver(() => {{
        const btn = document.getElementById('speak-btn');
        if (btn) {{
            btn.setAttribute('tabindex', '0');
            btn.setAttribute('role', 'button');
            btn.setAttribute('aria-label', 'اضغط للتحدث');
            observer.disconnect();
        }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""

# ── Build Gradio interface ────────────────────────────────────────────────────
with gr.Blocks(
    title="BASIRA – المعلم الصوتي العربي",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as demo:

    # Inject hover-audio + recording JS
    gr.HTML(HOVER_AUDIO_JS)

    with gr.Column(elem_classes="basira-wrap"):

        # ── Header ──
        gr.HTML("""
        <div class="basira-header">
            <div class="basira-title">BASIRA &nbsp;<span>بصيرة</span></div>
            <div class="basira-subtitle">المعلم الصوتي للغة العربية · نظام التعرف على الكلام</div>
        </div>
        """)

        # ── Pulse rings + big button ──
        gr.HTML("""
        <div class="btn-zone">
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <button
                id="speak-btn"
                aria-label="اضغط للتحدث"
                aria-live="polite"
                tabindex="0">
                <span class="mic-icon" aria-hidden="true">🎙️</span>
                <span class="btn-label">اضغط للتحدث</span>
            </button>
        </div>
        """)

        # ── Hidden Gradio audio recorder (accessibility: still present in DOM) ──
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="تسجيل صوتي",
            elem_id="gradio-audio-input",
        )

        # ── Status output ──
        status_out = gr.Textbox(
            value="مرحباً. اضغط الزر للبدء.",
            label="",
            elem_classes="status-box",
            interactive=False,
            show_label=False,
        )

        # ── Hidden output for pipeline (numpy array passed downstream) ──
        # In the full multi-agent graph, stt_agent.py reads from this state slot.
        pipeline_audio = gr.State(None)

        # Wire up Gradio's own audio component as fallback
        audio_input.change(
            fn=process_voice,
            inputs=[audio_input],
            outputs=[status_out, pipeline_audio],
        )

        gr.HTML('<div class="basira-footer">BASIRA v0.1 · Phase 1 · Audio Ingestion</div>')


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )