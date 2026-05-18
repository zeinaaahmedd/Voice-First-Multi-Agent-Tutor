"""
BASIRA – Arabic Voice Tutor
Phase 1 · Audio Ingestion UI (Gradio)

Run:
    pip install gradio numpy soundfile scipy
    python app.py
"""

import numpy as np
import gradio as gr
from pathlib import Path
from audio_ingestion import ingest_audio

# ── Assets ───────────────────────────────────────────────────────────────────
ASSETS_DIR  = Path(__file__).parent / "assets"
HOVER_AUDIO = ASSETS_DIR / "hover_cue.mp3"

# Gradio serves local files at /file=<absolute_path> when allowed_paths is set.
# This is the URL the BROWSER can actually fetch — filesystem paths don't work.
HOVER_AUDIO_URL = f"/file={HOVER_AUDIO.resolve()}" if HOVER_AUDIO.exists() else ""

# ── Core callback ─────────────────────────────────────────────────────────────
def process_voice(audio_tuple):
    if audio_tuple is None:
        return "⚠️  لم يتم التقاط صوت — حاول مرة أخرى", None, gr.update(visible=False)

    sample_rate, audio_array = audio_tuple

    try:
        audio_np, meta = ingest_audio(audio_array, sample_rate)
    except ValueError as e:
        return f"❌  {e}", None, gr.update(visible=False)

    warnings_str = " | ".join(meta["warnings"]) if meta["warnings"] else "لا تنبيهات"

    status = (
        f"✅  تم استلام الصوت بنجاح\n"
        f"المدة: {meta['duration_sec']:.1f} ثانية  ·  "
        f"الجودة (RMS): {meta['rms']:.4f}  ·  "
        f"التردد: {meta['sample_rate']} Hz\n"
        f"التنبيهات: {warnings_str}"
    )

    debug_info = (
        f"Shape      : {meta['shape']}\n"
        f"Original SR: {meta['original_sr']} Hz  →  resampled to {meta['sample_rate']} Hz\n"
        f"Pipeline   : Ready to pass to STT Agent ✅"
    )

    return status, audio_np, gr.update(value=debug_info, visible=True)


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
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

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text-hi) !important;
    font-family: 'Tajawal', sans-serif !important;
    min-height: 100vh;
}

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

.basira-title span { color: var(--accent); }

.basira-subtitle {
    font-size: 1.15rem !important;
    color: var(--text-lo) !important;
    margin-top: 10px !important;
    font-weight: 400 !important;
    direction: rtl;
}

.btn-zone {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
}

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
    transition: transform 0.18s cubic-bezier(.34,1.56,.64,1), box-shadow 0.18s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    position: relative !important;
    outline: none !important;
    -webkit-tap-highlight-color: transparent !important;
}

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

#speak-btn:active:not(.recording) { transform: scale(0.97) !important; }

.mic-icon {
    font-size: 72px !important;
    line-height: 1 !important;
    filter: drop-shadow(0 0 16px var(--accent)) !important;
    user-select: none !important;
    transition: transform 0.2s ease !important;
}

.recording .mic-icon {
    filter: drop-shadow(0 0 20px var(--danger)) !important;
    animation: mic-bounce 0.6s ease-in-out infinite alternate !important;
}

@keyframes mic-bounce {
    from { transform: scale(1); }
    to   { transform: scale(1.1); }
}

.btn-label {
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: 0.04em !important;
    transition: color 0.2s !important;
    user-select: none !important;
    direction: rtl !important;
}

.recording .btn-label { color: var(--danger) !important; }

/* ── Welcome message — centered under button with breathing room ── */
.welcome-msg {
    margin-top: 44px !important;
    text-align: center !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
    direction: rtl !important;
    letter-spacing: 0.01em !important;
    z-index: 1;
    position: relative;
}

/* ── Status / debug boxes ── */
.status-wrap {
    width: 540px;
    max-width: 90vw;
    margin-top: 28px;
    position: relative;
    z-index: 1;
}

.status-box textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 24px !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.05rem !important;
    color: var(--text-lo) !important;
    text-align: center !important;
    direction: rtl !important;
    resize: none !important;
}

.debug-box textarea {
    background: #0d1f1a !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: 12px !important;
    color: var(--accent) !important;
    font-family: monospace !important;
    font-size: 0.82rem !important;
    padding: 12px 16px !important;
    direction: ltr !important;
    resize: none !important;
}

/* Hide Gradio chrome we don't need */
.status-wrap .label-wrap,
.debug-wrap  .label-wrap  { display: none !important; }

/* ── Hidden Gradio audio recorder ── */
#gradio-audio-input {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

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

# ── JS ────────────────────────────────────────────────────────────────────────
INTERACTION_JS = f"""
<script>
(function () {{
    // ── Hover audio
    // HOVER_AUDIO_URL is a /file=... URL served by Gradio's allowed_paths.
    // The browser can fetch this; a raw filesystem path cannot be fetched.
    const HOVER_SRC = {repr(HOVER_AUDIO_URL)};
    let hoverAudio = null;

    if (HOVER_SRC) {{
        hoverAudio = new Audio(HOVER_SRC);
        hoverAudio.preload = 'auto';
        // Browser autoplay policy: warm up on first user gesture
        document.addEventListener('click', () => hoverAudio.load(), {{ once: true }});
    }}

    let isRecording   = false;
    let mediaRecorder = null;
    let audioChunks   = [];
    let stream        = null;

    // ── Hover: play audio cue
    document.addEventListener('mouseover', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && (btn === e.target || btn.contains(e.target)) && hoverAudio && !isRecording) {{
            hoverAudio.currentTime = 0;
            hoverAudio.play().catch(err => console.warn('Hover audio:', err));
        }}
    }});

    document.addEventListener('mouseout', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && (btn === e.target || btn.contains(e.target)) && hoverAudio) {{
            hoverAudio.pause();
            hoverAudio.currentTime = 0;
        }}
    }});

    // ── Click / keyboard
    document.addEventListener('click', function(e) {{
        const btn = document.getElementById('speak-btn');
        if (btn && (btn === e.target || btn.contains(e.target))) toggleRecording();
    }});

    document.addEventListener('keydown', function(e) {{
        if ((e.code === 'Space' || e.code === 'Enter') && !e.repeat
            && document.activeElement?.id === 'speak-btn') {{
            e.preventDefault();
            toggleRecording();
        }}
    }});

    async function toggleRecording() {{
        isRecording ? stopRecording() : await startRecording();
    }}

    async function startRecording() {{
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.ondataavailable = e => {{ if (e.data.size > 0) audioChunks.push(e.data); }};
            mediaRecorder.onstop = handleStop;
            mediaRecorder.start(100);
            isRecording = true;
            setUI(true);
        }} catch(err) {{
            updateStatus('❌  تعذّر الوصول إلى الميكروفون — تأكد من منح الإذن');
            console.error(err);
        }}
    }}

    function stopRecording() {{
        if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
        stream?.getTracks().forEach(t => t.stop());
        stream = null;
        isRecording = false;
        setUI(false);
    }}

    async function handleStop() {{
        const blob = new File(
            [new Blob(audioChunks, {{type:'audio/webm'}})],
            'recording.webm', {{type:'audio/webm'}}
        );
        const inputs = document.querySelectorAll('input[type="file"]');
        for (const inp of inputs) {{
            if (inp.accept?.includes('audio') || inp.closest('#gradio-audio-input')) {{
                const dt = new DataTransfer();
                dt.items.add(blob);
                inp.files = dt.files;
                inp.dispatchEvent(new Event('change', {{bubbles: true}}));
                break;
            }}
        }}
        window.BASIRA_LAST_BLOB = blob;
        updateStatus('⏳  جارٍ معالجة الصوت…');
    }}

    function setUI(rec) {{
        const btn   = document.getElementById('speak-btn');
        const icon  = document.querySelector('.mic-icon');
        const label = document.querySelector('.btn-label');
        if (!btn) return;
        btn.classList.toggle('recording', rec);
        btn.setAttribute('aria-label', rec ? 'إيقاف التسجيل' : 'اضغط للتحدث');
        if (icon)  icon.textContent  = rec ? '⏹' : '🎙️';
        if (label) label.textContent = rec ? 'اضغط للإيقاف' : 'اضغط للتحدث';
        if (rec) updateStatus('🔴  جارٍ التسجيل…');
    }}

    function updateStatus(msg) {{
        document.querySelectorAll('.status-box textarea')
            .forEach(b => {{ b.value = msg; }});
    }}

    const obs = new MutationObserver(() => {{
        const btn = document.getElementById('speak-btn');
        if (btn) {{
            btn.setAttribute('tabindex', '0');
            btn.setAttribute('role', 'button');
            btn.setAttribute('aria-label', 'اضغط للتحدث');
            obs.disconnect();
        }}
    }});
    obs.observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""

# ── Gradio app ────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:

    gr.HTML(INTERACTION_JS)

    with gr.Column(elem_classes="basira-wrap"):

        gr.HTML("""
        <div class="basira-header">
            <div class="basira-title">BASIRA &nbsp;<span>بصيرة</span></div>
            <div class="basira-subtitle">المعلم الصوتي للغة العربية </div>
        </div>
        """)

        gr.HTML("""
        <div class="btn-zone">
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <button id="speak-btn" aria-label="اضغط للتحدث" tabindex="0">
                <span class="mic-icon" aria-hidden="true">🎙️</span>
                <span class="btn-label">اضغط للتحدث</span>
            </button>
        </div>
        """)

        # Welcome message — centered under button
        gr.HTML('<div class="welcome-msg">مرحباً ! اضغط الزر للتحدث مع المعلم</div>')

        # Hidden Gradio audio recorder (still in DOM for accessibility + pipeline feed)
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="تسجيل صوتي",
            elem_id="gradio-audio-input",
        )

        with gr.Column(elem_classes="status-wrap"):
            status_out = gr.Textbox(
                value="",
                elem_classes="status-box",
                interactive=False,
                show_label=False,
                lines=3,
            )

        with gr.Column(elem_classes="debug-wrap"):
            debug_out = gr.Textbox(
                value="",
                elem_classes="debug-box",
                interactive=False,
                show_label=False,
                lines=3,
                visible=False,
            )

        pipeline_audio = gr.State(None)

        audio_input.change(
            fn=process_voice,
            inputs=[audio_input],
            outputs=[status_out, pipeline_audio, debug_out],
        )

        gr.HTML('<div class="basira-footer">BASIRA v0.1 · Phase 1 · Audio Ingestion</div>')


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(ASSETS_DIR.resolve())],  # lets browser fetch hover_cue.mp3
        css=CUSTOM_CSS,                              # Gradio 6.0: css goes in launch()
    )