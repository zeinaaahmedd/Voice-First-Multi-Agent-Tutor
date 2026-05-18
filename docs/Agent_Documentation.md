# BASIRA: A Lightweight Arabic Voice Tutor for Visually Impaired Learners
## IEEE Technical Report — Phase 1 Draft: Audio Ingestion Layer

---

### II. SYSTEM ARCHITECTURE

The BASIRA system follows a multi-agent pipeline architecture:

    [User Voice] → [Audio Ingestion] → [STT Agent (WhisperX)]
        → [Pedagogical LLM Agent] → [Quiz/Summary Agents] → [TTS Agent]

This section documents the **Audio Ingestion Layer**, which constitutes the
system's acoustic interface.

---

### III. AUDIO INGESTION PIPELINE (Phase 1)

#### A. Design Rationale

Visually impaired users require a frictionless, audio-guided interface. We
chose a Gradio-based web UI over CLI or API-only access because:

1. **Zero installation burden** for end users.
2. **Browser MediaRecorder API** enables in-browser microphone capture without
   native OS audio drivers.
3. **Hover-audio guidance** provides auditory affordance — a short Arabic
   prompt plays on button hover, replacing visual tooltips.
4. **Keyboard accessibility**: the record button is navigable by Tab and
   activatable by Space/Enter, conforming to WCAG 2.1 AA.

#### B. UI Component Design

The interface centers on a single 340 × 340 px circular button. Design
decisions are accessibility-motivated:

| Decision | Rationale |
|---|---|
| Large touch target (340 px) | Reduces motor-accuracy demand |
| Single interaction element | Reduces cognitive load |
| Pulsing ring animation | Provides visual heartbeat without requiring it |
| Hover audio cue | Screen-reader-independent guidance |
| High-contrast palette (bg #0a0a0f, accent #00e5a0) | WCAG AA contrast ratio ≥ 7:1 |
| Arabic RTL typography (Tajawal font) | Native-language button labels |

State transitions:

    Idle ──(hover)──→ Audio cue plays
    Idle ──(click)──→ Recording [button turns red, mic-bounce animation]
    Recording ──(click)──→ Stop → audio passed to ingestion pipeline

#### C. Audio Ingestion Module (`audio_ingestion.py`)

The ingestion module normalizes any audio source to the format required by
WhisperX: **mono float32 NumPy array at 16 000 Hz**.

**Supported input sources:**

| Source Type | Origin |
|---|---|
| `(int, np.ndarray)` tuple | Gradio `Audio` component (primary path) |
| `np.ndarray` | Direct programmatic invocation |
| `str` / `Path` | Evaluation pipeline (benchmark files) |
| `bytes` / `bytearray` | Browser-streamed WebM/WAV |

**Processing pipeline:**

    Input → float32 cast → stereo→mono (mean of channels)
          → amplitude normalization (peak-normalize to [-1, 1])
          → resample to 16 kHz (scipy.signal.resample_poly; numpy fallback)
          → quality validation (duration, RMS silence check)
          → return (audio_np, metadata)

**Quality validation thresholds:**

| Check | Threshold | Action |
|---|---|---|
| Minimum duration | 0.5 s | Raise `ValueError` |
| Maximum duration | 300 s | Warn in metadata |
| RMS silence | < 1×10⁻⁴ | Warn in metadata |

**Resampling strategy:** `scipy.signal.resample_poly` with GCD-reduced
up/down factors is used when scipy is available, providing anti-aliased
resampling. A numpy linear-interpolation fallback ensures operation in
constrained environments.

**Metadata output (passed through the LangGraph state):**

```python
{
    "original_sr":  44100,           # Hz of source audio
    "sample_rate":  16000,           # Hz of output (always TARGET_SR)
    "duration_sec": 3.84,            # seconds
    "shape":        (61440,),        # samples
    "rms":          0.124631,        # RMS amplitude
    "warnings":     []               # list of non-fatal warnings
}
```

#### D. Integration Contract

The ingestion module exposes a single public function:

```python
audio_np, metadata = ingest_audio(source, sample_rate=None)
```

This contract is consumed by the **STT Agent** (Phase 3) as the
`audio_input` field of the shared `BASIRAState` LangGraph state.
The agent does not need to know the audio source type — all normalization
is handled transparently by `ingest_audio`.

---

### REFERENCES (Phase 1 additions)

[R1] Gradio Documentation, "Audio Component," Hugging Face, 2024.
     https://www.gradio.app/docs/gradio/audio

[R2] W3C, "Web Content Accessibility Guidelines (WCAG) 2.1," 2018.
     https://www.w3.org/TR/WCAG21/

[R3] Mozilla Developer Network, "MediaRecorder API," 2024.
     https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder

[R4] scipy.signal.resample_poly — SciPy Reference,
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html

---
*This section will be expanded as subsequent phases are completed.*
*Next section: IV. Model Enhancement & STT Pipeline (Phase 2)*