"""
stt_agent.py  –  Speech-to-Text agent for the Voice-First Multi-Agent Tutor

Tries three backends in priority order:
    1. whisperx      – production (word-level timestamps, GPU-friendly)
    2. openai-whisper – lighter CPU fallback
    3. manual input  – zero-dependency dev / testing mode

Public API
    stt = WhisperXSTT(language="ar", model_size="base")
    text = stt.transcribe("path/to/audio.wav")
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

log = logging.getLogger("stt_agent")

# Ensure ffmpeg.exe is discoverable as a plain subprocess name before whisperx
# tries to call it. ffmpeg_utils copies the imageio-ffmpeg bundle to a stable
# location with the correct filename and prepends it to PATH.
import ffmpeg_utils  # noqa: F401  (side-effect import)


class WhisperXSTT:
    """
    Lazy-loading STT wrapper.  The model is not loaded until the first
    call to transcribe(), keeping startup fast.
    """

    def __init__(self, language: str = "ar", model_size: str = "base"):
        self.language = language
        self.model_size = model_size
        self._backend: Optional[str] = None
        self._model = None

    # ── backend resolution ───────────────────────────────────────────────────

    def _resolve_backend(self) -> str:
        for lib in ("whisperx", "whisper"):
            try:
                importlib.import_module(lib)
                log.info("STT backend selected: %s", lib)
                return lib
            except ImportError:
                pass
        log.warning(
            "Neither whisperx nor openai-whisper is installed — "
            "falling back to manual terminal input.\n"
            "  pip install openai-whisper\n"
            "  pip install whisperx"
        )
        return "manual"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._backend == "whisperx":
            import whisperx
            device = "cuda" if self._cuda_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = whisperx.load_model(
                self.model_size, device, compute_type=compute_type
            )
        elif self._backend == "whisper":
            import whisper
            self._model = whisper.load_model(self.model_size)
        log.info("STT model loaded  (backend=%s, size=%s)", self._backend, self.model_size)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── public API ───────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> str:
        """Return the transcription for *audio_path* in the configured language."""
        if self._backend is None:
            self._backend = self._resolve_backend()

        if self._backend == "manual":
            return self._manual(audio_path)

        self._ensure_model()
        if self._backend == "whisperx":
            return self._run_whisperx(audio_path)
        return self._run_openai_whisper(audio_path)

    # ── backend implementations ──────────────────────────────────────────────

    def _run_whisperx(self, audio_path: str) -> str:
        import numpy as np
        import whisperx

        audio = whisperx.load_audio(audio_path)

        # pyannote VAD (inside whisperx) requires a (1, N) tensor where N > 0.
        # A recording shorter than ~1 s — or ffmpeg producing 0 bytes for silence —
        # yields shape (1, 0) which triggers "waveform must be (channel, time)".
        # Pad to 1 second (16 000 samples at 16 kHz) with silence so VAD can run.
        MIN_SAMPLES = 16_000
        if audio.size == 0:
            log.warning("Audio file is empty; returning empty transcription.")
            return ""
        if audio.size < MIN_SAMPLES:
            audio = np.pad(audio, (0, MIN_SAMPLES - audio.size))

        result = self._model.transcribe(audio, language=self.language, batch_size=16)
        return " ".join(s["text"] for s in result.get("segments", [])).strip()

    def _run_openai_whisper(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path, language=self.language)
        return result["text"].strip()

    def _manual(self, audio_path: str) -> str:
        print(f"\n  [STT PLACEHOLDER]  Audio: {audio_path}")
        print("  Install whisperx or openai-whisper to skip this step.")
        return input("  ↳ Enter transcription manually: ").strip()
