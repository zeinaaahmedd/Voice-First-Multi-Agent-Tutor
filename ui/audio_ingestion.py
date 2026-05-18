"""
BASIRA – Audio Ingestion Module
Phase 1 · audio_ingestion.py

Responsibilities:
  1. Accept audio from multiple sources (numpy array, file path, bytes)
  2. Validate quality (duration, sample rate, non-silence)
  3. Normalize to WhisperX requirements: mono, float32, 16 kHz
  4. Return (audio_np, metadata_dict) for the STT Agent

WhisperX requirement:
  • numpy float32 array, shape (N,), sample rate 16 000 Hz
"""

from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path
from typing import Union, Tuple, Dict, Any

import numpy as np

# ── Optional imports (graceful degradation) ─────────────────────────────────
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False
    warnings.warn("soundfile not installed. File-path ingestion disabled.", stacklevel=2)

try:
    from scipy.signal import resample_poly
    from math import gcd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SR        = 16_000          # WhisperX native sample rate
MIN_DURATION_SEC = 0.5             # Reject recordings shorter than this
MAX_DURATION_SEC = 300.0           # Soft cap – warn but allow
SILENCE_THRESHOLD = 1e-4           # RMS below this → considered silence


# ── Public API ───────────────────────────────────────────────────────────────
AudioSource = Union[
    Tuple[int, np.ndarray],        # (sample_rate, array)  – Gradio numpy mode
    np.ndarray,                    # raw array (assumed TARGET_SR)
    str,                           # file path
    Path,                          # file path
    bytes,                         # raw audio bytes (WAV/WEBM)
]

def ingest_audio(
    source: AudioSource,
    sample_rate: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize any audio source to WhisperX-ready float32 mono @ 16 kHz.

    Parameters
    ----------
    source       : one of (np.ndarray, tuple, str/Path, bytes)
    sample_rate  : only used when source is a bare np.ndarray

    Returns
    -------
    audio_np : np.ndarray  – float32, mono, 16 kHz
    metadata : dict        – duration_sec, sample_rate, shape, original_sr, warnings
    """
    meta: Dict[str, Any] = {
        "original_sr": None,
        "shape":       None,
        "duration_sec": 0.0,
        "sample_rate": TARGET_SR,
        "warnings":    [],
    }

    # ── 1. Decode source into (array, sr) ───────────────────────────────────
    audio, sr = _decode_source(source, sample_rate, meta)

    # ── 2. Ensure float32 ───────────────────────────────────────────────────
    audio = audio.astype(np.float32)

    # ── 3. Convert stereo → mono ────────────────────────────────────────────
    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[1] < audio.shape[0] else 0)

    audio = audio.flatten()

    # ── 4. Normalize amplitude to [-1, 1] ───────────────────────────────────
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # ── 5. Resample to TARGET_SR ─────────────────────────────────────────────
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)

    # ── 6. Quality checks ───────────────────────────────────────────────────
    duration = len(audio) / TARGET_SR
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Recording too short ({duration:.2f}s < {MIN_DURATION_SEC}s). "
            "Please speak for longer."
        )

    if duration > MAX_DURATION_SEC:
        meta["warnings"].append(
            f"Long recording ({duration:.0f}s). Processing may be slow."
        )

    if rms < SILENCE_THRESHOLD:
        meta["warnings"].append(
            "Audio appears to be silent. Check microphone connection."
        )

    # ── 7. Populate metadata ─────────────────────────────────────────────────
    meta.update({
        "duration_sec": round(duration, 3),
        "shape":        audio.shape,
        "sample_rate":  TARGET_SR,
        "rms":          round(rms, 6),
    })

    return audio, meta


# ── Internal helpers ─────────────────────────────────────────────────────────
def _decode_source(
    source: AudioSource,
    sample_rate: int | None,
    meta: dict,
) -> Tuple[np.ndarray, int]:

    # Gradio tuple (sr, array)
    if isinstance(source, tuple) and len(source) == 2:
        sr, arr = source
        meta["original_sr"] = sr
        return np.array(arr), int(sr)

    # Bare numpy array
    if isinstance(source, np.ndarray):
        sr = sample_rate or TARGET_SR
        meta["original_sr"] = sr
        return source, int(sr)

    # File path (str or Path)
    if isinstance(source, (str, Path)):
        if not _HAS_SF:
            raise ImportError("Install soundfile: pip install soundfile")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        meta["original_sr"] = sr
        return arr, int(sr)

    # Raw bytes (webm / wav from browser)
    if isinstance(source, (bytes, bytearray)):
        return _decode_bytes(source, meta)

    raise TypeError(f"Unsupported audio source type: {type(source)}")


def _decode_bytes(data: bytes, meta: dict) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes using soundfile or fallback to numpy WAV parser."""
    if _HAS_SF:
        with io.BytesIO(data) as buf:
            try:
                arr, sr = sf.read(buf, dtype="float32", always_2d=False)
                meta["original_sr"] = sr
                return arr, int(sr)
            except Exception:
                pass  # fall through to WAV parser

    # Minimal WAV parser fallback (handles browser-recorded WAV chunks)
    return _parse_wav_bytes(data, meta)


def _parse_wav_bytes(data: bytes, meta: dict) -> Tuple[np.ndarray, int]:
    """
    Minimal PCM WAV parser.
    Handles 16-bit PCM which is what most browsers emit.
    """
    if data[0:4] != b"RIFF":
        raise ValueError(
            "Received non-WAV bytes and soundfile is not available. "
            "Install soundfile: pip install soundfile"
        )

    import struct
    # fmt chunk starts at offset 20
    audio_format  = struct.unpack_from("<H", data, 20)[0]
    num_channels  = struct.unpack_from("<H", data, 22)[0]
    sr            = struct.unpack_from("<I", data, 24)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]

    if audio_format != 1:  # 1 = PCM
        raise ValueError("Only PCM WAV supported in fallback parser.")

    # Data chunk
    data_offset = 44
    pcm = np.frombuffer(data[data_offset:], dtype=np.int16)

    # Reshape for channels
    if num_channels > 1:
        pcm = pcm.reshape(-1, num_channels).mean(axis=1).astype(np.int16)

    arr = pcm.astype(np.float32) / 32768.0
    meta["original_sr"] = sr
    return arr, int(sr)


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample audio array from from_sr to to_sr.
    Uses scipy.signal.resample_poly when available (better quality),
    falls back to numpy linear interpolation.
    """
    if _HAS_SCIPY:
        g = gcd(to_sr, from_sr)
        return resample_poly(audio, to_sr // g, from_sr // g).astype(np.float32)

    # Numpy fallback: linear interpolation
    original_len = len(audio)
    target_len   = int(original_len * to_sr / from_sr)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic 1-second tone @ 44.1 kHz (simulates browser audio)
    sr_test = 44_100
    t       = np.linspace(0, 1, sr_test, dtype=np.float32)
    tone    = np.sin(2 * np.pi * 440 * t) * 0.5   # 440 Hz A4

    audio_out, meta_out = ingest_audio((sr_test, tone))

    print("✅  Ingestion test passed")
    print(f"   Input : {sr_test} Hz, {len(tone)} samples")
    print(f"   Output: {meta_out['sample_rate']} Hz, {meta_out['shape']} samples")
    print(f"   Duration: {meta_out['duration_sec']} sec")
    print(f"   Warnings: {meta_out['warnings'] or 'none'}")