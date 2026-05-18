"""
ffmpeg_utils.py  –  Makes imageio-ffmpeg's bundled binary discoverable system-wide.

whisperx.load_audio(), pydub.AudioSegment, and other tools call the ffmpeg
executable by name ("ffmpeg") via subprocess.  imageio-ffmpeg ships a real
binary but names it "ffmpeg-win-x86_64-v7.1.exe" on Windows, so simply adding
its directory to PATH is not enough.

This module copies the binary to a stable cache location as "ffmpeg.exe" once,
then prepends that directory to PATH and configures pydub — so every subsequent
subprocess call to "ffmpeg" just works.

Usage:
    import ffmpeg_utils          # side-effect: PATH is patched immediately
    exe = ffmpeg_utils.FFMPEG    # full path to the usable executable
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# ── locate the bundled binary ─────────────────────────────────────────────────
try:
    import imageio_ffmpeg
    _bundled = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    _bundled = None

# ── resolve or create an "ffmpeg.exe"-named copy ─────────────────────────────
def _resolve_ffmpeg() -> str:
    # 1. System ffmpeg already in PATH — nothing to do
    system = shutil.which("ffmpeg")
    if system:
        return system

    if not _bundled:
        raise RuntimeError(
            "ffmpeg not found. Install imageio-ffmpeg:\n"
            "  pip install imageio-ffmpeg"
        )

    # 2. Copy bundled binary to a stable location with the plain name "ffmpeg.exe"
    cache_dir = Path.home() / ".cache" / "basira_ffmpeg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = cache_dir / "ffmpeg.exe"

    if not ffmpeg_exe.exists():
        shutil.copy2(_bundled, ffmpeg_exe)

    # 3. Prepend cache directory to PATH so subprocess("ffmpeg") resolves it
    _prepend_to_path(str(cache_dir))

    return str(ffmpeg_exe)


def _prepend_to_path(directory: str) -> None:
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current


# ── configure pydub to use the resolved binary ────────────────────────────────
def _configure_pydub(exe: str) -> None:
    try:
        from pydub import AudioSegment
        AudioSegment.converter = exe
    except ImportError:
        pass


# ── run once at import time ───────────────────────────────────────────────────
FFMPEG: str = _resolve_ffmpeg()
_configure_pydub(FFMPEG)
