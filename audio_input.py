"""Audio input normalization for the desktop application and analyzer.

The analysis pipeline always receives a 16 kHz, mono, signed 16-bit PCM WAV.
Users can provide a microphone recording or a common media file; ffmpeg does
the conversion automatically and the normalized copy is cached in the system
temporary directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class PreparedAudio:
    original_path: str
    analysis_path: str
    converted: bool


def is_analysis_wav(path: str | os.PathLike[str]) -> bool:
    """Return True for mono/16 kHz/16-bit uncompressed PCM WAV files."""
    try:
        with wave.open(os.fspath(path), "rb") as wf:
            return (
                wf.getnchannels() == 1
                and wf.getframerate() == TARGET_SAMPLE_RATE
                and wf.getsampwidth() == 2
                and wf.getcomptype() == "NONE"
            )
    except (OSError, EOFError, wave.Error):
        return False


def _cache_path(source: Path, cache_dir: Path) -> Path:
    stat = source.stat()
    fingerprint = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                        for ch in source.stem)[:48] or "audio"
    return cache_dir / f"{safe_stem}_{digest}_16k_mono.wav"


def prepare_audio(
    input_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
) -> PreparedAudio:
    """Validate and, when necessary, normalize an audio/video input.

    ffmpeg is invoked without a shell, so paths containing spaces or non-ASCII
    characters are handled safely. A compliant WAV is used directly.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {source}")

    if is_analysis_wav(source):
        return PreparedAudio(str(source), str(source), False)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Для автоматической подготовки аудио нужен ffmpeg в PATH. "
            "Установите ffmpeg и перезапустите приложение."
        )

    target_dir = Path(cache_dir) if cache_dir else (
        Path(tempfile.gettempdir()) / "pronunciation_analyzer_audio"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(source, target_dir)
    if target.is_file() and is_analysis_wav(target):
        return PreparedAudio(str(source), str(target), True)

    partial = target.with_suffix(".tmp.wav")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(partial),
    ]
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "неизвестная ошибка ffmpeg"
            raise RuntimeError(f"Не удалось прочитать аудио: {detail}")
        if not is_analysis_wav(partial):
            raise RuntimeError("ffmpeg создал аудио в неподдерживаемом формате")
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()

    return PreparedAudio(str(source), str(target), True)
