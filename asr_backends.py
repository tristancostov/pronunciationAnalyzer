"""Pluggable offline speech recognizers with a common word-timestamp format."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import os
from pathlib import Path
import re
import wave


SCRIPT_DIR = Path(__file__).resolve().parent
WHISPER_MODEL_NAME = "large-v3-turbo"
WHISPER_MODEL_DIR = SCRIPT_DIR / "whisper-models"
CUDA_RUNTIME_DIR = SCRIPT_DIR / "cuda-runtime"


@dataclass
class RecognitionResult:
    engine: str
    text: str
    words: list[dict]
    device: str


def recognize_vosk(audio_path, vosk_model, recognizer_class) -> RecognitionResult:
    """Recognize a normalized PCM WAV with the existing VOSK model."""
    with wave.open(os.fspath(audio_path), "rb") as wf:
        recognizer = recognizer_class(vosk_model, wf.getframerate())
        recognizer.SetWords(True)
        words, text_parts = [], []
        while True:
            chunk = wf.readframes(4000)
            if not chunk:
                break
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                words.extend(result.get("result", []))
                if result.get("text"):
                    text_parts.append(result["text"])
        final = json.loads(recognizer.FinalResult())
        words.extend(final.get("result", []))
        if final.get("text"):
            text_parts.append(final["text"])
    return RecognitionResult("vosk", " ".join(text_parts), words, "cpu")


_cached_whisper = None
_dll_directory_handle = None


def _enable_private_cuda_runtime():
    global _dll_directory_handle
    if not CUDA_RUNTIME_DIR.is_dir():
        return
    runtime = str(CUDA_RUNTIME_DIR)
    os.environ["PATH"] = runtime + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _dll_directory_handle = os.add_dll_directory(runtime)


def load_whisper_model(verbose=True):
    """Load large-v3-turbo once, preferring CUDA and falling back to CPU."""
    global _cached_whisper
    if _cached_whisper is not None:
        return _cached_whisper

    _enable_private_cuda_runtime()
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Высокоточный режим требует faster-whisper. "
            "Установите зависимости из requirements.txt."
        ) from exc

    has_cuda = ctranslate2.get_cuda_device_count() > 0
    device = "cuda" if has_cuda else "cpu"
    compute_type = "float16" if has_cuda else "int8"
    if verbose:
        print(f"Whisper {WHISPER_MODEL_NAME}: {device}/{compute_type}")

    try:
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=device,
            compute_type=compute_type,
            download_root=str(WHISPER_MODEL_DIR),
        )
    except RuntimeError as exc:
        # A CUDA driver can be visible while cuBLAS/cuDNN is unavailable.
        # CPU mode remains functional and produces the same model output.
        if not has_cuda:
            raise
        if verbose:
            print(f"Whisper GPU недоступен ({exc}); переход на CPU/int8.")
        device, compute_type = "cpu", "int8"
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=device,
            compute_type=compute_type,
            download_root=str(WHISPER_MODEL_DIR),
        )

    _cached_whisper = (model, device)
    return _cached_whisper


def _clean_whisper_word(token: str) -> str:
    token = token.strip().lower()
    return re.sub(r"[^0-9a-zа-яё-]", "", token, flags=re.IGNORECASE)


def recognize_whisper(audio_path, reference_text="", verbose=True) -> RecognitionResult:
    """Recognize Russian speech with word-level timestamps."""
    global _cached_whisper
    model, device = load_whisper_model(verbose=verbose)
    def transcribe(current_model):
        segments, _info = current_model.transcribe(
            os.fspath(audio_path),
            language="ru",
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
            hotwords=(reference_text.strip() or None),
        )
        result = []
        for segment in segments:
            for item in segment.words or []:
                token = _clean_whisper_word(item.word)
                if not token:
                    continue
                duration = float(item.end) - float(item.start)
                probability = float(item.probability)
                # Whisper can append a very short, low-probability word after
                # a hard file cut. Keep genuine short function words when
                # their probability is high.
                if duration < 0.080 and probability < 0.85:
                    continue
                result.append({
                    "word": token,
                    "start": round(float(item.start), 3),
                    "end": round(float(item.end), 3),
                    "conf": round(probability, 4),
                })
        return result

    try:
        words = transcribe(model)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        if verbose:
            print(f"Whisper CUDA inference недоступен ({exc}); CPU/int8.")
        from faster_whisper import WhisperModel
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_MODEL_DIR),
        )
        device = "cpu"
        _cached_whisper = (model, device)
        words = transcribe(model)
    text = " ".join(word["word"] for word in words)
    return RecognitionResult("whisper-large-v3-turbo", text, words, device)


def merge_whisper_text_with_vosk_times(
    whisper: RecognitionResult,
    vosk: RecognitionResult,
) -> RecognitionResult:
    """Use Whisper words and VOSK's tighter acoustic time anchors.

    Equal and one-to-one replaced words receive the exact VOSK interval.
    Unequal replacement blocks are proportionally fitted into the VOSK block.
    Whisper-only insertions retain their native timestamps.
    """
    whisper_tokens = [word["word"] for word in whisper.words]
    vosk_tokens = [word["word"] for word in vosk.words]
    matcher = difflib.SequenceMatcher(None, vosk_tokens, whisper_tokens,
                                      autojunk=False)
    merged = []

    def anchored(whisper_word, start, end):
        item = dict(whisper_word)
        item["start"] = round(float(start), 3)
        item["end"] = round(float(end), 3)
        item["timingSource"] = "vosk"
        return item

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        vblock = vosk.words[i1:i2]
        wblock = whisper.words[j1:j2]
        if opcode in {"equal", "replace"} and len(vblock) == len(wblock):
            for vword, wword in zip(vblock, wblock):
                merged.append(anchored(wword, vword["start"], vword["end"]))
        elif opcode == "replace" and vblock and wblock:
            vstart, vend = vblock[0]["start"], vblock[-1]["end"]
            wstart, wend = wblock[0]["start"], wblock[-1]["end"]
            wspan = max(float(wend) - float(wstart), 1e-6)
            vspan = max(float(vend) - float(vstart), 1e-6)
            for wword in wblock:
                rel_start = (float(wword["start"]) - float(wstart)) / wspan
                rel_end = (float(wword["end"]) - float(wstart)) / wspan
                merged.append(anchored(
                    wword,
                    float(vstart) + rel_start * vspan,
                    float(vstart) + rel_end * vspan))
        elif opcode == "insert":
            for wword in wblock:
                item = dict(wword)
                item["timingSource"] = "whisper"
                merged.append(item)
        # A VOSK-only deletion is omitted because Whisper owns the transcript.

    text = " ".join(word["word"] for word in merged)
    return RecognitionResult(
        "whisper-large-v3-turbo+vosk-timestamps",
        text,
        merged,
        whisper.device,
    )
