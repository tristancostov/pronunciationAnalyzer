#!/usr/bin/env python3
"""Text-independent acoustic syllable-nucleus detector (experimental V2).

The detector deliberately does not accept a transcript or an expected
syllable count.  It combines a robust, multi-band sonority envelope with
periodicity and spectral cues, then applies confidence-aware peak selection.
The initial weights are engineering priors; they are intended to be calibrated
only after human-reviewed nucleus points are available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from audio_input import prepare_audio


EPSILON = 1e-10

# Frozen after development on 3local + 6fori and one final evaluation on
# previously untouched 7fori.  Keeping the production preset in code makes
# the application reproducible even when research result files are moved.
PRODUCTION_V2_CONFIG_NAME = "recall_4_noise_gate"
PRODUCTION_V2_OVERRIDES = {
    "strong_prominence": 0.35,
    "weak_prominence": 0.08,
    "strong_min_confidence": 0.34,
    "weak_min_confidence": 0.48,
    "strong_min_periodicity": 0.0,
    "strong_min_strength": 0.4,
    "strong_min_vowel_likeness": 0.45,
}


@dataclass(frozen=True)
class NucleusV2Config:
    """Parameters that are safe to calibrate without changing the algorithm."""

    sample_rate: int = 16_000
    frame_seconds: float = 0.030
    hop_seconds: float = 0.010
    n_fft: int = 1024
    fmin: float = 65.0
    fmax: float = 450.0
    speech_floor_db: float = 34.0
    smoothing_seconds: float = 0.022
    min_distance_seconds: float = 0.055
    strong_prominence: float = 0.60
    weak_prominence: float = 0.20
    strong_min_confidence: float = 0.43
    weak_min_confidence: float = 0.60
    # Strong spectral peaks still need minimum vowel evidence.  These gates are
    # separate from prominence so breath/noise cannot pass merely by being tall.
    strong_min_periodicity: float = 0.0
    strong_min_strength: float = 0.0
    strong_min_vowel_likeness: float = 0.30
    band_edges_hz: tuple[tuple[float, float], ...] = (
        (60.0, 370.0),
        (370.0, 800.0),
        (800.0, 1400.0),
        (1400.0, 2250.0),
        (2250.0, 3450.0),
        (3450.0, 5130.0),
        (5130.0, 7500.0),
    )
    # Mid bands that commonly carry vowel-formant energy receive most weight;
    # high-frequency noise bands slightly suppress fricative-like peaks.
    band_weights: tuple[float, ...] = (
        0.20, 0.95, 1.00, 0.62, 0.18, -0.08, -0.12
    )


def production_v2_config() -> NucleusV2Config:
    """Return the frozen configuration used by the desktop application."""
    return NucleusV2Config(**PRODUCTION_V2_OVERRIDES)


@dataclass(frozen=True)
class NucleusCandidate:
    frame: int
    time_seconds: float
    confidence: float
    prominence: float
    strength: float
    periodicity: float
    vowel_likeness: float
    sonorant_penalty: float
    weak_recovery: bool


@dataclass
class NucleusDetection:
    sample_rate: int
    duration_seconds: float
    hop_seconds: float
    nuclei: list[NucleusCandidate]
    frame_times: np.ndarray
    likelihood: np.ndarray
    speech_mask: np.ndarray
    feature_tracks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def times(self) -> list[float]:
        return [candidate.time_seconds for candidate in self.nuclei]

    def best_time_in_interval(self, start: float, end: float) -> float:
        """Return the most nucleus-like frame inside an annotation interval."""
        if end <= start or len(self.frame_times) == 0:
            return max(0.0, start)
        indices = np.flatnonzero(
            (self.frame_times >= start) & (self.frame_times <= end)
        )
        if len(indices) == 0:
            return (start + end) / 2.0
        best = indices[int(np.argmax(self.likelihood[indices]))]
        return float(np.clip(self.frame_times[best], start, end))

    def to_dict(self, include_tracks: bool = False) -> dict:
        payload = {
            "algorithm": "acoustic_nuclei_v2",
            "text_independent": True,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 6),
            "hop_seconds": self.hop_seconds,
            "count": len(self.nuclei),
            "nuclei": [asdict(candidate) for candidate in self.nuclei],
        }
        if include_tracks:
            payload["tracks"] = {
                "frame_times": self.frame_times.tolist(),
                "likelihood": self.likelihood.tolist(),
                "speech_mask": self.speech_mask.astype(int).tolist(),
                **{
                    name: values.tolist()
                    for name, values in self.feature_tracks.items()
                },
            }
        return payload


def _align(values: np.ndarray, length: int, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == length:
        return values
    if len(values) > length:
        return values[:length]
    return np.pad(values, (0, length - len(values)), constant_values=fill)


def _robust_z(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sample = values[mask] if mask is not None and np.any(mask) else values
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return np.zeros_like(values)
    median = float(np.median(sample))
    q25, q75 = np.percentile(sample, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if scale < 1e-6:
        scale = float(np.std(sample))
    if scale < 1e-6:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -5.0, 5.0)


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _pyin_periodicity(
    signal: np.ndarray,
    sr: int,
    hop_length: int,
    n_frames: int,
    config: NucleusV2Config,
) -> np.ndarray:
    """Estimate periodicity; fall back gracefully on pathological recordings."""
    frame_length = max(config.n_fft, 1024)
    try:
        _, voiced, probability = librosa.pyin(
            signal,
            fmin=config.fmin,
            fmax=min(config.fmax, sr / 2.0 - 1.0),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
        probability = np.nan_to_num(probability, nan=0.0)
        probability = _align(probability, n_frames)
        voiced = _align(np.asarray(voiced, dtype=float), n_frames)
        return np.clip(np.maximum(probability, 0.35 * voiced), 0.0, 1.0)
    except (ValueError, FloatingPointError):
        return np.zeros(n_frames, dtype=float)


def _band_log_energies(
    power: np.ndarray,
    frequencies: np.ndarray,
    bands: Sequence[tuple[float, float]],
) -> np.ndarray:
    values = []
    for low, high in bands:
        bins = (frequencies >= low) & (frequencies < high)
        if not np.any(bins):
            values.append(np.full(power.shape[1], -100.0))
            continue
        energy = np.mean(power[bins, :], axis=0)
        values.append(10.0 * np.log10(energy + EPSILON))
    return np.vstack(values)


def _candidate_confidence(
    frame: int,
    prominence: float,
    envelope: np.ndarray,
    speech_mask: np.ndarray,
    periodicity: np.ndarray,
    vowel_likeness: np.ndarray,
    flux_score: np.ndarray,
    sonorant_penalty: np.ndarray,
) -> tuple[float, float]:
    valid = envelope[speech_mask] if np.any(speech_mask) else envelope
    centre = float(np.median(valid)) if len(valid) else 0.0
    spread = float(np.percentile(valid, 75) - np.percentile(valid, 25))
    spread = max(spread, 0.25)
    strength = float(_sigmoid((envelope[frame] - centre) / spread))
    prominence_score = float(_sigmoid((prominence - 0.28) / 0.22))
    confidence = (
        0.20 * strength
        + 0.22 * prominence_score
        + 0.27 * vowel_likeness[frame]
        + 0.22 * periodicity[frame]
        + 0.09 * flux_score[frame]
        - 0.12 * sonorant_penalty[frame]
    )
    return float(np.clip(confidence, 0.0, 1.0)), strength


def _non_maximum_suppression(
    candidates: Iterable[NucleusCandidate],
    min_distance_frames: int,
) -> list[NucleusCandidate]:
    ranked = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    kept: list[NucleusCandidate] = []
    for candidate in ranked:
        if all(
            abs(candidate.frame - existing.frame) >= min_distance_frames
            for existing in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item.frame)


def detect_nuclei_v2(
    signal: np.ndarray,
    sample_rate: int,
    config: NucleusV2Config | None = None,
) -> NucleusDetection:
    """Detect syllable nuclei from audio alone.

    No transcript, word timestamp, expected syllable count, dictionary, or
    forced alignment is used here.
    """
    config = config or NucleusV2Config(sample_rate=sample_rate)
    y = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sample_rate != config.sample_rate:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=config.sample_rate)
        sample_rate = config.sample_rate
    if len(y) == 0:
        empty = np.array([], dtype=float)
        return NucleusDetection(
            sample_rate, 0.0, config.hop_seconds, [], empty, empty,
            empty.astype(bool), {})

    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak

    frame_length = max(64, int(round(config.frame_seconds * sample_rate)))
    hop_length = max(16, int(round(config.hop_seconds * sample_rate)))
    n_fft = max(config.n_fft, 2 ** int(math.ceil(math.log2(frame_length))))
    spectrum = librosa.stft(
        y,
        n_fft=n_fft,
        win_length=frame_length,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    n_frames = power.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sample_rate, hop_length=hop_length
    )

    rms = np.sqrt(np.mean(power, axis=0) + EPSILON)
    rms_db = 20.0 * np.log10(rms + EPSILON)
    reference_db = float(np.percentile(rms_db, 95.0))
    speech_mask = rms_db >= reference_db - config.speech_floor_db
    speech_mask = median_filter(speech_mask.astype(np.uint8), size=3) > 0

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band_db = _band_log_energies(power, frequencies, config.band_edges_hz)
    band_z = np.vstack([_robust_z(row, speech_mask) for row in band_db])

    # Keep a compact Mel/cepstral representation for research models.  These
    # tracks are not consulted by the frozen V2 peak rules, so adding them does
    # not change the production detector.  Per-recording robust normalization
    # reduces microphone/gain differences; train-only standardization is still
    # applied by supervised experiments to prevent speaker leakage.
    mel_filter = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=20,
        fmin=60.0,
        fmax=min(7500.0, sample_rate / 2.0),
        norm="slaney",
    )
    mel_power = np.maximum(mel_filter @ power, EPSILON)
    mel_db = 10.0 * np.log10(mel_power)
    mel_z = np.vstack([_robust_z(row, speech_mask) for row in mel_db])
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=12)

    weights = np.asarray(config.band_weights, dtype=float)
    if len(weights) != band_z.shape[0]:
        raise ValueError("band_weights and band_edges_hz must have equal lengths")
    weights = weights / max(float(np.sum(np.abs(weights))), EPSILON)
    spectral_sonority = np.sum(band_z * weights[:, None], axis=0)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    zcr = _align(zcr, n_frames, fill=1.0)
    zcr_score = 1.0 - np.clip(zcr / 0.24, 0.0, 1.0)

    flatness = librosa.feature.spectral_flatness(S=power + EPSILON)[0]
    flatness = _align(flatness, n_frames, fill=1.0)
    flatness_score = np.exp(-10.0 * np.clip(flatness, 0.0, 1.0))

    normalized_spectrum = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + EPSILON)
    spectral_entropy = -np.sum(
        normalized_spectrum * np.log(normalized_spectrum + EPSILON), axis=0
    ) / math.log(max(normalized_spectrum.shape[0], 2))
    formant_bins = (frequencies >= 250.0) & (frequencies <= 2000.0)
    formant_energy = np.mean(power[formant_bins, :], axis=0)
    formant_energy_db = 10.0 * np.log10(formant_energy + EPSILON)
    # Energy-to-entropy ratio expands the separation between structured,
    # formant-rich speech and diffuse noise (Wang & Yi, 2017).
    energy_entropy_ratio = np.log1p(
        formant_energy / (spectral_entropy + EPSILON)
    )
    flux = np.zeros(n_frames, dtype=float)
    if n_frames > 1:
        flux[1:] = np.sqrt(np.sum(np.diff(normalized_spectrum, axis=1) ** 2, axis=0))
    flux_z = _robust_z(flux, speech_mask)
    flux_score = np.asarray(_sigmoid(1.0 - flux_z), dtype=float)

    periodicity = _pyin_periodicity(y, sample_rate, hop_length, n_frames, config)
    # pYIN may be cautious on very short reduced vowels.  Spectral regularity
    # provides a conservative backup without turning it into a hard voicing gate.
    pseudo_periodicity = 0.55 * zcr_score + 0.45 * flatness_score
    periodicity = np.maximum(periodicity, 0.35 * pseudo_periodicity)

    mid_energy = np.mean(band_z[1:4, :], axis=0)
    high_energy = np.mean(band_z[4:, :], axis=0)
    low_dominance = band_z[0, :] - np.mean(band_z[1:3, :], axis=0)
    sonorant_penalty = np.asarray(_sigmoid(1.4 * (low_dominance - 0.65)), dtype=float)
    vowel_raw = (
        0.62 * mid_energy
        - 0.28 * high_energy
        + 1.15 * (periodicity - 0.40)
        + 0.38 * zcr_score
        + 0.30 * flatness_score
        - 0.50 * sonorant_penalty
    )
    vowel_likeness = np.asarray(_sigmoid(vowel_raw), dtype=float)

    envelope = (
        spectral_sonority
        + 0.42 * (periodicity - 0.5)
        + 0.30 * (vowel_likeness - 0.5)
        - 0.12 * np.maximum(flux_z, 0.0)
    )
    sigma_frames = max(0.5, config.smoothing_seconds / config.hop_seconds)
    envelope = gaussian_filter1d(envelope, sigma=sigma_frames, mode="nearest")
    if np.any(speech_mask):
        muted_level = float(np.min(envelope[speech_mask])) - 2.0
        envelope = envelope.copy()
        envelope[~speech_mask] = muted_level

    likelihood = np.asarray(
        _sigmoid(1.15 * _robust_z(envelope, speech_mask)
                 + 1.00 * (vowel_likeness - 0.5)
                 + 0.65 * (periodicity - 0.5)),
        dtype=float,
    )
    likelihood[~speech_mask] = 0.0

    weak_distance = max(1, int(round(0.035 / config.hop_seconds)))
    strong_peaks, strong_props = find_peaks(
        envelope,
        prominence=config.strong_prominence,
        distance=weak_distance,
    )
    weak_peaks, weak_props = find_peaks(
        envelope,
        prominence=config.weak_prominence,
        distance=weak_distance,
    )
    prominence_by_frame: dict[int, float] = {
        int(frame): float(value)
        for frame, value in zip(weak_peaks, weak_props["prominences"])
    }
    strong_frames = {int(frame) for frame in strong_peaks}
    for frame, value in zip(strong_peaks, strong_props["prominences"]):
        prominence_by_frame[int(frame)] = max(
            prominence_by_frame.get(int(frame), 0.0), float(value)
        )

    provisional: list[NucleusCandidate] = []
    for frame in sorted(prominence_by_frame):
        if not speech_mask[frame]:
            continue
        prominence = prominence_by_frame[frame]
        confidence, strength = _candidate_confidence(
            frame,
            prominence,
            envelope,
            speech_mask,
            periodicity,
            vowel_likeness,
            flux_score,
            sonorant_penalty,
        )
        weak_recovery = frame not in strong_frames
        if weak_recovery:
            keep = (
                confidence >= config.weak_min_confidence
                and vowel_likeness[frame] >= 0.52
                and periodicity[frame] >= 0.32
            )
        else:
            keep = (
                confidence >= config.strong_min_confidence
                and vowel_likeness[frame] >= config.strong_min_vowel_likeness
                and periodicity[frame] >= config.strong_min_periodicity
                and strength >= config.strong_min_strength
            )
        if not keep:
            continue
        provisional.append(NucleusCandidate(
            frame=frame,
            time_seconds=round(float(frame_times[frame]), 6),
            confidence=round(confidence, 6),
            prominence=round(prominence, 6),
            strength=round(strength, 6),
            periodicity=round(float(periodicity[frame]), 6),
            vowel_likeness=round(float(vowel_likeness[frame]), 6),
            sonorant_penalty=round(float(sonorant_penalty[frame]), 6),
            weak_recovery=weak_recovery,
        ))

    min_distance_frames = max(
        1, int(round(config.min_distance_seconds / config.hop_seconds))
    )
    nuclei = _non_maximum_suppression(provisional, min_distance_frames)
    return NucleusDetection(
        sample_rate=sample_rate,
        duration_seconds=len(y) / sample_rate,
        hop_seconds=hop_length / sample_rate,
        nuclei=nuclei,
        frame_times=frame_times,
        likelihood=likelihood,
        speech_mask=speech_mask,
        feature_tracks={
            "envelope": envelope,
            "rms_db": rms_db,
            "spectral_sonority": spectral_sonority,
            "periodicity": periodicity,
            "vowel_likeness": vowel_likeness,
            "zcr_score": zcr_score,
            "flatness_score": flatness_score,
            "flux_score": flux_score,
            "sonorant_penalty": sonorant_penalty,
            "spectral_entropy": spectral_entropy,
            "formant_energy_db": formant_energy_db,
            "energy_entropy_ratio": energy_entropy_ratio,
            "band_z": band_z,
            "mel_z": mel_z,
            "mfcc": mfcc,
        },
    )


def detect_file(
    path: str | os.PathLike[str],
    config: NucleusV2Config | None = None,
) -> NucleusDetection:
    prepared = prepare_audio(path)
    target_sr = (config or NucleusV2Config()).sample_rate
    signal, sample_rate = librosa.load(
        prepared.analysis_path, sr=target_sr, mono=True, res_type="kaiser_fast"
    )
    return detect_nuclei_v2(signal, sample_rate, config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Audio or video file")
    parser.add_argument("--output", help="Write candidates to this JSON file")
    parser.add_argument("--tracks", action="store_true", help="Include frame tracks")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detection = detect_file(args.audio)
    payload = detection.to_dict(include_tracks=args.tracks)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved {len(detection.nuclei)} nuclei to {output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
