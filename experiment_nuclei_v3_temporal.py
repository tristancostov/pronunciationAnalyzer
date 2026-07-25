#!/usr/bin/env python3
"""Frame-regression experiment for text-independent syllable nuclei.

The model follows the useful part of the BLSTM literature: predict a smooth
target through time from Mel/cepstral and acoustic trajectories, then recover
one event from each local maximum.  Every reported fold holds out a complete
recording/speaker.  Event thresholds are chosen only from inner cross-speaker
predictions, never from the held-out recording.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch
from torch import nn
from torch.nn import functional as F

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from evaluate_nuclei_v2 import match_events
from evaluate_sylber_zero_shot import annotation
from prepare_nucleus_annotations import find_audio


SEED = 37
TOLERANCE_SECONDS = 0.050
TARGET_SIGMA_SECONDS = 0.025
MIN_DISTANCE_SECONDS = 0.055
SCALAR_TRACKS = (
    "likelihood", "envelope", "rms_db", "spectral_sonority",
    "periodicity", "vowel_likeness", "zcr_score", "flatness_score",
    "flux_score", "sonorant_penalty", "spectral_entropy",
    "formant_energy_db", "energy_entropy_ratio",
)


@dataclass
class SequenceData:
    name: str
    times: np.ndarray
    features: np.ndarray
    target: np.ndarray
    speech: np.ndarray
    reference: list[float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def align_matrix(values: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] > frames:
        return values[:, :frames]
    if values.shape[1] < frames:
        return np.pad(values, ((0, 0), (0, frames - values.shape[1])), mode="edge")
    return values


def build_sequence(root: Path, name: str) -> SequenceData:
    signal, sample_rate = librosa.load(
        find_audio(root / "audio", name), sr=16000, mono=True
    )
    detection = detect_nuclei_v2(signal, sample_rate, production_v2_config())
    reference, span = annotation(
        root / "analysis" / "v2_annotations" / f"{name}_v2_reviewed.TextGrid"
    )
    frames = len(detection.frame_times)
    tracks = dict(detection.feature_tracks)
    tracks["likelihood"] = detection.likelihood
    mel = align_matrix(tracks["mel_z"], frames)
    mfcc = align_matrix(tracks["mfcc"], frames)
    # Width five corresponds to a short 50 ms dynamic window.  Deltas and
    # delta-deltas capture the local rise/fall that distinguishes a nucleus
    # from a merely energetic consonant.
    delta = librosa.feature.delta(mfcc, width=5, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, width=5, order=2, mode="nearest")
    scalars = np.vstack([
        align_matrix(tracks[key], frames)[0] for key in SCALAR_TRACKS
    ])
    bands = align_matrix(tracks["band_z"], frames)
    features = np.vstack([mel, mfcc, delta, delta2, scalars, bands]).T

    selected = (
        (detection.frame_times >= span[0])
        & (detection.frame_times <= span[1])
    )
    times = detection.frame_times[selected].astype(np.float32)
    features = features[selected].astype(np.float32)
    speech = detection.speech_mask[selected].astype(bool)
    reference = [value for value in reference if span[0] <= value <= span[1]]
    target = np.zeros(len(times), dtype=np.float32)
    for value in reference:
        target = np.maximum(
            target,
            np.exp(-0.5 * ((times - value) / TARGET_SIGMA_SECONDS) ** 2),
        )
    return SequenceData(name, times, features, target, speech, reference)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.conv = nn.Conv1d(
            width, width, kernel_size=5, dilation=dilation, padding=padding
        )
        self.norm = nn.GroupNorm(4, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.conv(values)
        update = self.dropout(F.gelu(self.norm(update)))
        return values + update


class TemporalRegressor(nn.Module):
    """Small dilated convolutional network with about 0.65 s context."""

    def __init__(self, features: int, width: int = 40) -> None:
        super().__init__()
        self.input = nn.Conv1d(features, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(width, dilation, 0.12)
            for dilation in (1, 2, 4, 8)
        ])
        self.output = nn.Conv1d(width, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.gelu(self.input(values))
        for block in self.blocks:
            values = block(values)
        return self.output(values)[:, 0, :]


def normalization(training: list[SequenceData]) -> tuple[np.ndarray, np.ndarray]:
    selected = np.vstack([
        item.features[item.speech] if np.any(item.speech) else item.features
        for item in training
    ])
    mean = np.mean(selected, axis=0)
    scale = np.std(selected, axis=0)
    scale[scale < 1e-4] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def train_model(
    training: list[SequenceData], seed: int, epochs: int = 72,
) -> tuple[TemporalRegressor, np.ndarray, np.ndarray]:
    seed_everything(seed)
    mean, scale = normalization(training)
    model = TemporalRegressor(training[0].features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=2e-4
    )
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(training))
        for index in order:
            item = training[int(index)]
            standardized = (item.features - mean) / scale
            # Mild feature noise and channel masking simulate microphones and
            # discourage dependence on one spectral coefficient.
            noisy = standardized + generator.normal(
                0.0, 0.025, standardized.shape
            ).astype(np.float32)
            if generator.random() < 0.7:
                dropped = generator.choice(
                    noisy.shape[1], size=max(1, noisy.shape[1] // 16), replace=False
                )
                noisy[:, dropped] = 0.0
            inputs = torch.from_numpy(noisy.T[None, :, :])
            targets = torch.from_numpy(item.target[None, :])
            logits = model(inputs)
            weights = 0.7 + 4.3 * targets
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                ) * weights
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        if epoch == 47:
            for group in optimizer.param_groups:
                group["lr"] *= 0.35
    return model.eval(), mean, scale


def predict(
    model: TemporalRegressor, item: SequenceData,
    mean: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    inputs = torch.from_numpy(((item.features - mean) / scale).T[None, :, :])
    with torch.inference_mode():
        values = torch.sigmoid(model(inputs))[0].numpy()
    values = gaussian_filter1d(values, sigma=0.7, mode="nearest")
    values[~item.speech] = 0.0
    return values


def events(
    item: SequenceData, values: np.ndarray,
    threshold: float, prominence: float,
) -> list[float]:
    hop = float(np.median(np.diff(item.times))) if len(item.times) > 1 else 0.010
    distance = max(1, int(round(MIN_DISTANCE_SECONDS / hop)))
    peaks, _ = find_peaks(
        values, height=threshold, prominence=prominence, distance=distance
    )
    return item.times[peaks].astype(float).tolist()


def pooled(items: list[dict]) -> dict:
    tp = sum(item["tp"] for item in items)
    fp = sum(item["fp"] for item in items)
    fn = sum(item["fn"] for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def compact(metrics: dict) -> dict:
    return {key: metrics[key] for key in (
        "tp", "fp", "fn", "precision", "recall", "f1", "timing_median_ms"
    )}


def choose_event_rule(
    training: list[SequenceData], outer_index: int,
) -> tuple[float, float, dict]:
    inner_predictions: list[tuple[SequenceData, np.ndarray]] = []
    for held_index, held in enumerate(training):
        inner_train = [
            item for index, item in enumerate(training) if index != held_index
        ]
        model, mean, scale = train_model(
            inner_train, SEED + 100 * outer_index + held_index, epochs=64
        )
        inner_predictions.append((held, predict(model, held, mean, scale)))
    combined = np.concatenate([values for _, values in inner_predictions])
    positive = combined[combined > 0]
    threshold_grid = np.unique(np.concatenate([
        np.linspace(0.10, 0.80, 29),
        np.quantile(positive, np.linspace(0.15, 0.90, 24))
        if len(positive) else np.array([0.5]),
    ]))
    best = None
    for threshold in threshold_grid:
        for prominence in (0.00, 0.025, 0.05, 0.075, 0.10, 0.14):
            metrics = [
                match_events(
                    events(item, values, float(threshold), prominence),
                    item.reference,
                    TOLERANCE_SECONDS,
                )
                for item, values in inner_predictions
            ]
            aggregate = pooled(metrics)
            rank = (
                aggregate["f1"], aggregate["recall"],
                aggregate["precision"], -float(threshold),
            )
            if best is None or rank > best[0]:
                best = (rank, float(threshold), prominence, aggregate)
    assert best is not None
    return best[1], best[2], best[3]


def evaluate(recordings: list[SequenceData]) -> dict:
    folds = []
    for held_index, held in enumerate(recordings):
        training = [
            item for index, item in enumerate(recordings) if index != held_index
        ]
        threshold, prominence, inner = choose_event_rule(training, held_index)
        model, mean, scale = train_model(
            training, SEED + 1000 + held_index, epochs=72
        )
        values = predict(model, held, mean, scale)
        predicted = events(held, values, threshold, prominence)
        metrics = match_events(predicted, held.reference, TOLERANCE_SECONDS)
        fold = {
            "heldout": held.name,
            "threshold": threshold,
            "prominence": prominence,
            "inner_validation": inner,
            "reference_count": len(held.reference),
            "predicted_count": len(predicted),
            "metrics_50ms": compact(metrics),
        }
        folds.append(fold)
        print(
            f"{held.name}: P={metrics['precision']:.3f} "
            f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )
    return {
        "folds": folds,
        "aggregate_50ms": pooled([item["metrics_50ms"] for item in folds]),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    recordings = [build_sequence(root, name) for name in ("3local", "6fori", "7fori")]
    for item in recordings:
        print(
            f"{item.name}: {len(item.times)} frames, "
            f"{item.features.shape[1]} features, {len(item.reference)} nuclei"
        )
    result = evaluate(recordings)
    payload = {
        "experiment": "mel_mfcc_temporal_frame_regression",
        "text_independent": True,
        "evaluation": "nested leave-one-recording-out; +/-50 ms",
        "target_sigma_seconds": TARGET_SIGMA_SECONDS,
        "feature_count": int(recordings[0].features.shape[1]),
        "result": result,
    }
    output = root / "results" / "nuclei_v3_temporal_experiment.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
