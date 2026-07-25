#!/usr/bin/env python3
"""Evaluate pretrained Sylber on reviewed Russian nucleus points.

Sylber (ICLR 2025) is a speech-only, self-supervised syllabic representation.
This script deliberately uses its acoustic hidden states without ASR or text.
It tests both segment midpoints and a hybrid in which the strongest V2 vowel
likelihood frame inside each Sylber segment becomes the nucleus location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import librosa
import numpy as np
import torch
from transformers import HubertConfig, HubertModel

from acoustic_nuclei_v2 import detect_nuclei_v2, production_v2_config
from evaluate_nuclei_v2 import match_events
from nuclei_annotations import (
    read_interval_tier,
    read_point_tier,
    read_textgrid_file,
)
from prepare_nucleus_annotations import find_audio


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=-1) / (
        np.sqrt(np.sum(left * left, axis=-1) + 1e-8)
        * np.sqrt(np.sum(right * right, axis=-1) + 1e-8)
    )


def sylber_segments(
    states: np.ndarray, norm_threshold: float, merge_threshold: float,
) -> np.ndarray:
    """Linear greedy segmentation from the official Sylber inference code."""
    norms = np.sqrt(np.sum(states * states, axis=-1) + 1e-8)
    speech = norms >= norm_threshold
    segments = []
    start = -1
    centre = None
    count = 0
    for frame, is_speech in enumerate(speech):
        if not is_speech:
            if start >= 0:
                segments.append([start, frame])
            start, centre, count = -1, None, 0
            continue
        if start < 0:
            start, centre, count = frame, states[frame].copy(), 1
            continue
        similarity = float(cosine(centre, states[frame]))
        if similarity >= merge_threshold:
            centre = (centre * count + states[frame]) / (count + 1)
            count += 1
        else:
            segments.append([start, frame])
            start, centre, count = frame, states[frame].copy(), 1
    if start >= 0:
        segments.append([start, len(states)])
    return np.asarray(segments, dtype=int).reshape(-1, 2)


def load_model(device: str) -> HubertModel:
    config = HubertConfig.from_pretrained(
        "facebook/hubert-base-ls960", num_hidden_layers=9
    )
    model = HubertModel(config)
    checkpoint = hf_hub_download(
        repo_id="cheoljun95/sylber", filename="sylber.ckpt"
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def encode(model: HubertModel, signal: np.ndarray, device: str) -> np.ndarray:
    normalized = (signal - float(np.mean(signal))) / max(float(np.std(signal)), 1e-6)
    tensor = torch.from_numpy(normalized.astype(np.float32))[None, :].to(device)
    with torch.inference_mode():
        return model(tensor).last_hidden_state[0].cpu().numpy()


def annotation(path: Path) -> tuple[list[float], tuple[float, float]]:
    grid = read_textgrid_file(path)
    reference = [
        point.time for point in read_point_tier(grid, "nucleus")
        if point.mark.strip() and "?" not in point.mark
    ]
    intervals = read_interval_tier(grid, "syllable", include_empty=True)
    nonempty = [item for item in intervals if item.text.strip()]
    selected = nonempty or intervals
    return reference, (
        min(item.start for item in selected),
        max(item.end for item in selected),
    )


def event_times(
    segments: np.ndarray,
    hidden_duration: float,
    hidden_frame_count: int,
    method: str,
    v2_detection,
) -> list[float]:
    # HuBERT's convolutional frontend emits approximately 50 frames/second.
    # The exact ratio below removes accumulated drift at the end of a file.
    if len(segments) == 0:
        return []
    seconds_per_frame = hidden_duration / max(hidden_frame_count, 1)
    events = []
    for start_frame, end_frame in segments:
        start = float(start_frame) * seconds_per_frame
        end = float(end_frame) * seconds_per_frame
        if end <= start:
            continue
        if method == "midpoint":
            events.append((start + end) / 2.0)
            continue
        indices = np.flatnonzero(
            (v2_detection.frame_times >= start)
            & (v2_detection.frame_times <= end)
        )
        if len(indices):
            best = indices[int(np.argmax(v2_detection.likelihood[indices]))]
            events.append(float(v2_detection.frame_times[best]))
        else:
            events.append((start + end) / 2.0)
    return events


def aggregate(items: list[dict]) -> dict:
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


def priority_union(primary: list[float], secondary: list[float], distance: float = 0.055) -> list[float]:
    kept = list(primary)
    for time in secondary:
        if all(abs(time - existing) >= distance for existing in kept):
            kept.append(float(time))
    return sorted(kept)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=["3local", "6fori", "7fori"])
    parser.add_argument(
        "--output", type=Path,
        default=root / "results" / "nuclei_v3_sylber_zero_shot.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Sylber on {device}…")
    model = load_model(device)
    data = {}
    for name in args.recordings:
        audio = find_audio(root / "audio", name)
        signal, sample_rate = librosa.load(audio, sr=16000, mono=True)
        states = encode(model, signal, device)
        reference, span = annotation(
            root / "analysis" / "v2_annotations" / f"{name}_v2_reviewed.TextGrid"
        )
        v2 = detect_nuclei_v2(signal, sample_rate, production_v2_config())
        data[name] = {
            "states": states,
            "duration": len(signal) / sample_rate,
            "reference": reference,
            "span": span,
            "v2": v2,
        }
        print(f"{name}: {len(states)} Sylber frames")

    grid = [
        (round(norm, 2), round(merge, 2))
        for norm in np.arange(2.2, 3.01, 0.1)
        for merge in np.arange(0.60, 0.91, 0.05)
    ]
    predictions = {}
    for name, item in data.items():
        predictions[name] = {}
        for norm, merge in grid:
            segments = sylber_segments(item["states"], norm, merge)
            key = f"norm={norm:.2f},merge={merge:.2f}"
            predictions[name][key] = {}
            for method in ("midpoint", "v2_peak_inside_segment"):
                events = event_times(
                    segments, item["duration"], len(item["states"]),
                    method, item["v2"]
                )
                events = [time for time in events if item["span"][0] <= time <= item["span"][1]]
                predictions[name][key][method] = events

    results = {}
    for method in ("midpoint", "v2_peak_inside_segment"):
        folds = []
        for heldout in args.recordings:
            development = [name for name in args.recordings if name != heldout]
            best = None
            for norm, merge in grid:
                key = f"norm={norm:.2f},merge={merge:.2f}"
                dev_metrics = [
                    match_events(
                        predictions[name][key][method], data[name]["reference"], 0.050
                    ) for name in development
                ]
                pooled = aggregate(dev_metrics)
                rank = (pooled["f1"], pooled["precision"], pooled["recall"])
                if best is None or rank > best[0]:
                    best = (rank, key, pooled)
            assert best is not None
            metrics = match_events(
                predictions[heldout][best[1]][method],
                data[heldout]["reference"], 0.050,
            )
            sylber_events = predictions[heldout][best[1]][method]
            base_events = [
                float(time) for time in data[heldout]["v2"].times
                if data[heldout]["span"][0] <= time <= data[heldout]["span"][1]
            ]
            v2_priority = priority_union(base_events, sylber_events)
            sylber_priority = priority_union(sylber_events, base_events)
            folds.append({
                "heldout": heldout,
                "selected_on_other_speakers": best[1],
                "development_metrics": best[2],
                "predicted_count": len(predictions[heldout][best[1]][method]),
                "metrics_50ms": compact(metrics),
                "v2_priority_union": compact(match_events(
                    v2_priority, data[heldout]["reference"], 0.050
                )),
                "sylber_priority_union": compact(match_events(
                    sylber_priority, data[heldout]["reference"], 0.050
                )),
            })
        results[method] = {
            "folds": folds,
            "aggregate_50ms": aggregate([fold["metrics_50ms"] for fold in folds]),
            "v2_priority_union_aggregate_50ms": aggregate([
                fold["v2_priority_union"] for fold in folds
            ]),
            "sylber_priority_union_aggregate_50ms": aggregate([
                fold["sylber_priority_union"] for fold in folds
            ]),
        }

    payload = {
        "experiment": "pretrained_sylber_zero_shot_russian",
        "text_independent": True,
        "device": device,
        "evaluation": "leave-one-recording-out threshold selection; ±50 ms",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
