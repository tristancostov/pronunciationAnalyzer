#!/usr/bin/env python3
"""Grid-search syllable-nucleus parameters on Praat annotations.

The script reuses existing VOSK word timestamps, so it does not load the
multi-gigabyte recognition model. It optimizes the *raw acoustic nucleus
count* and reports leave-one-recording-out performance to make overfitting
visible. No analyzer JSON is overwritten.
"""

import argparse
import itertools
import json
import os
from statistics import mean, median

import librosa
import numpy as np
from scipy.signal import find_peaks

import pronunciationAnalyzer as pa
from EvaluateAnnotations import (
    align_system_to_ground,
    allocate_syllables_to_words,
    load_ground_truth,
    read_textgrid,
)


def parse_args():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", default=["3local", "6fori", "7fori"])
    parser.add_argument("--audio-dir", default=os.path.join(root, "audio"))
    parser.add_argument("--textgrid-dir", default=os.path.join(root, "analysis"))
    parser.add_argument("--json-dir", default=os.path.join(root, "analysis"))
    parser.add_argument("--ground", default=os.path.join(root, "ground_truth_words.json"))
    parser.add_argument("--output", default=os.path.join(root, "results", "nuclei_calibration.json"))
    return parser.parse_args()


def prepare_items(args):
    ground_truth = load_ground_truth(args.ground)
    items = []
    for recording in args.recordings:
        manual_syllables = read_textgrid(
            os.path.join(args.textgrid_dir, f"{recording}.TextGrid")
        )
        manual_words, _ = allocate_syllables_to_words(
            manual_syllables, ground_truth[recording]
        )
        with open(os.path.join(args.json_dir, f"{recording}_syllable_analysis.json"),
                  encoding="utf-8") as source:
            analysis = json.load(source)
        raw_words = analysis["wordAnalysis"]
        system_words = [
            {"word": word["word"], "raw_index": index}
            for index, word in enumerate(raw_words)
        ]
        pairs = align_system_to_ground(system_words, manual_words)
        signal, sr = librosa.load(
            os.path.join(args.audio_dir, f"{recording}.wav"),
            sr=pa.SR, mono=True, res_type="kaiser_fast"
        )

        for manual_word, system_word, _ in pairs:
            if manual_word is None or system_word is None:
                continue
            word_index = system_word["raw_index"]
            raw_word = raw_words[word_index]
            start = float(raw_word["start"])
            end = float(raw_word["end"])
            pad_end = 0.080
            if word_index + 1 < len(raw_words):
                next_start = float(raw_words[word_index + 1].get("start", end))
                pad_end = min(pad_end, max(0.0, (next_start - end) * 0.5))
            segment = signal[int(start * sr):min(len(signal), int((end + pad_end) * sr))]
            if len(segment) < int(sr * 0.04):
                continue
            trim_start, trim_end = pa.trimSilenceBounds(segment, sr, floorDb=40.0)
            core = segment[trim_start:trim_end]
            contours = pa.computeContours(core, sr)
            items.append({
                "recording": recording,
                "word": manual_word["name"],
                "manual_syllables": manual_word["sylls"],
                "manual_count": len(manual_word["sylls"]),
                "text_count": int(raw_word["syllableCount"]),
                "word_start": start,
                "trim_start": trim_start,
                "core_samples": len(core),
                "sr": sr,
                "contours": contours,
            })
    return items


def select_with_params(item, params):
    contours = item["contours"]
    energy = contours["energyDb"]
    voiced = contours["voiced"]
    vowel_score = contours["vowelScore"]
    hop = contours["hopLen"]
    sr = item["sr"]
    if len(energy) == 0:
        return np.array([], dtype=int), 0

    contour = energy.astype(float).copy()
    contour += (vowel_score - 0.5) * 2.0 * params["gain_db"]
    contour[~voiced] = contour.min() - 10.0
    floor = float(np.max(energy)) - params["floor_db"]
    min_distance = max(1, int(params["distance_s"] * sr / hop))

    raw_peaks, _ = find_peaks(
        contour,
        prominence=params["dip_db"],
        distance=min_distance,
        height=floor,
    )
    candidates, props = find_peaks(
        contour, prominence=0.5, distance=min_distance, height=floor
    )
    if len(candidates) == 0:
        return np.array([int(np.argmax(contour))]), len(raw_peaks)

    score = (
        pa._norm(energy[candidates]) +
        pa._norm(props["prominences"]) +
        0.8 * pa._norm(vowel_score[candidates])
    )
    expected = item["text_count"]
    if len(candidates) >= expected:
        candidates = candidates[np.argsort(score)[-expected:]]
    return np.sort(candidates), len(raw_peaks)


def evaluate(items, params, recordings=None):
    selected_items = [
        item for item in items
        if recordings is None or item["recording"] in recordings
    ]
    by_recording = {}
    boundary_errors = []
    for item in selected_items:
        peaks, raw_count = select_with_params(item, params)
        stats = by_recording.setdefault(item["recording"], {"hits": 0, "total": 0})
        stats["total"] += 1
        stats["hits"] += int(raw_count == item["manual_count"])

        if item["text_count"] != item["manual_count"]:
            continue
        bounds = pa.nucleiToBoundaries(
            peaks,
            item["contours"]["energyDb"],
            item["contours"]["hopLen"],
            item["core_samples"],
            item["sr"],
            item["text_count"],
            item["contours"]["vowelScore"],
        )
        absolute_bounds = [
            (
                item["word_start"] + (start + item["trim_start"]) / item["sr"],
                item["word_start"] + (end + item["trim_start"]) / item["sr"],
            )
            for start, end in bounds
        ]
        for index, manual_syllable in enumerate(item["manual_syllables"]):
            boundary_errors.append(abs(manual_syllable[0] - absolute_bounds[index][0]) * 1000)
        boundary_errors.append(
            abs(item["manual_syllables"][-1][1] - absolute_bounds[-1][1]) * 1000
        )

    accuracies = {
        name: values["hits"] / values["total"] * 100
        for name, values in by_recording.items()
    }
    return {
        "balanced_count_accuracy": mean(accuracies.values()) if accuracies else 0.0,
        "overall_count_accuracy": (
            sum(v["hits"] for v in by_recording.values()) /
            sum(v["total"] for v in by_recording.values()) * 100
            if by_recording else 0.0
        ),
        "by_recording": accuracies,
        "boundary_median_ms": median(boundary_errors) if boundary_errors else 0.0,
        "boundary_mean_ms": mean(boundary_errors) if boundary_errors else 0.0,
    }


def parameter_grid():
    for dip, distance, gain, floor in itertools.product(
        (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        (0.06, 0.08, 0.10, 0.12, 0.14),
        (0.0, 2.0, 4.0, 6.0, 8.0),
        (20.0, 25.0, 30.0, 35.0, 40.0),
    ):
        yield {
            "dip_db": dip,
            "distance_s": distance,
            "gain_db": gain,
            "floor_db": floor,
        }


def rank_key(result):
    # Count accuracy is primary; boundary quality breaks near-ties.
    return (result["balanced_count_accuracy"], -result["boundary_median_ms"])


def main():
    args = parse_args()
    items = prepare_items(args)
    candidates = []
    for params in parameter_grid():
        metrics = evaluate(items, params)
        candidates.append({"params": params, **metrics})
    candidates.sort(key=rank_key, reverse=True)

    current_params = {
        "dip_db": pa.DIP_DB,
        "distance_s": pa.MIN_NUCLEUS_DIST_S,
        "gain_db": pa.VOWEL_PEAK_GAIN_DB,
        "floor_db": pa.PEAK_FLOOR_DB,
    }
    current = {"params": current_params, **evaluate(items, current_params)}

    loocv = []
    recordings = sorted({item["recording"] for item in items})
    for held_out in recordings:
        train = [name for name in recordings if name != held_out]
        ranked = []
        for params in parameter_grid():
            metrics = evaluate(items, params, train)
            ranked.append((rank_key(metrics), params, metrics))
        ranked.sort(key=lambda row: row[0], reverse=True)
        _, params, train_metrics = ranked[0]
        loocv.append({
            "held_out": held_out,
            "params": params,
            "train": train_metrics,
            "test": evaluate(items, params, [held_out]),
        })

    output = {
        "recordings": recordings,
        "word_count": len(items),
        "current": current,
        "best": candidates[0],
        "top_10": candidates[:10],
        "leave_one_recording_out": loocv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
