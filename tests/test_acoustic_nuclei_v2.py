import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from acoustic_nuclei_v2 import (
    PRODUCTION_V2_CONFIG_NAME,
    NucleusV2Config,
    detect_nuclei_v2,
    production_v2_config,
)
from evaluate_nuclei_v2 import match_events
from nuclei_annotations import (
    TextGridInterval,
    TextGridPoint,
    append_interval_tier,
    append_point_tier,
    read_interval_tier,
    read_point_tier,
    read_textgrid_file,
)
from prepare_nucleus_annotations import has_written_vowel


class AcousticNucleiV2Tests(unittest.TestCase):
    def test_production_config_is_frozen_calibrated_preset(self):
        config = production_v2_config()
        self.assertEqual(PRODUCTION_V2_CONFIG_NAME, "recall_4_noise_gate")
        self.assertEqual(config.strong_prominence, 0.35)
        self.assertEqual(config.weak_prominence, 0.08)
        self.assertEqual(config.strong_min_strength, 0.4)
        self.assertEqual(config.strong_min_vowel_likeness, 0.45)

    def test_detector_is_text_independent_and_finds_synthetic_bursts(self):
        sr = 16000
        duration = 1.4
        time = np.arange(int(sr * duration)) / sr
        signal = np.zeros_like(time)
        expected = (0.30, 0.70, 1.10)
        for centre, frequency in zip(expected, (180.0, 220.0, 195.0)):
            envelope = np.exp(-0.5 * ((time - centre) / 0.055) ** 2)
            signal += 0.45 * envelope * (
                np.sin(2 * math.pi * frequency * time)
                + 0.35 * np.sin(2 * math.pi * 2 * frequency * time)
                + 0.20 * np.sin(2 * math.pi * 3 * frequency * time)
            )
        result = detect_nuclei_v2(signal.astype(np.float32), sr, NucleusV2Config())
        self.assertEqual(len(result.nuclei), 3)
        for detected, reference in zip(result.times, expected):
            self.assertLess(abs(detected - reference), 0.06)

    def test_strong_noise_gate_can_reject_nonperiodic_peak(self):
        config = NucleusV2Config(strong_min_periodicity=0.95)
        sr = config.sample_rate
        rng = np.random.default_rng(7)
        signal = np.zeros(sr, dtype=np.float32)
        signal[4000:5200] = rng.normal(0.0, 0.3, 1200)
        result = detect_nuclei_v2(signal, sr, config)
        self.assertEqual(result.times, [])

    def test_textgrid_point_tier_roundtrip(self):
        base = (
            'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
            'xmin = 0\nxmax = 1.5\ntiers? <exists>\nsize = 0\nitem []:\n'
        )
        rendered = append_point_tier(
            base,
            "nucleus",
            [TextGridPoint(0.2, "N"), TextGridPoint(0.8, "?")],
        )
        points = read_point_tier(rendered, "nucleus")
        self.assertEqual(points, [TextGridPoint(0.2, "N"), TextGridPoint(0.8, "?")])

    def test_textgrid_interval_hint_tier_roundtrip(self):
        base = (
            'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
            'xmin = 0\nxmax = 1.5\ntiers? <exists>\nsize = 0\nitem []:\n'
        )
        expected = [
            TextGridInterval(0.0, 0.2, ""),
            TextGridInterval(0.2, 0.8, "слово"),
            TextGridInterval(0.8, 1.5, ""),
        ]
        rendered = append_interval_tier(base, "word_hint", expected)
        self.assertEqual(
            read_interval_tier(rendered, "word_hint", include_empty=True),
            expected,
        )

    def test_event_matching_is_one_to_one(self):
        metrics = match_events([0.10, 0.13, 0.50], [0.11, 0.49], 0.03)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["false_positive_times_seconds"], [0.13])
        self.assertEqual(metrics["missed_reference_times_seconds"], [])

    def test_vowelless_preposition_is_not_an_acoustic_nucleus(self):
        self.assertFalse(has_written_vowel("в"))
        self.assertFalse(has_written_vowel("с"))
        self.assertTrue(has_written_vowel("во"))

    def test_textgrid_reader_accepts_utf16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.TextGrid"
            path.write_text("тест", encoding="utf-16")
            self.assertEqual(read_textgrid_file(path), "тест")


if __name__ == "__main__":
    unittest.main()
