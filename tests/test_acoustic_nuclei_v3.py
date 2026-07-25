import unittest

from acoustic_nuclei_v3 import (
    NucleusV3Detection,
    apply_asr_overcount_pruning,
)


class AcousticNucleiV3Tests(unittest.TestCase):
    def test_asr_pruner_removes_only_lowest_quality_overcount(self):
        detection = NucleusV3Detection(
            duration_seconds=1.0,
            times=[0.10, 0.20, 0.30, 0.90],
            event_scores=[0.9, 0.1, 0.8, 0.2],
            v2_times=[0.10, 0.20],
            sylber_candidate_times=[],
            sylber_candidate_scores=[],
            rescued_times=[0.30, 0.90],
        )
        payload = {
            "wordAnalysis": [{
                "start": 0.0,
                "end": 0.5,
                "syllableCount": 1,
            }]
        }
        result = apply_asr_overcount_pruning(
            detection, payload, quota_slack=1
        )
        self.assertEqual(result.times, [0.10, 0.30, 0.90])
        self.assertEqual(result.pruned_times, [0.20])
        self.assertEqual(result.rescued_times, [0.30, 0.90])
        self.assertTrue(result.asr_pruned)

    def test_asr_pruner_preserves_events_outside_recognized_words(self):
        detection = NucleusV3Detection(
            duration_seconds=1.0,
            times=[0.80],
            event_scores=[0.2],
            v2_times=[],
            sylber_candidate_times=[],
            sylber_candidate_scores=[],
            rescued_times=[0.80],
        )
        result = apply_asr_overcount_pruning(
            detection,
            {"wordAnalysis": [{
                "start": 0.0, "end": 0.5, "syllableCount": 1,
            }]},
        )
        self.assertEqual(result.times, [0.80])
        self.assertEqual(result.pruned_times, [])

    def test_asr_pruner_rejects_events_inside_vowelless_asr_token(self):
        detection = NucleusV3Detection(
            duration_seconds=1.0,
            times=[0.20, 0.80],
            event_scores=[0.9, 0.2],
            v2_times=[0.20],
            sylber_candidate_times=[],
            sylber_candidate_scores=[],
            rescued_times=[0.80],
        )
        result = apply_asr_overcount_pruning(
            detection,
            {"wordAnalysis": [{
                "word": "в",
                "start": 0.0,
                "end": 0.5,
                "syllableCount": 1,
            }]},
            quota_slack=1,
        )
        self.assertEqual(result.times, [0.80])
        self.assertEqual(result.pruned_times, [0.20])


if __name__ == "__main__":
    unittest.main()
