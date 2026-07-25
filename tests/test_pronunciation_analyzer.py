import unittest

from pronunciationAnalyzer import (
    assignV2NucleiToWords,
    findStressedSyllableByDict,
    splitIntoSyllables,
)
from EvaluateAnnotations import manually_marked_stress_idx


class FakeAccentizer:
    def __init__(self, accented):
        self.accented = accented

    def process_all(self, _word):
        return self.accented


class SyllabificationTests(unittest.TestCase):
    def test_each_syllable_has_one_vowel(self):
        cases = {
            "эфир": ["э", "фир"],
            "воспроизвести": ["вос", "про", "из", "вес", "ти"],
            "видео": ["ви", "де", "о"],
            "организации": ["ор", "га", "ни", "за", "ци", "и"],
            "сегодня": ["се", "го", "дня"],
            "семья": ["се", "мья"],
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(splitIntoSyllables(word), expected)


class DictionaryStressTests(unittest.TestCase):
    def test_plus_before_stressed_vowel(self):
        syllables = splitIntoSyllables("молоко")
        accentizer = FakeAccentizer("молок+о")
        self.assertEqual(findStressedSyllableByDict("молоко", syllables, accentizer), 2)

    def test_combining_acute_after_stressed_vowel(self):
        syllables = splitIntoSyllables("молоко")
        accentizer = FakeAccentizer("молоко\u0301")
        self.assertEqual(findStressedSyllableByDict("молоко", syllables, accentizer), 2)

    def test_yo_is_stressed_without_extra_marker(self):
        syllables = splitIntoSyllables("проведём")
        accentizer = FakeAccentizer("проведём")
        self.assertEqual(findStressedSyllableByDict("проведём", syllables, accentizer), 2)


class ManualStressTests(unittest.TestCase):
    def test_requires_explicit_marker(self):
        syllables = [(0.0, 0.1, "мо"), (0.1, 0.2, "ло"), (0.2, 0.3, "ко")]
        self.assertIsNone(manually_marked_stress_idx(syllables))

    def test_accepts_plus_and_yo(self):
        plus = [(0.0, 0.1, "мо"), (0.1, 0.2, "л+о"), (0.2, 0.3, "ко")]
        yo = [(0.0, 0.1, "про"), (0.1, 0.2, "ве"), (0.2, 0.3, "дём")]
        self.assertEqual(manually_marked_stress_idx(plus), 1)
        self.assertEqual(manually_marked_stress_idx(yo), 2)


class V2WordAssignmentTests(unittest.TestCase):
    def test_candidates_are_assigned_once_without_using_text_counts(self):
        words = [
            {"word": "одно", "start": 0.10, "end": 0.45},
            {"word": "два", "start": 0.55, "end": 0.85},
        ]
        candidates = [
            {"time_seconds": 0.20, "confidence": 0.8},
            # In the ASR gap, but within 80 ms of the second word.
            {"time_seconds": 0.51, "confidence": 0.7},
            {"time_seconds": 1.20, "confidence": 0.9},
        ]
        assigned, unassigned = assignV2NucleiToWords(words, candidates)
        self.assertEqual([len(group) for group in assigned], [1, 1])
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(assigned[1][0]["timeSec"], 0.51)
        self.assertAlmostEqual(assigned[1][0]["timeInWordSec"], -0.04)

    def test_overlapping_word_intervals_do_not_duplicate_candidate(self):
        words = [
            {"start": 0.0, "end": 0.6},
            {"start": 0.5, "end": 1.0},
        ]
        assigned, unassigned = assignV2NucleiToWords(
            words, [{"time_seconds": 0.55, "confidence": 0.8}])
        self.assertEqual(sum(map(len, assigned)), 1)
        self.assertEqual(unassigned, [])


if __name__ == "__main__":
    unittest.main()
