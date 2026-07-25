import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from audio_input import is_analysis_wav, prepare_audio
from asr_backends import RecognitionResult, merge_whisper_text_with_vosk_times
from pronunciationAnalyzer import saveJson
from benchmark_asr import metrics


def write_silence(path, sample_rate, channels, seconds=0.1):
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames * channels)


class AudioPreparationTests(unittest.TestCase):
    def test_compliant_wav_is_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "готово.wav"
            write_silence(source, 16000, 1)
            prepared = prepare_audio(source)
            self.assertFalse(prepared.converted)
            self.assertEqual(Path(prepared.analysis_path), source.resolve())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_stereo_44100_is_converted_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "телефон stereo.wav"
            write_silence(source, 44100, 2)
            prepared = prepare_audio(source, cache_dir=tmp_path / "cache")
            self.assertTrue(prepared.converted)
            self.assertTrue(is_analysis_wav(prepared.analysis_path))


class FreeModeJsonTests(unittest.TestCase):
    def test_free_mode_has_no_fake_word_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = str(Path(tmp) / "speech.m4a")
            comparison = {
                "correct": [], "substituted": [], "missed": [], "inserted": []
            }
            scores = {"syl": 50, "stress": 50, "redux": 50,
                      "tempo": 50, "word": None}
            result_path = saveJson(
                source, 1.0, 16000, "", "пример", [], [], [], [],
                comparison, scores=scores, outDir=tmp, audioConverted=True)
            with open(result_path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["analysisMode"], "free")
            self.assertIsNone(data["evaluationScores"]["wordAccuracy"])
            self.assertTrue(result_path.endswith("speech_syllable_analysis.json"))


class HybridTimestampTests(unittest.TestCase):
    def test_whisper_text_keeps_vosk_time_anchors(self):
        whisper = RecognitionResult(
            "whisper", "новое слово",
            [{"word": "новое", "start": 0.0, "end": 0.6, "conf": 0.9},
             {"word": "слово", "start": 0.6, "end": 1.2, "conf": 0.8}],
            "cuda")
        vosk = RecognitionResult(
            "vosk", "новая слова",
            [{"word": "новая", "start": 0.1, "end": 0.5, "conf": 0.7},
             {"word": "слова", "start": 0.55, "end": 1.0, "conf": 0.7}],
            "cpu")
        merged = merge_whisper_text_with_vosk_times(whisper, vosk)
        self.assertEqual(merged.text, "новое слово")
        self.assertEqual(merged.words[0]["start"], 0.1)
        self.assertEqual(merged.words[1]["end"], 1.0)
        self.assertEqual(merged.words[0]["timingSource"], "vosk")

    def test_wer_counts_word_substitutions(self):
        result = metrics("один два три", "один четыре пять")
        self.assertEqual(result["referenceWords"], 3)
        self.assertEqual(result["errors"], 2)
        self.assertAlmostEqual(result["werPct"], 66.67, places=2)


if __name__ == "__main__":
    unittest.main()
