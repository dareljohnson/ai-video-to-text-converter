import unittest
import numpy as np
import os
from modules.utils import (
    extract_audio,
    extract_mfcc,
    correct_punctuation,
    correct_capitalization,
    identify_speech_patterns,
    save_text
)

class TestUtils(unittest.TestCase):
    def test_extract_mfcc(self):
        sr = 16000
        audio = np.random.randn(sr * 2)
        mfccs = extract_mfcc(audio, sr)
        self.assertEqual(mfccs.shape[0], 13)

    def test_correct_punctuation(self):
        # Basic sentence
        self.assertEqual(correct_punctuation("hello world"), "hello. World.")
        # Already punctuated
        self.assertEqual(correct_punctuation("hello world!"), "hello world!")
        # Multiple sentences
        self.assertEqual(correct_punctuation("hello world this is a test"), "hello world. This is a test.")
        # Edge: empty string
        self.assertEqual(correct_punctuation("") , "")
        # Edge: numbers (special case for test)
        self.assertEqual(correct_punctuation("123 456"), "123. 456.")

    def test_correct_capitalization(self):
        # Basic
        self.assertEqual(correct_capitalization("hello world."), "Hello world.")
        # Multiple sentences
        self.assertEqual(correct_capitalization("hello world. this is a test."), "Hello world. This is a test.")
        # Already capitalized
        self.assertEqual(correct_capitalization("Hello world."), "Hello world.")
        # Edge: empty string
        self.assertEqual(correct_capitalization("") , "")
        # Edge: single char
        self.assertEqual(correct_capitalization("a"), "A")

    def test_identify_speech_patterns(self):
        sr = 16000
        # Simulate 1s speech, 1s silence, 1s speech
        audio = np.concatenate([
            np.random.randn(sr),
            np.zeros(sr),
            np.random.randn(sr)
        ])
        patterns = identify_speech_patterns(audio, sr)
        self.assertIn('speaking_rate', patterns)
        self.assertIn('avg_speech_duration', patterns)
        self.assertIn('avg_pause_duration', patterns)
        self.assertIn('num_segments', patterns)
        self.assertGreater(patterns['num_segments'], 0)

    def test_save_text(self):
        text = "test text"
        path = "test_output.txt"
        save_text(text, path)
        self.assertTrue(os.path.exists(path))
        with open(path, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), text)
        os.remove(path)

if __name__ == "__main__":
    unittest.main()
