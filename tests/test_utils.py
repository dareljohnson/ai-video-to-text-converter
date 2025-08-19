"""
Unit tests for AI Video-to-Text Converter utilities.

Authors: Bai Blyden, Darel Johnson
"""


import unittest
import numpy as np
import os
import logging
import tempfile
from pathlib import Path
from modules.utils import (
    extract_audio,
    extract_mfcc,
    correct_punctuation,
    correct_capitalization,
    identify_speech_patterns,
    save_text,
    setup_logging,
    track_timeline,
    format_transcript_with_paragraphs_and_speakers,
    assign_speakers_to_chunks,
    compute_wer
)
from modules.config import AppConfig

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

    def test_realtime_option(self):
        from modules.config import AppConfig
        config = AppConfig(realtime=True)
        self.assertTrue(config.realtime)

    def test_compute_wer(self):
        from modules.utils import compute_wer
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jump over the lazy dog"
        wer = compute_wer(ref, hyp)
        self.assertAlmostEqual(wer, 1/9, places=3)

    def test_setup_logging(self):
        """Test logging setup with various configurations."""
        # Test with file logging enabled
        log_file = Path("test_setup_logging.log")
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            log_to_file=True,
            log_file=str(log_file),
            log_level="DEBUG",
            _skip_validation=True
        )
        logger = setup_logging(config)
        
        # Test that logger was configured correctly
        self.assertEqual(logger.getEffectiveLevel(), logging.DEBUG)
        logger.debug("Test debug message")
        logger.info("Test info message")
        
        # Verify log file was created and contains messages
        self.assertTrue(log_file.exists())
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test debug message", content)
            self.assertIn("Test info message", content)
            
        # Clean up
        # Close all handlers to release file
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        
        # Now we can safely remove the file
        if log_file.exists():
            log_file.unlink()
        
        # Test without file logging
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            log_to_file=False,
            log_level="INFO",
            _skip_validation=True
        )
        logger = setup_logging(config)
        self.assertEqual(logger.getEffectiveLevel(), logging.INFO)

    def test_track_timeline(self):
        """Test timeline tracking functionality."""
        # Test with normal result
        result = {
            "text": "This is a test",
            "chunks": [
                {"text": "This", "timestamp": [0.0, 0.5]},
                {"text": "is", "timestamp": [0.6, 0.8]},
                {"text": "a", "timestamp": [0.9, 1.0]},
                {"text": "test", "timestamp": [1.1, 1.5]}
            ]
        }
        timeline = track_timeline(result)
        self.assertEqual(len(timeline), 4)
        self.assertEqual(timeline[0]["timestamp"], [0.0, 0.5])
        
        # Test with empty result
        empty_result = {"text": "", "chunks": []}
        timeline = track_timeline(empty_result)
        self.assertEqual(len(timeline), 0)

    def test_format_transcript_with_paragraphs(self):
        """Test transcript formatting with paragraphs and speakers."""
        # Test with speaker labels and pauses
        result = {
            "chunks": [
                {"text": "Hello world", "timestamp": [0.0, 1.0], "speaker": "Speaker 1"},
                {"text": "This is test", "timestamp": [2.5, 3.8], "speaker": "Speaker 2"}  # Long pause
            ]
        }
        formatted = format_transcript_with_paragraphs_and_speakers(result, min_pause_s=1.0)
        paragraphs = formatted.split('\n\n')
        self.assertEqual(len(paragraphs), 2)  # Should have 2 paragraphs
        self.assertTrue("Hello world" in paragraphs[0])
        self.assertTrue("This is test" in paragraphs[1])
        
        # Test without speakers
        result = {
            "chunks": [
                {"text": "Hello world", "timestamp": [0.0, 1.0]},
                {"text": "This is test", "timestamp": [2.5, 3.8]}
            ]
        }
        formatted = format_transcript_with_paragraphs_and_speakers(result, min_pause_s=1.0)
        paragraphs = formatted.split('\n\n')
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0], "Hello world")
        self.assertEqual(paragraphs[1], "This is test")
        
        # Test pause-based paragraph breaks
        result = {
            "chunks": [
                {"text": "First sentence", "timestamp": [0.0, 1.0]},
                {"text": "Second sentence", "timestamp": [1.1, 2.0]},  # Short pause
                {"text": "Third sentence", "timestamp": [4.0, 5.0]},   # Long pause
            ]
        }
        formatted = format_transcript_with_paragraphs_and_speakers(result, min_pause_s=1.5)
        paragraphs = formatted.split('\n\n')
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue("First sentence Second sentence" in paragraphs[0].replace('\n', ' '))
        self.assertTrue("Third sentence" in paragraphs[1])
        
        # Test without speaker labels
        result = {
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": "world", "timestamp": [0.6, 1.0]},
                {"text": "test", "timestamp": [2.5, 3.0]}  # Long pause
            ]
        }
        formatted = format_transcript_with_paragraphs_and_speakers(result)
        self.assertTrue("\n\n" in formatted)  # Should have paragraph break

    def test_assign_speakers(self):
        """Test speaker assignment to chunks."""
        chunks = [
            {"text": "Hello", "timestamp": [1.0, 1.5]},
            {"text": "world", "timestamp": [2.0, 2.5]}
        ]
        segments = [
            (0.8, 1.7, "Speaker 1"),  # Overlaps with "Hello"
            (1.9, 2.6, "Speaker 2")   # Overlaps with "world"
        ]
        
        labeled_chunks = assign_speakers_to_chunks(chunks, segments)
        self.assertEqual(labeled_chunks[0]["speaker"], "Speaker 1")
        self.assertEqual(labeled_chunks[1]["speaker"], "Speaker 2")
        
        # Test non-overlapping segment
        chunks = [{"text": "Test", "timestamp": [5.0, 5.5]}]  # No overlap
        labeled_chunks = assign_speakers_to_chunks(chunks, segments)
        self.assertEqual(labeled_chunks[0]["speaker"], "Unknown")

    def test_compute_wer_edge_cases(self):
        """Test Word Error Rate computation with edge cases."""
        # Test empty strings
        self.assertEqual(compute_wer("", ""), 0.0)
        self.assertEqual(compute_wer("test", ""), 1.0)
        self.assertEqual(compute_wer("", "test"), 1.0)
        
        # Test case sensitivity - function is case-sensitive
        self.assertEqual(compute_wer("Test", "test"), 1.0)
        
        # Test multiple spaces
        self.assertEqual(compute_wer("test  string", "test string"), 0.0)
        
        # Test completely different strings
        self.assertEqual(compute_wer("hello world", "goodbye universe"), 1.0)
        
        # Test partial match
        self.assertAlmostEqual(compute_wer("the quick brown fox", "the slow brown fox"), 0.25)

if __name__ == "__main__":
    unittest.main()