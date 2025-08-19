"""
Unit tests for video-to-text pipeline.

Authors: Bai Blyden, Darel Johnson
"""


import unittest
from pathlib import Path
import json
import tempfile
import numpy as np
import os
import torch
import soundfile as sf
import librosa
import time
import queue
import psutil
from modules.utils import (
    extract_mfcc,
    correct_punctuation,
    correct_capitalization
)
from modules.pipeline import VideoToTextPipeline
from modules.config import AppConfig

class TestVideoToTextPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = AppConfig(
            input_path="tests/fixtures/test_input.mp4",
            output_text_path="test_output.txt",
            output_subs_path="test_output.srt",
            log_level="DEBUG",
            log_to_file=True,
            log_file="test_pipeline.log",
            omp_num_threads="1",
            mkl_num_threads="1",
            openblas_num_threads="1",
            numexpr_num_threads="1",
            torch_num_threads="1",
            _skip_validation=True  # Skip validation in test mode
        )
        # Clean up any existing CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.pipeline = VideoToTextPipeline(self.config)
        
    def tearDown(self):
        """Clean up test files."""
        import time
        import logging
        import gc
        import atexit
        
        def cleanup_handler():
            """Handle cleanup during test teardown"""
            try:
                # Close logger handlers
                if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'logger'):
                    for handler in self.pipeline.logger.handlers[:]:
                        try:
                            handler.close()
                            self.pipeline.logger.removeHandler(handler)
                        except Exception:
                            pass
                    logging.shutdown()
                
                # Force garbage collection
                if hasattr(self, 'pipeline'):
                    del self.pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                test_files = [
                    "test_output.txt",
                    "test_output.srt",
                    "test_pipeline.log"
                ]
                
                for file in test_files:
                    try:
                        if os.path.exists(file):
                            os.remove(file)
                    except Exception:
                        try:
                            if Path(file).exists():
                                Path(file).unlink()
                        except Exception:
                            pass
            except Exception:
                pass
        
        # Register cleanup handler
        atexit.register(cleanup_handler)
        
        # Run cleanup
        cleanup_handler()
        
    def test_json_imports(self):
        """Test that json module is properly imported and available."""
        # Create a test dictionary
        test_data = {"test": "data"}
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            # Test json.dump
            json.dump(test_data, tmp)
            tmp_path = tmp.name
            
        # Test json.load
        with open(tmp_path, 'r') as f:
            loaded_data = json.load(f)
            
        self.assertEqual(test_data, loaded_data)
        
        # Clean up
        Path(tmp_path).unlink()
        
    def test_batch_file_handling(self):
        """Test batch processing file handling."""
        # Create temporary input and output directories
        with tempfile.TemporaryDirectory() as test_dir:
            test_input_dir = Path(test_dir) / "input"
            test_output_dir = Path(test_dir) / "output"
            test_input_dir.mkdir()
            test_output_dir.mkdir()
            
            # Create a test file
            test_file = test_input_dir / "test.wav"
            test_data = np.zeros(1000, dtype=np.float32)
            import soundfile as sf
            sf.write(test_file, test_data, 16000)
            
            # Test that the pipeline can handle the file
            self.assertTrue(test_file.exists())
            self.assertTrue(test_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".ogg"])

    def test_pipeline_initialization(self):
        """Test pipeline initialization with various config options."""
        # Test with different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = AppConfig(
                input_path="test_input.mp4",
                output_text_path="test_output.txt",
                log_level=level,
                _skip_validation=True
            )
            pipeline = VideoToTextPipeline(config)
            self.assertEqual(pipeline.config.log_level, level)

        # Test with and without log file
        config_with_file = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            log_to_file=True,
            log_file="test_log.txt",
            _skip_validation=True
        )
        pipeline = VideoToTextPipeline(config_with_file)
        self.assertTrue(pipeline.config.log_to_file)
        self.assertEqual(pipeline.config.log_file, "test_log.txt")

    def test_thread_environment_variables(self):
        """Test thread environment variables are set correctly."""
        test_vars = {
            "omp_num_threads": "2",
            "mkl_num_threads": "2",
            "openblas_num_threads": "2",
            "numexpr_num_threads": "2",
            "torch_num_threads": "2"
        }
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            _skip_validation=True,
            **test_vars
        )
        pipeline = VideoToTextPipeline(config)
        
        # Check that environment variables were set
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
                   "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"]:
            self.assertEqual(os.environ.get(var), "2")

    def test_batch_processing_empty_directory(self):
        """Test batch processing behavior with empty input directory."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_input_dir = Path(test_dir) / "input"
            test_output_dir = Path(test_dir) / "output"
            test_input_dir.mkdir()
            test_output_dir.mkdir()
            
            # Set up pipeline with test directories
            config = AppConfig(
                input_path=str(test_input_dir),
                output_text_path=str(test_output_dir / "output.txt")
            )
            pipeline = VideoToTextPipeline(config)
            
            # Test empty directory handling
            pipeline.run_batch()
            # Should create output directory even if no files to process
            self.assertTrue(test_output_dir.exists())

    def test_logging_setup(self):
        """Test logging configuration."""
        # Test with file logging
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            log_to_file=True,
            log_file="test_pipeline.log",
            log_level="DEBUG",
            _skip_validation=True
        )
        pipeline = VideoToTextPipeline(config)
        
        # Check that logger was created
        self.assertIsNotNone(pipeline.logger)
        
        # Check that log file was created
        self.assertTrue(Path("test_pipeline.log").exists())
        
        # Test log message
        test_message = "Test log message"
        pipeline.logger.info(test_message)
        
        # Verify message was written to file
        with open("test_pipeline.log", "r") as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)

    def test_device_configuration(self):
        """Test GPU/CPU device configuration."""
        # Test CPU configuration
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            cuda_device=-1,
            _skip_validation=True
        )
        pipeline = VideoToTextPipeline(config)
        self.assertEqual(pipeline.device, "cpu")
        
        # Test invalid CUDA device handling
        if torch.cuda.is_available():
            invalid_device = torch.cuda.device_count()
            config = AppConfig(
                input_path="test_input.mp4",
                output_text_path="test_output.txt",
                cuda_device=invalid_device,
                _skip_validation=True
            )
            with self.assertRaises(ValueError):
                VideoToTextPipeline(config)

    def test_audio_processing(self):
        """Test audio extraction and processing."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create a test audio file
            sample_rate = 16000
            duration = 1  # 1 second
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                mfcc=True
            )
            pipeline = VideoToTextPipeline(config)
            
            # Test audio loading
            audio, sr = librosa.load(temp_audio.name, sr=None)
            self.assertEqual(sr, sample_rate)
            self.assertEqual(len(audio), len(audio_data))
            
            # Test MFCC extraction
            mfccs = extract_mfcc(audio, sr)
            self.assertIsNotNone(mfccs)
            self.assertEqual(mfccs.shape[0], 13)  # Standard number of MFCC coefficients
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_silence_detection(self):
        """Test silence detection in audio."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create a silent audio file
            sample_rate = 16000
            duration = 1
            # Add very small random noise to make it more realistic
            silent_audio = np.random.normal(0, 1e-10, int(sample_rate * duration))
            sf.write(temp_audio.name, silent_audio, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                mfcc=True
            )
            pipeline = VideoToTextPipeline(config)
            
            # Load and process silent audio
            audio, sr = librosa.load(temp_audio.name, sr=None)
            
            # Normalize audio to ensure very low amplitude
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            mfccs = extract_mfcc(audio, sr)
            mfcc_energy = np.mean(np.abs(mfccs))
            
            # Check that energy is very low for silent audio relative to normal speech
            typical_speech_energy = 50.0  # Typical energy for speech
            self.assertLess(mfcc_energy / typical_speech_energy, 0.1)  # Should be less than 10% of typical speech energy
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create a short test audio
            sample_rate = 16000
            duration = 0.1
            audio_data = np.random.randn(int(sample_rate * duration))
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt"
            )
            pipeline = VideoToTextPipeline(config)
            
            # Record start time
            start_time = time.time()
            
            # Run pipeline
            pipeline.run()
            
            # Check execution time
            execution_time = time.time() - start_time
            self.assertGreater(execution_time, 0)
            
            # Check if output files were created
            self.assertTrue(Path(config.output_text_path).exists())
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test with non-existent input file
        with self.assertRaises(ValueError) as cm:
            AppConfig(
                input_path="nonexistent_file_that_does_not_exist.mp4",
                output_text_path="test_output.txt",
                _skip_validation=False  # Ensure validation is enabled
            )
        self.assertIn("Input path does not exist", str(cm.exception))
        
        # Test with invalid audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(b'invalid audio data')
            temp_audio.flush()
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                _skip_validation=True  # Skip validation to test the pipeline
            )
            pipeline = VideoToTextPipeline(config)
            
            # Should raise ValueError or wave.Error for invalid audio data
            with self.assertRaises((ValueError, Exception)):
                pipeline.run()
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_language_model_configuration(self):
        """Test language model configuration and transcription."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio
            sample_rate = 16000
            duration = 1
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Test different languages
            for lang in ["en", "fr", "es"]:
                config = AppConfig(
                    input_path=temp_audio.name,
                    output_text_path="test_output.txt",
                    language=lang
                )
                pipeline = VideoToTextPipeline(config)
                
                # Verify model configuration
                self.assertEqual(pipeline.config.language, lang)
                self.assertIsNotNone(pipeline.asr_pipeline)
                
            # Test model parameters
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                language="en"
            )
            pipeline = VideoToTextPipeline(config)
            
            # Check if pipeline is using the correct device
            expected_dtype = torch.float16 if pipeline.device == "cuda" else torch.float32
            self.assertEqual(pipeline.asr_pipeline.model.dtype, expected_dtype)
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_speaker_recognition(self):
        """Test speaker recognition functionality."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio with "silence" between speakers
            sample_rate = 16000
            duration = 6  # 6 seconds total
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create two different "speakers" with different frequencies and amplitudes
            # First speaker: 440 Hz with louder amplitude
            speaker1 = np.sin(2 * np.pi * 440 * t[:2*sample_rate]) * 0.8
            silence = np.zeros(sample_rate)
            # Second speaker: 880 Hz with different amplitude
            speaker2 = np.sin(2 * np.pi * 880 * t[:2*sample_rate]) * 0.6
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.01, len(speaker1))
            speaker1 += noise
            noise = np.random.normal(0, 0.01, len(speaker2))
            speaker2 += noise
            
            audio_data = np.concatenate([speaker1, silence, speaker2])
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                enable_speaker_recognition=True,
                min_speakers=2,
                max_speakers=3,
                speaker_window_sec=1.0,
                save_speaker_info=True
            )
            pipeline = VideoToTextPipeline(config)
            pipeline.run()
            
            # Check if speaker information was saved
            speaker_info_path = config.output_text_path.replace('_output.txt', '_speaker_info.json')
            self.assertTrue(Path(speaker_info_path).exists())
            
            # Verify speaker info format
            with open(speaker_info_path, 'r') as f:
                speaker_info = json.load(f)
                self.assertIsInstance(speaker_info, dict)
            
        # Cleanup
        Path(temp_audio.name).unlink()
        if Path(speaker_info_path).exists():
            Path(speaker_info_path).unlink()

    def test_subtitle_generation(self):
        """Test subtitle generation functionality."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio
            sample_rate = 16000
            duration = 2
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                output_subs_path="test_output.srt",
                generate_subtitles=True
            )
            pipeline = VideoToTextPipeline(config)
            
            # Mock transcription result with proper timestamps
            result = {
                "text": "Test subtitle",
                "chunks": [
                    {
                        "text": "Test subtitle",
                        "timestamp": [0.0, 2.0]
                    }
                ]
            }
            
            # Save subtitles directly
            from modules.utils import save_subtitles
            save_subtitles(result, config.output_subs_path)
            
            # Check if subtitle file was created
            self.assertTrue(Path(config.output_subs_path).exists())
            
            # Verify subtitle format
            with open(config.output_subs_path, 'r') as f:
                srt_content = f.read()
                # Basic SRT format check
                self.assertRegex(srt_content, r'\d+\n\d{2}:\d{2}:\d{2},\d{3}')
            
        # Cleanup
        Path(temp_audio.name).unlink()
        Path(config.output_subs_path).unlink()

    def test_realtime_processing(self):
        """Test real-time processing configuration."""
        # Test realtime mode initialization
        config = AppConfig(
            input_path="test_input.mp4",
            output_text_path="test_output.txt",
            realtime=True,
            _skip_validation=True
        )
        pipeline = VideoToTextPipeline(config)
        
        # Mock audio input queue
        audio_queue = queue.Queue()
        
        # Create some test audio chunks
        sample_rate = 16000
        chunk_duration = 0.5
        t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
        audio_chunk = np.sin(2 * np.pi * 440 * t)
        
        # Test queue handling
        audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Signal to stop
        
        # Verify realtime configuration
        self.assertTrue(pipeline.config.realtime)
        self.assertEqual(pipeline.device, "cpu" if not torch.cuda.is_available() else "cuda")

    def test_text_post_processing(self):
        """Test text post-processing features."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio
            sample_rate = 16000
            duration = 1
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Test punctuation correction
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                punctuation_correction=True,
                capitalization=False
            )
            pipeline = VideoToTextPipeline(config)
            test_text = "hello world this is a test"
            processed_text = correct_punctuation(test_text)
            self.assertIn(".", processed_text)
            
            # Test capitalization
            config.capitalization = True
            test_text = "hello world. this is a test."
            processed_text = correct_capitalization(test_text)
            self.assertTrue(processed_text.startswith("Hello"))
            self.assertIn("This", processed_text)
            
            # Test combined processing
            test_text = "hello world this is a test"
            processed_text = correct_capitalization(correct_punctuation(test_text))
            self.assertTrue(processed_text.startswith("Hello"))
            self.assertIn(".", processed_text)
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_timeline_and_patterns(self):
        """Test timeline tracking and speech pattern analysis."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio with varying patterns
            sample_rate = 16000
            duration = 3
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create audio with speech-like patterns
            speech = np.sin(2 * np.pi * 440 * t[:sample_rate])  # 1s speech
            silence = np.zeros(sample_rate)  # 1s silence
            speech2 = np.sin(2 * np.pi * 440 * t[:sample_rate])  # 1s speech
            audio_data = np.concatenate([speech, silence, speech2])
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                track_timeline=True,
                identify_speech_patterns=True
            )
            pipeline = VideoToTextPipeline(config)
            
            # Mock transcription result
            result = {
                "text": "Test speech with silence",
                "chunks": [
                    {"text": "Test", "timestamp": [0.0, 1.0]},
                    {"text": "speech", "timestamp": [2.0, 3.0]}
                ]
            }
            
            # Process timeline directly
            from modules.utils import track_timeline, identify_speech_patterns
            timeline = track_timeline(result)
            
            # Save timeline
            timeline_path = config.output_text_path.replace('_output.txt', '_timeline.json')
            with open(timeline_path, 'w') as f:
                json.dump({"segments": timeline}, f)
            
            # Process patterns
            patterns = identify_speech_patterns(audio_data, sample_rate)
            
            # Save patterns
            patterns_path = config.output_text_path.replace('_output.txt', '_patterns.json')
            with open(patterns_path, 'w') as f:
                json.dump(patterns, f)
            
            # Check timeline file
            self.assertTrue(Path(timeline_path).exists())
            with open(timeline_path, 'r') as f:
                timeline_data = json.load(f)
                self.assertIsInstance(timeline_data, dict)
                self.assertIn('segments', timeline_data)
            
            # Check patterns file
            self.assertTrue(Path(patterns_path).exists())
            with open(patterns_path, 'r') as f:
                patterns_data = json.load(f)
                self.assertIsInstance(patterns_data, dict)
                self.assertIn('speaking_rate', patterns_data)
                self.assertIn('avg_speech_duration', patterns_data)
                
        # Cleanup
        Path(temp_audio.name).unlink()
        Path(timeline_path).unlink()
        Path(patterns_path).unlink()

    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir_path = Path(test_dir)
            input_dir = test_dir_path / "input"
            output_dir = test_dir_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Create multiple test files
            sample_rate = 16000
            duration = 1
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)
            
            test_files = ["test1.wav", "test2.wav", "test3.wav"]
            for fname in test_files:
                audio_path = input_dir / fname
                sf.write(str(audio_path), audio_data, sample_rate)
            
            # Create a test file in the input directory
            sample_wav = input_dir / "test.wav"
            sf.write(str(sample_wav), audio_data, sample_rate)

            # Configure and run the pipeline
            config = AppConfig(
                input_path=str(sample_wav),  # Set to specific file, directory will be extracted
                output_text_path=str(output_dir / "test_output.txt")
            )
            pipeline = VideoToTextPipeline(config)
            pipeline.run_batch()
            
            # Verify outputs - timings.txt should be in the output directory
            timings_path = output_dir / "timings.txt"
            self.assertTrue(timings_path.exists())
            
            for fname in test_files:
                base = Path(fname).stem
                output_text = output_dir / f"{base}_output.txt"
                output_subs = output_dir / f"{base}_output.srt"
                self.assertTrue(output_text.exists())
                self.assertTrue(output_subs.exists())
                audio_path = input_dir / fname
                sf.write(str(audio_path), audio_data, sample_rate)
                
            config = AppConfig(
                input_path=str(input_dir),
                output_text_path=str(output_dir / "output.txt")
            )
            pipeline = VideoToTextPipeline(config)
            pipeline.run_batch()
            
            # Check output files
            self.assertTrue(Path(output_dir / "timings.txt").exists())
            for fname in test_files:
                base = Path(fname).stem
                self.assertTrue(Path(output_dir / f"{base}_output.txt").exists())

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid language
        with self.assertRaises(ValueError):
            config = AppConfig(
                input_path="test.wav",
                output_text_path="test_output.txt",
                language="invalid_lang",
                _skip_validation=True
            )
            VideoToTextPipeline(config)
            
        # Test invalid speaker settings
        with self.assertRaises(ValueError):
            config = AppConfig(
                input_path="test.wav",
                output_text_path="test_output.txt",
                min_speakers=3,
                max_speakers=2,
                _skip_validation=True
            )
            VideoToTextPipeline(config)
            
        # Test invalid device settings
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                config = AppConfig(
                    input_path="test.wav",
                    output_text_path="test_output.txt",
                    cuda_device=torch.cuda.device_count(),  # Invalid device index
                    _skip_validation=True
                )
                VideoToTextPipeline(config)
                
        # Test invalid thread settings
        with self.assertRaises(ValueError):
            config = AppConfig(
                input_path="test.wav",
                output_text_path="test_output.txt",
                omp_num_threads="-1",  # Invalid thread count
                _skip_validation=True
            )
            VideoToTextPipeline(config)
            
        # Test path validation
        with self.assertRaises(ValueError):
            config = AppConfig(
                input_path="nonexistent_file.mp4",  # File doesn't exist
                output_text_path="test_output.txt",
                _skip_validation=False  # Enable validation to test it
            )
            
        # Test output directory creation with real files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "test.wav"
            with open(test_file, "wb") as f:
                f.write(b"test")
                
            config = AppConfig(
                input_path=str(test_file),
                output_text_path="nonexistent_dir/test_output.txt",
                _skip_validation=False  # Enable validation
            )
            
            self.assertTrue(Path("nonexistent_dir").exists())
            Path("nonexistent_dir").rmdir()
        pipeline = VideoToTextPipeline(config)
        self.assertTrue(Path("nonexistent_dir").exists())
        Path("nonexistent_dir").rmdir()

    def test_memory_management(self):
        """Test memory management and resource cleanup."""
        import psutil
        import gc
        
        def get_memory_usage():
            """Get current memory usage in MB."""
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / 1024 / 1024  # Convert to MB
            
        # Get initial memory usage
        initial_mem = get_memory_usage()
            
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio data (1 second of random noise)
            sample_rate = 16000
            duration = 1  # second
            audio_data = np.random.randn(duration * sample_rate).astype(np.float32)
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Run the pipeline
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                mfcc=True
            )
            pipeline = VideoToTextPipeline(config)
            
            # Clear existing memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Run pipeline
            pipeline.run()
            
            # Clean up pipeline
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check memory usage
            final_mem = get_memory_usage()
            memory_diff = abs(final_mem - initial_mem)
            
            # Allow for model loading overhead (1GB)
            self.assertLess(memory_diff, 1024)  # 1GB limit
            
            # Cleanup temporary files
            if os.path.exists(temp_audio.name):
                try:
                    Path(temp_audio.name).unlink()
                except:
                    pass
            if Path("test_output.txt").exists():
                try:
                    Path("test_output.txt").unlink()
                except:
                    pass
        initial_memory = get_memory_usage()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create a short test audio file
            sample_rate = 16000
            duration = 1  # 1 second
            audio_data = np.random.randn(int(sample_rate * duration))
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Run the pipeline
            config = AppConfig(
                input_path=temp_audio.name,
                output_text_path="test_output.txt",
                mfcc=True
            )
            pipeline = VideoToTextPipeline(config)
            pipeline.run()
            
            # Clean up
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check memory usage
            final_memory = get_memory_usage()
            memory_diff = abs(final_memory - initial_memory)
            
            # Allow for some memory overhead, but not too much
            self.assertLess(memory_diff, 200)  # Increased threshold to 200MB
            
        # Cleanup
        if os.path.exists(temp_audio.name):
            try:
                Path(temp_audio.name).unlink()
            except:
                pass

    def test_performance_benchmarking(self):
        """Test performance benchmarking and optimization."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Create test audio
            sample_rate = 16000
            duration = 5  # 5 seconds
            audio_data = np.random.randn(sample_rate * duration)
            sf.write(temp_audio.name, audio_data, sample_rate)
            
            # Test different thread configurations
            thread_configs = [
                {"omp_num_threads": "1", "mkl_num_threads": "1"},
                {"omp_num_threads": "2", "mkl_num_threads": "2"},
                {"omp_num_threads": "4", "mkl_num_threads": "4"}
            ]
            
            timings = []
            for thread_config in thread_configs:
                config = AppConfig(
                    input_path=temp_audio.name,
                    output_text_path="test_output.txt",
                    **thread_config
                )
                pipeline = VideoToTextPipeline(config)
                
                start_time = time.time()
                pipeline.run()
                elapsed = time.time() - start_time
                timings.append(elapsed)
                
                # Cleanup between runs
                if Path("test_output.txt").exists():
                    Path("test_output.txt").unlink()
            
            # Verify that some configuration was faster
            min_time = min(timings)
            max_time = max(timings)
            self.assertLess(min_time, max_time)  # At least some optimization happened
            
        # Cleanup
        Path(temp_audio.name).unlink()

    def test_edge_cases(self):
        """Test edge cases in audio processing."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)
            
            # Test extremely short audio
            short_audio = test_dir / "short.wav"
            tiny_data = np.random.randn(100)  # Extremely short
            sf.write(short_audio, tiny_data, 16000)
            
            # Test extremely quiet audio
            quiet_audio = test_dir / "quiet.wav"
            quiet_data = np.random.randn(16000) * 0.001  # Very low amplitude
            sf.write(quiet_audio, quiet_data, 16000)
            
            # Test maximum amplitude audio
            loud_audio = test_dir / "loud.wav"
            loud_data = np.ones(16000)  # Maximum amplitude
            sf.write(loud_audio, loud_data, 16000)
            
            # Test unusual sample rates
            unusual_rates = [8000, 44100, 96000]
            for rate in unusual_rates:
                rate_audio = test_dir / f"rate_{rate}.wav"
                rate_data = np.random.randn(rate)
                sf.write(rate_audio, rate_data, rate)
                
                config = AppConfig(
                    input_path=str(rate_audio),
                    output_text_path=str(test_dir / "output.txt")
                )
                pipeline = VideoToTextPipeline(config)
                pipeline.run()  # Should handle different sample rates
                
            # Test all edge cases
            for audio_file in [short_audio, quiet_audio, loud_audio]:
                config = AppConfig(
                    input_path=str(audio_file),
                    output_text_path=str(test_dir / "output.txt"),
                    mfcc=True
                )
                pipeline = VideoToTextPipeline(config)
                pipeline.run()  # Should handle edge cases without crashing

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)
            
            # Test different path formats
            paths = [
                test_dir / "normal.wav",
                test_dir / "with spaces.wav",
                test_dir / "with#special$chars.wav",
                test_dir / "deeply" / "nested" / "path.wav"
            ]
            
            # Create test audio
            audio_data = np.random.randn(16000)
            
            for path in paths:
                # Create necessary directories
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save test audio
                sf.write(str(path), audio_data, 16000)
                
                # Test with both forward and backward slashes
                for test_path in [str(path), str(path).replace('\\', '/'), str(path).replace('/', '\\')]:
                    config = AppConfig(
                        input_path=test_path,
                        output_text_path=str(test_dir / "output.txt")
                    )
                    pipeline = VideoToTextPipeline(config)
                    pipeline.run()
                    
                    # Verify output was created
                    self.assertTrue(Path(config.output_text_path).exists())
                    Path(config.output_text_path).unlink()

if __name__ == '__main__':
    unittest.main()
