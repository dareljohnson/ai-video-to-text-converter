"""
Pipeline for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""

import os
import time
import shutil
import torch
import sounddevice as sd
import queue
import numpy as np
import tempfile
import soundfile as sf
import logging
import threading
import _thread
from typing import Optional, Dict, List, Any
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from modules.utils import setup_logging
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from moviepy.editor import VideoFileClip, AudioFileClip
from modules.speaker_recognition import SpeakerRecognizer
import librosa
from modules.config import AppConfig
from modules.json_utils import json_dump
from modules.utils import (
    extract_audio,
    extract_mfcc,
    correct_punctuation,
    correct_capitalization,
    track_timeline,
    identify_speech_patterns,
    save_subtitles,
    save_text
)
from modules.utils import format_transcript_with_paragraphs_and_speakers
from transformers.utils import logging


class VideoToTextPipeline:
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'asr_pipeline'):
            del self.asr_pipeline
        torch.cuda.empty_cache()

    def run_batch(self):
        """
        Transcribe all audio/video files in the input directory.
        """
        batch_start = time.time()

        # Check ffmpeg availability
        if not shutil.which("ffmpeg"):
            print("ERROR: ffmpeg is not found in your PATH. Please install ffmpeg and add it to your system PATH, then restart your terminal.")
            return
            
        # Get input and output paths from config
        input_dir = os.path.abspath(self.config.input_path if self.config.input_path else 'input')
        if not os.path.isdir(input_dir):
            # If input_path is a file, use its directory
            input_dir = os.path.dirname(input_dir)
        output_dir = os.path.dirname(os.path.abspath(self.config.output_text_path))
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Print directory info for debugging
        print(f"Looking for media files in: {input_dir}")
        print(f"Output files will be saved to: {output_dir}")

        # Supported extensions
        exts = (".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".ogg")
        try:
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        except FileNotFoundError:
            self.logger.error(f"Input directory not found: {input_dir}")
            print(f"Error: Input directory not found: {input_dir}")
            print(f"Created input directory. Please add media files to {input_dir} and try again.")
            return
        if not files:
            print(f"No audio/video files found in {input_dir}")
            return

        print(f"Found {len(files)} files in {input_dir}: {files}")
        timings_path = os.path.join(output_dir, 'timings.txt')

        # Clear timings file at start of batch
        with open(timings_path, 'w', encoding='utf-8') as tf:
            tf.write('')

        for fname in files:
            print(f"\n[Batch] Processing {fname} ...")
            file_start = time.time()

            # Set input/output paths for this file
            self.config.input_path = os.path.join(input_dir, fname)
            base = os.path.splitext(fname)[0]
            self.config.output_text_path = os.path.join(output_dir, f"{base}_output.txt")
            self.config.output_subs_path = os.path.join(output_dir, f"{base}_output.srt")

            try:
                self.run()
            except Exception as e:
                print(f"[Batch] Error processing {fname}: {e}")

            file_elapsed = time.time() - file_start
            print(f"[Batch] Processing time for {fname}: {file_elapsed:.2f} seconds.")

            # Save per-file timing
            with open(timings_path, 'a', encoding='utf-8') as tf:
                tf.write(f"{fname}: {file_elapsed:.2f} seconds\n")

        batch_elapsed = time.time() - batch_start
        print(f"Total batch processing time: {batch_elapsed:.2f} seconds.")
        with open(timings_path, 'a', encoding='utf-8') as tf:
            tf.write(f"Total batch processing time: {batch_elapsed:.2f} seconds.\n")

    def _validate_config(self):
        """Validate pipeline configuration."""
        # Validate language
        if self.config.language is not None and self.config.language not in ["en", "fr", "es", "de", "it", "pt", "nl", "pl", "ja", "zh", "ko"]:
            raise ValueError(f"Invalid language: {self.config.language}")
            
        # Validate speaker settings
        if self.config.min_speakers > self.config.max_speakers:
            raise ValueError(f"min_speakers ({self.config.min_speakers}) cannot be greater than max_speakers ({self.config.max_speakers})")
            
        # Validate CUDA device if using GPU
        if torch.cuda.is_available() and self.config.cuda_device >= torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA device index {self.config.cuda_device}. Only {torch.cuda.device_count()} devices available")
            
        # Validate thread counts
        for thread_var in ["omp_num_threads", "mkl_num_threads", "openblas_num_threads", "numexpr_num_threads", "torch_num_threads"]:
            value = getattr(self.config, thread_var)
            if value < 1:
                raise ValueError(f"Invalid {thread_var}: {value}. Must be >= 1")
            
        # Validate paths
        if not self.config.input_path:
            raise ValueError("Input path must be specified")
        if not self.config.output_text_path:
            raise ValueError("Output text path must be specified")
            
        # Create output directory if needed
        outdir = os.path.dirname(self.config.output_text_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

    def __init__(self, config: AppConfig):
        self.config = config
        self.asr_pipeline = None
        self.device = None
        
        # Convert relative paths to absolute paths
        if self.config.input_path:
            self.config.input_path = os.path.abspath(self.config.input_path)
        if self.config.output_text_path:
            self.config.output_text_path = os.path.abspath(self.config.output_text_path)
            os.makedirs(os.path.dirname(self.config.output_text_path), exist_ok=True)
        if self.config.output_subs_path:
            self.config.output_subs_path = os.path.abspath(self.config.output_subs_path)
            os.makedirs(os.path.dirname(self.config.output_subs_path), exist_ok=True)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize logging using utils.setup_logging
        self.logger = setup_logging(config)
        self.logger.info("Initializing Video-to-Text Pipeline")
        
        # Set thread environment variables from config
        thread_vars = {
            "OMP_NUM_THREADS": config.omp_num_threads,
            "MKL_NUM_THREADS": config.mkl_num_threads,
            "OPENBLAS_NUM_THREADS": config.openblas_num_threads,
            "NUMEXPR_NUM_THREADS": config.numexpr_num_threads,
            "TORCH_NUM_THREADS": config.torch_num_threads
        }
        for var, value in thread_vars.items():
            os.environ[var] = str(value)
            self.logger.debug(f"Set {var}={value}")
            
        # Configure device
        self.logger.info("Configuring computation device")
        try:
            if torch.cuda.is_available() and getattr(config, 'cuda_device', -1) >= 0:
                self.device = "cuda"
                cuda_idx = config.cuda_device
                n_gpus = torch.cuda.device_count()
                if cuda_idx >= n_gpus:
                    self.logger.error(f"Requested CUDA device {cuda_idx}, but only {n_gpus} GPUs available")
                    raise ValueError(f"Requested CUDA device {cuda_idx}, but only {n_gpus} GPUs are available.")
                device_name = torch.cuda.get_device_name(cuda_idx)
                self.logger.info(f"Using CUDA device {cuda_idx}: {device_name}")
                # Set memory allocation
                torch.cuda.set_per_process_memory_fraction(0.7)  # Use up to 70% of GPU memory
                torch.cuda.empty_cache()
            else:
                self.device = "cpu"
                cuda_idx = -1
                self.logger.info("Using CPU for computation")
        except Exception as e:
            self.logger.error(f"Error configuring device: {e}")
            self.device = "cpu"
            cuda_idx = -1
            self.logger.info("Falling back to CPU computation")

        # Pipeline for ASR    
        try:
            # Use cached model if available
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
            os.makedirs(model_path, exist_ok=True)
            
            logging.set_verbosity_warning()
            
            # Configure longer timeouts and retries for model downloads
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_DOWNLOAD_RETRY_MULTIPLIER = 2  # Double retry backoff
            huggingface_hub.constants.HF_HUB_DOWNLOAD_RETRY_MAX_RETRIES = 10  # More retries
            
            # Load model with caching enabled and proper retry handling
            max_retries = 5
            retry_delay = 10  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Try to load model from cache first
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        config.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        attn_implementation="eager",
                        cache_dir=model_path,
                        local_files_only=True  # Try local cache first
                    )
                    processor = AutoProcessor.from_pretrained(
                        config.model_name, 
                        cache_dir=model_path,
                        local_files_only=True  # Try local cache first
                    )
                    break
                except Exception as cache_error:
                    self.logger.warning(f"Could not load from cache, attempting download (attempt {attempt + 1}/{max_retries})")
                    try:
                        # If cache load fails, try downloading
                        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            config.model_name,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            attn_implementation="eager",
                            cache_dir=model_path,
                            force_download=attempt == max_retries - 1  # Force download on last attempt
                        )
                        processor = AutoProcessor.from_pretrained(
                            config.model_name, 
                            cache_dir=model_path,
                            force_download=attempt == max_retries - 1  # Force download on last attempt
                        )
                        break
                    except Exception as download_error:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Download failed, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise download_error
            
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=cuda_idx
            )
            # Clear CUDA cache after model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ASR pipeline: {e}")
            raise

    def run(self, timeout=600):  # 10 minute default timeout
        """Run the video-to-text conversion pipeline."""
        # Windows-compatible timeout implementation
        def timeout_handler():
            _thread.interrupt_main()
            
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        
        try:
            start_time = time.time()
            self.logger.info("Starting video-to-text conversion pipeline")
            
            # Put main execution in a try block to handle timer interruption and cleanup
            try:
                if self.config.realtime:
                    self.logger.info("Running in realtime mode")
                    self.run_realtime()
                    return
                    
                # Check ffmpeg availability
                self.logger.debug("Checking ffmpeg availability")
                if not shutil.which("ffmpeg"):
                    self.logger.error("ffmpeg not found in PATH. Please install ffmpeg and add it to system PATH.")
                    print("ERROR: ffmpeg is not found in your PATH. Please install ffmpeg and add it to your system PATH, then restart your terminal.")
                    return

                # Project root is one level up from this file's directory
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                
                # Input path: can be absolute or relative to project root
                input_path = os.path.abspath(os.path.join(project_root, self.config.input_path))
                if not os.path.isfile(input_path):
                    raise FileNotFoundError(f"File not found: {input_path}")
                    
                # Output paths: always relative to project root
                self.config.output_text_path = os.path.join(project_root, self.config.output_text_path)
                self.config.output_subs_path = os.path.join(project_root, self.config.output_subs_path)
                self.logger.info(f"Processing input file: {input_path}")
                
                # Extract audio
                self.logger.debug("Extracting audio from input file")
                audio_path = extract_audio(input_path)
                self.logger.info(f"Audio extracted to: {audio_path}")
                
                # Load audio
                self.logger.debug("Loading audio file")
                try:
                    # Use soundfile for more efficient memory usage
                    audio, sr = sf.read(audio_path)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)  # Convert to mono if stereo
                    self.logger.info(f"Loaded audio: {len(audio)/sr:.2f} seconds, {sr}Hz sample rate")
                except Exception as e:
                    self.logger.error(f"Error loading audio: {e}")
                    # Fallback to librosa if soundfile fails
                    audio, sr = librosa.load(audio_path, sr=None)
                    self.logger.info(f"Loaded audio with librosa: {len(audio)/sr:.2f} seconds, {sr}Hz sample rate")
                
                # Process MFCCs and check for silence
                skip_transcription = False
                mfccs = None
                if self.config.mfcc:
                    self.logger.debug("Extracting MFCCs")
                    mfccs = extract_mfcc(audio, sr)
                    self.logger.debug(f"MFCC shape: {mfccs.shape}")
                    
                    # Silence detection
                    mfcc_energy = np.mean(np.abs(mfccs))
                    silence_threshold = 1e-2
                    self.logger.info(f"MFCC energy: {mfcc_energy:.5f} (threshold: {silence_threshold})")
                    
                    if mfcc_energy < silence_threshold:
                        self.logger.warning(f"Low energy detected (mean abs MFCC: {mfcc_energy:.5f}). Skipping transcription.")
                        skip_transcription = True

                # Transcription
                if skip_transcription:
                    self.logger.info("Skipping transcription due to silence detection")
                    text = ""
                    result = {"text": "", "chunks": []}
                else:
                    self.logger.info("Starting speech-to-text transcription")
                    result = self.asr_pipeline(
                        audio_path, 
                        return_timestamps="word", 
                        generate_kwargs={"task": "transcribe", "language": self.config.language}
                    )
                    text = result["text"]
                    self.logger.info(f"Transcription completed: {len(text.split())} words")
                    self.logger.debug(f"First 100 chars: {text[:100]}...")
                    
                # Post-processing
                if self.config.punctuation_correction:
                    self.logger.debug("Applying punctuation correction")
                    text = correct_punctuation(text)
                    
                if self.config.capitalization:
                    self.logger.debug("Applying capitalization correction")
                    text = correct_capitalization(text)
                    
                # Timeline tracking
                if self.config.track_timeline:
                    self.logger.info("Processing speech timeline")
                    timeline = track_timeline(result)
                    timeline_path = self.config.output_text_path.replace('_output.txt', '_timeline.json')
                    with open(timeline_path, 'w', encoding='utf-8') as f:
                        json_dump(timeline, f, indent=2)
                    self.logger.info(f"Timeline saved to: {timeline_path}")
                else:
                    timeline = None

                if self.config.identify_speech_patterns:
                    patterns = identify_speech_patterns(audio, sr)
                    patterns_path = self.config.output_text_path.replace('_output.txt', '_patterns.json')
                    with open(patterns_path, 'w', encoding='utf-8') as f:
                        json_dump(patterns, f, indent=2)
                else:
                    patterns = None
                    
                # Speaker Recognition
                if self.config.enable_speaker_recognition:
                    self.logger.info("Starting speaker recognition")
                    
                    speaker_results = SpeakerRecognizer.process_audio(
                        config=self.config,
                        audio_path=audio_path,
                        min_speakers=self.config.min_speakers,
                        max_speakers=self.config.max_speakers,
                        window_sec=self.config.speaker_window_sec
                    )
                    
                    if speaker_results and 'speaker_labels' in speaker_results:
                        self.logger.info(f"Speaker recognition completed. Found {len(set(speaker_results['speaker_labels']))} speakers")
                        
                        # Assign speakers to chunks
                        self.logger.debug("Assigning speakers to transcript chunks")
                        speaker_labels = speaker_results['speaker_labels']
                        chunks_with_speakers = 0
                        
                        for i, chunk in enumerate(result["chunks"]):
                            chunk_time = chunk["timestamp"][0]
                            chunk_idx = int((chunk_time * 16000) / (self.config.speaker_window_sec * 16000))
                            if chunk_idx < len(speaker_labels):
                                chunk["speaker"] = f"Speaker {speaker_labels[chunk_idx] + 1}"
                                chunks_with_speakers += 1
                        
                        self.logger.info(f"Assigned speakers to {chunks_with_speakers}/{len(result['chunks'])} chunks")
                        
                        # Save speaker information if enabled
                        if self.config.save_speaker_info:
                            speaker_info_path = self.config.output_text_path.replace('_output.txt', '_speaker_info.json')
                            with open(speaker_info_path, 'w', encoding='utf-8') as f:
                                json_dump(speaker_results['speaker_info'], f, indent=2)
                            self.logger.info(f"Speaker information saved to: {speaker_info_path}")
                    else:
                        self.logger.warning("Speaker recognition failed or returned no results")

                # Format transcript with paragraphs and speakers
                self.logger.info("Formatting transcript with paragraphs and speakers")
                formatted_text = format_transcript_with_paragraphs_and_speakers(result)
                save_text(formatted_text, self.config.output_text_path)
                self.logger.info(f"Transcription saved to: {self.config.output_text_path}")
                
                if self.config.generate_subtitles:
                    self.logger.info("Generating subtitles")
                    save_subtitles(result, self.config.output_subs_path)
                    self.logger.info(f"Subtitles saved to: {self.config.output_subs_path}")
                    
                if timeline is not None:
                    self.logger.info(f"Timeline saved to: {timeline_path}")
                if patterns is not None:
                    self.logger.info(f"Speech patterns saved to: {patterns_path}")
                    
                elapsed = time.time() - start_time
                self.logger.info(f"Processing completed in {elapsed:.2f} seconds")
                
                # Cleanup temporary files and memory
                try:
                    if audio_path != input_path:  # Only delete if it's a temporary extracted audio
                        os.unlink(audio_path)
                except Exception as e:
                    self.logger.warning(f"Could not delete temporary audio file: {e}")
                    
                # Clear memory
                del audio, sr
                if mfccs is not None:
                    del mfccs
                if 'result' in locals():
                    del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Save timing info
                timings_path = os.path.join(os.path.dirname(self.config.output_text_path), 'timings.txt')
                try:
                    os.makedirs(os.path.dirname(timings_path), exist_ok=True)
                    with open(timings_path, 'a', encoding='utf-8') as tf:
                        tf.write(f"{os.path.basename(self.config.output_text_path)}: {elapsed:.2f} seconds\n")
                except Exception as e:
                    self.logger.error(f"Could not save timing info: {e}")

            except KeyboardInterrupt:
                self.logger.warning("Processing interrupted by user")
                raise
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                raise
            finally:
                timer.cancel()

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by timeout")
            raise TimeoutError("Processing timeout exceeded")

    def run_realtime(self):
        """
        Real-time audio processing from microphone using sounddevice.
        Also extracts MFCCs and identifies speech patterns for each chunk.
        Saves results to output/results/ and transcripts to output/realtime/.
        """
        print("[Realtime Mode] Speak into your microphone. Press Ctrl+C to stop.")
        samplerate = 16000
        blocksize = 4096
        q = queue.Queue()
        # Prepare output directories
        results_dir = os.path.join("output", "results")
        realtime_dir = os.path.join("output", "realtime")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(realtime_dir, exist_ok=True)
        transcript_path = os.path.join(realtime_dir, "realtime_transcript.txt")
        chunk_idx = 1

        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())

        try:
            with open(transcript_path, 'w', encoding='utf-8') as transcript_file:
                with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', blocksize=blocksize, callback=callback):
                    audio_buffer = []
                    while True:
                        data = q.get()
                        audio_buffer.append(data)
                        # Process every ~5 seconds of audio
                        if len(audio_buffer) * blocksize >= samplerate * 5:
                            chunk = np.concatenate(audio_buffer, axis=0)
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                                sf.write(tmpfile.name, chunk, samplerate)
                                result = self.asr_pipeline(tmpfile.name, return_timestamps="word", generate_kwargs={"task": "transcribe", "language": self.config.language})
                                text = result["text"]
                                if self.config.punctuation_correction:
                                    text = correct_punctuation(text)
                                if self.config.capitalization:
                                    text = correct_capitalization(text)
                                # Extract MFCCs
                                mfccs = None
                                if self.config.mfcc:
                                    mfccs = extract_mfcc(chunk.flatten(), samplerate)
                                    mfcc_path = os.path.join(results_dir, f"mfcc_chunk_{chunk_idx}.npy")
                                    np.save(mfcc_path, mfccs)
                                # Identify speech patterns
                                patterns = None
                                if self.config.identify_speech_patterns:
                                    patterns = identify_speech_patterns(chunk.flatten(), samplerate)
                                    patterns_path = os.path.join(results_dir, f"patterns_chunk_{chunk_idx}.json")
                                    with open(patterns_path, 'w', encoding='utf-8') as pf:
                                        json_dump(patterns, pf, indent=2)
                                # Save transcript
                                transcript_file.write(text + "\n")
                                transcript_file.flush()
                                print(f"[Realtime Transcript] {text}")
                                if mfccs is not None:
                                    print(f"[Realtime MFCC shape] {mfccs.shape} (saved to {mfcc_path})")
                                if patterns is not None:
                                    print(f"[Realtime Speech Patterns] {patterns} (saved to {patterns_path})")
                            audio_buffer = []
                            chunk_idx += 1
        except KeyboardInterrupt:
            print("\n[Realtime Mode] Stopped.")
        except Exception as e:
            print(f"[Realtime Mode] Error: {e}")
