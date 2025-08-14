"""
Pipeline for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
from modules.config import AppConfig
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

class VideoToTextPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        cuda_idx = config.cuda_device if self.device == "cuda" else -1
        if self.device == "cuda":
            n_gpus = torch.cuda.device_count()
            if cuda_idx >= n_gpus:
                raise ValueError(f"Requested CUDA device {cuda_idx}, but only {n_gpus} GPUs are available.")
            print(f"[INFO] Using CUDA device {cuda_idx}: {torch.cuda.get_device_name(cuda_idx)}")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.model_name,
            device=cuda_idx,
            model_kwargs={
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "attn_implementation": "eager"
            }
        )

    def run(self):
        import os
        import shutil
        if self.config.realtime:
            self.run_realtime()
            return
        # Check ffmpeg availability
        if not shutil.which("ffmpeg"):
            print("ERROR: ffmpeg is not found in your PATH. Please install ffmpeg and add it to your system PATH, then restart your terminal.")
            return
        # Project root is one level up from this file's directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Input path: relative to project root
        input_path = self.config.input_path
        while not input_path or not os.path.isfile(os.path.join(project_root, input_path)):
            input_path = input("Enter the RELATIVE path to an audio or video file in the project root: ").strip()
            if not os.path.isfile(os.path.join(project_root, input_path)):
                print(f"File not found: {input_path}. Please try again.")
        input_path = os.path.join(project_root, input_path)
        # Output paths: always relative to project root
        self.config.output_text_path = os.path.join(project_root, self.config.output_text_path)
        self.config.output_subs_path = os.path.join(project_root, self.config.output_subs_path)
        audio_path = extract_audio(input_path)
        audio, sr = librosa.load(audio_path, sr=None)
        if self.config.mfcc:
            mfccs = extract_mfcc(audio, sr)
        result = self.asr_pipeline(audio_path, return_timestamps="word", generate_kwargs={"task": "transcribe", "language": self.config.language})
        text = result["text"]
        if self.config.punctuation_correction:
            text = correct_punctuation(text)
        if self.config.capitalization:
            text = correct_capitalization(text)
        if self.config.track_timeline:
            timeline = track_timeline(result)
        else:
            timeline = None
        if self.config.identify_speech_patterns:
            patterns = identify_speech_patterns(audio, sr)
        else:
            patterns = None
        save_text(text, self.config.output_text_path)
        if self.config.generate_subtitles:
            save_subtitles(result, self.config.output_subs_path)
        print(f"Transcription saved to {self.config.output_text_path}")
        if self.config.generate_subtitles:
            print(f"Subtitles saved to {self.config.output_subs_path}")

    def run_realtime(self):
        """
        Real-time audio processing from microphone using sounddevice.
        Also extracts MFCCs and identifies speech patterns for each chunk.
        Saves results to output/results/ and transcripts to output/realtime/.
        """
        import sounddevice as sd
        import queue
        import numpy as np
        import tempfile
        import soundfile as sf
        import os
        import json
        from modules.utils import extract_mfcc, identify_speech_patterns
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
                                        json.dump(patterns, pf, indent=2)
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
