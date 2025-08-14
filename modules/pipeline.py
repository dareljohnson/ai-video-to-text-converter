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
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config.model_name,
            device=0 if self.device == "cuda" else -1,
            model_kwargs={
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "attn_implementation": "eager"
            }
        )

    def run(self):
        import os
        import shutil
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
