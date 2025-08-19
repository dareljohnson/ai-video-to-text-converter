"""
Configuration for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""

from pydantic import BaseModel, validator
import os
from typing import Optional, Literal

class AppConfig(BaseModel):
    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True
    log_file: str = "output/pipeline.log"
    show_progress: bool = True  # Show detailed progress in console
    # Speaker recognition settings
    enable_speaker_recognition: bool = True  # Use built-in speaker recognition
    min_speakers: int = 1  # Minimum number of speakers to detect
    max_speakers: int = 10  # Maximum number of speakers to try when auto-estimating
    speaker_window_sec: float = 3.0  # Window size for speaker embedding (seconds)
    save_speaker_info: bool = True  # Save detailed speaker information
    
    # Diarization options
    n_speakers: Optional[int] = 6  # If None, auto-estimate
    model_name: str = "openai/whisper-small"
    device: str = "cuda"  # fallback to cpu if not available
    cuda_device: int = 0  # GPU index to use if device is cuda
    input_path: Optional[str] = "input"  # Can be a file or directory
    output_text_path: str = "output/output.txt"  # Output path for transcribed text
    output_subs_path: str = "output/output.srt"  # Output path for subtitles
    
    language: Optional[str] = None  # None = auto-detect
    generate_subtitles: bool = True
    mfcc: bool = True
    punctuation_correction: bool = True
    capitalization: bool = True
    track_timeline: bool = True
    identify_speech_patterns: bool = True
    realtime: bool = False  # If True, enable real-time audio processing

    # Thread/environment variable options
    omp_num_threads: int = 6
    mkl_num_threads: int = 6
    openblas_num_threads: int = 4
    numexpr_num_threads: int = 12
    torch_num_threads: int = 8

    # Validate paths on initialization
    def __init__(self, **data):
        super().__init__(**data)
        # Convert any path starting with ./ to absolute path
        if self.input_path and self.input_path.startswith("./"):
            self.input_path = os.path.abspath(self.input_path[2:])
        if self.output_text_path.startswith("./"):
            self.output_text_path = os.path.abspath(self.output_text_path[2:])
        if self.output_subs_path.startswith("./"):
            self.output_subs_path = os.path.abspath(self.output_subs_path[2:])

        # Only validate paths if we're not in test mode
        skip_validation = data.get('_skip_validation', False)
        
        if not skip_validation:
            # Create standard directories if they don't exist
            default_dirs = ['input', 'output', 'logs']
            for dir_name in default_dirs:
                os.makedirs(dir_name, exist_ok=True)
            
            # Check or create input path
            if self.input_path:
                if os.path.isfile(self.input_path):
                    # For files, ensure parent directory exists
                    input_dir = os.path.dirname(self.input_path)
                    if input_dir:
                        os.makedirs(input_dir, exist_ok=True)
                else:
                    # For directories, create them
                    os.makedirs(self.input_path, exist_ok=True)
            
            # Create output directories if they don't exist
            text_dir = os.path.dirname(self.output_text_path)
            subs_dir = os.path.dirname(self.output_subs_path)
            
            if text_dir:
                os.makedirs(text_dir, exist_ok=True)
            if subs_dir:
                os.makedirs(subs_dir, exist_ok=True)
            
            # Ensure log directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    @validator('input_path', 'output_text_path', 'output_subs_path')
    def validate_path(cls, v):
        if v is None:  # Allow None for optional input_path
            return v
        if any(c in v for c in ['<', '>', '|', '*', '?']):
            raise ValueError(f"Invalid characters in path: {v}")
        return v