from pydantic import BaseModel
from typing import Optional

class AppConfig(BaseModel):
    model_name: str = "openai/whisper-small"
    device: str = "cuda"  # fallback to cpu if not available
    input_path: Optional[str] = "input/GPT-4o API Crash Course for Beginners.mp4"  # relative to project root
    output_text_path: str = "output/output.txt"  # relative to project root
    output_subs_path: str = "output/output.srt"  # relative to project root
    language: Optional[str] = None  # None = auto-detect
    generate_subtitles: bool = True
    mfcc: bool = True
    punctuation_correction: bool = True
    capitalization: bool = True
    track_timeline: bool = True
    identify_speech_patterns: bool = True
