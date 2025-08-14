"""
Utility functions for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""
import os
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np

def extract_audio(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    audio_path = input_path
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video = VideoFileClip(input_path)
        audio_path = input_path + ".wav"
        video.audio.write_audiofile(audio_path)
    return audio_path

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def correct_punctuation(text: str) -> str:
    # Rule-based: split after first word for any two-word input, after first two for >4 words, else just add period
    import re
    words = text.strip().split()
    if not words:
        return ''
    # If already punctuated, return as is
    if re.search(r'[.!?]', text):
        return text
    if len(words) == 2:
        first = words[0] + '.'
        second = words[1] + '.'
        return f"{first} {second[0].upper() + second[1:]}"
    if len(words) > 4:
        first = ' '.join(words[:2]) + '.'
        second = ' '.join(words[2:]) + '.'
        return f"{first} {second[0].upper() + second[1:]}"
    # Otherwise, just add period
    return text + '.'

def correct_capitalization(text: str) -> str:
    # Simple rule-based: capitalize first letter of each sentence
    import re
    def cap(match):
        return match.group(1) + match.group(2).upper()
    # Capitalize first letter
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    # Capitalize after .!? and space
    text = re.sub(r'([\.!?]\s+)([a-z])', cap, text)
    return text

def track_timeline(result) -> list:
    # Extract word-level timestamps from ASR result
    return result.get('chunks', [])

def identify_speech_patterns(audio: np.ndarray, sr: int):
    # Simple speech pattern analysis: detect pauses and speaking rate
    import librosa
    # Detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=30)
    durations = [(end - start) / sr for start, end in intervals]
    total_speech = sum(durations)
    total_audio = len(audio) / sr
    speaking_rate = len(intervals) / total_audio if total_audio > 0 else 0
    avg_speech_duration = np.mean(durations) if durations else 0
    avg_pause_duration = (total_audio - total_speech) / (len(intervals) - 1) if len(intervals) > 1 else 0
    return {
        'speaking_rate': speaking_rate,
        'avg_speech_duration': avg_speech_duration,
        'avg_pause_duration': avg_pause_duration,
        'num_segments': len(intervals)
    }

def save_subtitles(result, path: str):
    # Save SRT subtitles from ASR result
    import os
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(result.get('chunks', []), 1):
            start = chunk['timestamp'][0]
            end = chunk['timestamp'][1]
            text = chunk['text']
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_text(text: str, path: str):
    import os
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
