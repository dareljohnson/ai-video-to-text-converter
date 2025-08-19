"""
Utility functions for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""

import os
import re
import sys
import logging
from pathlib import Path
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from typing import Optional
import numpy as np

def setup_logging(config):
    """Configure logging based on config settings.
    
    Args:
        config: AppConfig object containing logging settings
        
    Returns:
        logger: Configured logger instance
    """
    log_handlers = []
    
    # Console handler with custom formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    log_handlers.append(console_handler)
    
    # File handler if enabled
    if config.log_to_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        log_handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=log_handlers,
        force=True
    )
    
    # Create logger for the calling module
    logger = logging.getLogger(config.__class__.__module__)
    logger.info(f"Logging initialized at level {config.log_level}")
    
    return logger

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

def format_transcript_with_paragraphs_and_speakers(result, min_pause_s=1.0):
    """
    Format transcript with paragraph breaks (based on pauses) and speaker labels.
    
    Args:
        result: ASR pipeline output with 'chunks' (word-level timestamps with optional speaker labels)
        min_pause_s: minimum pause (in seconds) to start a new paragraph
    
    Returns:
        formatted string with speaker labels and paragraph breaks
    """
    chunks = result.get('chunks', [])
    if not chunks:
        return result.get('text', '')
        
    paragraphs = []
    current_para = []
    last_end = None
    current_speaker = None
    
    for chunk in chunks:
        word = chunk['text']
        start, end = chunk['timestamp']
        speaker = chunk.get('speaker', None)  # Get speaker if available
        
        # Start new paragraph on long pauses or speaker changes
        if last_end is not None:
            pause = start - last_end
            new_speaker = speaker != current_speaker
            
            if pause >= min_pause_s or new_speaker:
                if current_para:
                    para_text = ' '.join(current_para)
                    if current_speaker:
                        para_text = f"{current_speaker}: {para_text}"
                    paragraphs.append(para_text)
                current_para = []
                current_speaker = speaker
        # Add word to current paragraph
        current_para.append(word)
        last_end = end

    # Add final paragraph
    if current_para:
        para_text = ' '.join(current_para)
        if current_speaker:
            para_text = f"{current_speaker}: {para_text}"
        paragraphs.append(para_text)

    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs)

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

def assign_speakers_to_chunks(chunks, segments):
    """Assign speaker labels to each chunk based on diarization segments.
    Uses the built-in speaker recognition system."""
    for chunk in chunks:
        start, end = chunk['timestamp']
        overlaps = [(seg_start, seg_end, speaker) for seg_start, seg_end, speaker in segments if not (end < seg_start or start > seg_end)]
        if overlaps:
            best = max(overlaps, key=lambda seg: min(end, seg[1]) - max(start, seg[0]))
            chunk['speaker'] = best[2]
        else:
            chunk['speaker'] = 'Unknown'
    return chunks

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis strings.
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    d = np.zeros((len(ref_words)+1, len(hyp_words)+1), dtype=np.uint8)
    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    wer = d[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))
    return wer

def save_subtitles(result, path: str):
    # Save SRT subtitles from ASR result
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(result.get('chunks', []), 1):
            start = chunk['timestamp'][0] if chunk['timestamp'] and len(chunk['timestamp']) > 0 else 0
            end = chunk['timestamp'][1] if chunk['timestamp'] and len(chunk['timestamp']) > 1 else start + 1
            text = chunk['text']
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

def format_srt_time(seconds: float) -> str:
    if seconds is None:
        seconds = 0
    seconds = float(seconds)  # Convert to float in case we get an int or string
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60) 
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_text(text: str, path: str):
    dirpart = os.path.dirname(path)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)