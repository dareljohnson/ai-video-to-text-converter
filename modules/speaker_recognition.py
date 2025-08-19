"""
Speaker recognition utilities for AI Video-to-Text Converter.

Authors: Bai Blyden, Darel Johnson
"""

import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import torch
import warnings
from scipy.signal import medfilt
import logging
from modules.utils import setup_logging

class SpeakerRecognizer:
    def __init__(self, config, min_speakers: int = 1, max_speakers: int = 8, window_sec: float = 3.0):
        """Initialize speaker recognition pipeline.
        
        Args:
            config: AppConfig object for logging setup
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            window_sec: Window size in seconds for speaker embedding extraction
        """
        self.config = config
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.window_sec = window_sec
        self.encoder = VoiceEncoder()
        
        # Initialize logging
        self.logger = setup_logging(config)
        self.logger.info(f"Initialized SpeakerRecognizer with {min_speakers}-{max_speakers} speakers")
        
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract speaker embeddings from audio file."""
        try:
            # Preprocess audio with error handling
            self.logger.debug(f"Preprocessing audio file: {audio_path}")
            wav = preprocess_wav(audio_path)
            
            # Extract embeddings using sliding windows
            self.logger.debug("Extracting embeddings with sliding windows")
            
            # Process audio in windows
            window_samples = int(16000 * self.window_sec)
            embeddings = []
            
            for i in range(0, len(wav), window_samples):
                window = wav[i:i + window_samples]
                if len(window) >= window_samples // 2:  # Only process if window is at least half full
                    embedding = self.encoder.embed_utterance(window)
                    embeddings.append(embedding)
                    
            embeddings = np.array(embeddings)
            
            self.logger.debug(f"Successfully extracted {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {str(e)}")
            return None
            
    def optimize_clusters(self, embeddings: np.ndarray) -> Tuple[int, float]:
        """Find optimal number of speaker clusters using silhouette score."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
            
        best_n = self.min_speakers
        best_score = -1
        
        self.logger.debug(f"Finding optimal number of speakers between {self.min_speakers} and {self.max_speakers}")
        
        # Try different numbers of clusters
        for n in range(self.min_speakers, self.max_speakers + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                self.logger.debug(f"Tried {n} speakers, silhouette score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_n = n
                    self.logger.debug(f"New best score with {n} speakers")
                    
            except Exception as e:
                self.logger.warning(f"Error calculating silhouette score for {n} clusters: {str(e)}")
                continue
                
        return best_n, best_score

    def assign_speakers(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Assign speaker labels to embeddings.
        
        Returns:
            Tuple of (speaker_labels, speaker_info)
            speaker_info contains statistics about each speaker
        """
        if embeddings is None or len(embeddings) == 0:
            return np.array([]), {}
            
        # Find optimal number of speakers
        n_speakers, score = self.optimize_clusters(embeddings)
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_speakers, random_state=42)
        speaker_labels = kmeans.fit_predict(embeddings)
        
        # Get speaker statistics
        speaker_info = {
            'n_speakers': n_speakers,
            'silhouette_score': score,
            'speaker_segments': {i: np.sum(speaker_labels == i) for i in range(n_speakers)}
        }
        
        # Apply median filter to smooth out speaker assignments
        speaker_labels = medfilt(speaker_labels, kernel_size=5)
        
        return speaker_labels, speaker_info

    @classmethod
    def process_audio(cls, config, audio_path: str, min_speakers: int = 1, max_speakers: int = 8, window_sec: float = 3.0) -> Dict:
        """Process audio file and return speaker diarization results.
        
        Args:
            config: AppConfig object for logging setup
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            window_sec: Window size for speaker embedding
        
        Returns:
            Dictionary containing:
            - speaker_labels: Array of speaker IDs
            - speaker_info: Statistics about detected speakers
            - embeddings: Speaker embeddings
            
            Returns None if processing fails.
        """
        recognizer = cls(config, min_speakers, max_speakers, window_sec)
        recognizer.logger.info(f"Starting speaker recognition for {audio_path}")
        
        try:
            # Extract speaker embeddings
            recognizer.logger.info("Extracting speaker embeddings...")
            embeddings = recognizer.extract_embeddings(audio_path)
            if embeddings is None:
                raise ValueError("Failed to extract speaker embeddings")
            
            # Assign speakers
            recognizer.logger.info("Assigning speakers to segments...")
            speaker_labels, speaker_info = recognizer.assign_speakers(embeddings)
            
            # Convert numpy types to native Python types
            speaker_labels = speaker_labels.tolist() if hasattr(speaker_labels, 'tolist') else speaker_labels
            speaker_info = {k: int(v) if isinstance(v, np.integer) else v for k, v in speaker_info.items()}
            
            recognizer.logger.info(f"Speaker recognition completed. Found {len(set(speaker_labels))} speakers")
            return {
                'speaker_labels': speaker_labels,
                'speaker_info': speaker_info,
                'embeddings': embeddings
            }
            
        except Exception as e:
            recognizer.logger.error(f"Speaker recognition failed: {str(e)}")
            return None
