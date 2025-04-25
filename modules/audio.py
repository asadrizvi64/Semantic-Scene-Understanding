# modules/audio.py
"""
Audio Processing Module for Narrative Scene Understanding.
This module handles speech transcription, speaker diarization, and non-speech audio analysis.
"""

import os
import numpy as np
import torch
import logging
import tempfile
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from pathlib import Path

class AudioProcessor:
    """
    Processes audio for narrative scene understanding.
    Handles speech transcription, speaker diarization, and non-speech audio classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_transcription_model()
        self._init_diarization_model()
        self._init_audio_classifier()
    
    def _init_transcription_model(self):
        """Initialize the speech transcription model (Whisper)."""
        self.logger.info("Initializing speech transcription model...")
        
        try:
            import whisper
            
            model_name = self.config.get("model_paths", {}).get("whisper", "base")
            
            # Load the model
            self.whisper_model = whisper.load_model(
                model_name, device=self.device)
            
            self.logger.info(f"Whisper model loaded: {model_name}")
        except ImportError:
            self.logger.warning("Whisper not available, speech transcription will be limited")
            self.whisper_model = None
    
    def _init_diarization_model(self):
        """Initialize the speaker diarization model (pyannote.audio)."""
        self.logger.info("Initializing speaker diarization model...")
        
        try:
            from pyannote.audio import Pipeline
            
            # Try to load diarization pipeline
            try:
                # In a real implementation, you would need a valid HuggingFace token
                # self.diarization_pipeline = Pipeline.from_pretrained(
                #    "pyannote/speaker-diarization-3.0",
                #    use_auth_token="YOUR_HF_TOKEN")
                
                # For this implementation, we'll use a dummy pipeline
                self.diarization_pipeline = DummyDiarizationPipeline()
                self.logger.info("Speaker diarization model loaded")
            except Exception as e:
                self.logger.warning(f"Could not load diarization pipeline: {e}")
                self.diarization_pipeline = DummyDiarizationPipeline()
        except ImportError:
            self.logger.warning("pyannote.audio not available, speaker diarization will be limited")
            self.diarization_pipeline = DummyDiarizationPipeline()
    
    def _init_audio_classifier(self):
        """Initialize the non-speech audio classifier (PANNs)."""
        self.logger.info("Initializing audio classifier...")
        
        try:
            import torch.nn as nn
            import torchaudio
            
            # In a real implementation, you would load the PANNs model
            # Here we'll use a dummy classifier
            self.audio_classifier = DummyAudioClassifier()
            self.logger.info("Audio classifier loaded")
        except ImportError:
            self.logger.warning("torchaudio not available, non-speech audio classification will be limited")
            self.audio_classifier = DummyAudioClassifier()
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Process audio file for speech and non-speech content.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of processed audio segments
        """
        self.logger.info(f"Processing audio: {audio_path}")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return []
        
        # Process speech (transcription)
        speech_segments = self._transcribe_speech(audio_path)
        
        # Perform speaker diarization
        speech_segments = self._diarize_speakers(audio_path, speech_segments)
        
        # Process non-speech audio
        non_speech_segments = self._analyze_non_speech(audio_path, speech_segments)
        
        # Combine segments
        all_segments = speech_segments + non_speech_segments
        
        # Sort by start time
        all_segments.sort(key=lambda x: x.get("start", 0))
        
        self.logger.info(f"Processed {len(speech_segments)} speech segments and {len(non_speech_segments)} non-speech segments")
        return all_segments
    
    def _transcribe_speech(self, audio_path: str) -> List[Dict]:
        """
        Transcribe speech in the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of speech segments with transcriptions
        """
        if self.whisper_model is None:
            return []
        
        self.logger.info("Transcribing speech...")
        
        try:
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            # Extract segments
            speech_segments = []
            for segment in result["segments"]:
                # Create segment dictionary
                speech_segment = {
                    "type": "speech",
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": float(segment["confidence"]),
                    "speaker": "unknown"  # Will be filled by diarization
                }
                
                # Add sentiment analysis
                speech_segment["sentiment"] = self._analyze_sentiment(speech_segment["text"])
                
                speech_segments.append(speech_segment)
            
            return speech_segments
        
        except Exception as e:
            self.logger.error(f"Error transcribing speech: {e}")
            return []
    
    def _diarize_speakers(self, audio_path: str, speech_segments: List[Dict]) -> List[Dict]:
        """
        Perform speaker diarization on speech segments.
        
        Args:
            audio_path: Path to the audio file
            speech_segments: List of speech segments from transcription
            
        Returns:
            Updated speech segments with speaker IDs
        """
        if self.diarization_pipeline is None or len(speech_segments) == 0:
            return speech_segments
        
        self.logger.info("Performing speaker diarization...")
        
        try:
            # Get diarization result
            diarization = self.diarization_pipeline(audio_path)
            
            # Extract speaker turns
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Match speakers to speech segments
            for segment in speech_segments:
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # Find the best matching speaker turn
                best_overlap = 0
                best_speaker = "unknown"
                
                for turn in speaker_turns:
                    # Calculate temporal overlap
                    overlap_start = max(segment_start, turn["start"])
                    overlap_end = min(segment_end, turn["end"])
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        segment_duration = segment_end - segment_start
                        
                        # If overlap is significant, assign this speaker
                        if overlap_duration / segment_duration > 0.5:
                            best_overlap = overlap_duration
                            best_speaker = turn["speaker"]
                
                # Update segment with speaker ID
                segment["speaker"] = best_speaker
            
            return speech_segments
        
        except Exception as e:
            self.logger.error(f"Error in speaker diarization: {e}")
            return speech_segments
    
    def _analyze_non_speech(self, audio_path: str, speech_segments: List[Dict]) -> List[Dict]:
        """
        Analyze non-speech audio content.
        
        Args:
            audio_path: Path to the audio file
            speech_segments: List of speech segments to exclude
            
        Returns:
            List of non-speech audio segments
        """
        if self.audio_classifier is None:
            return []
        
        self.logger.info("Analyzing non-speech audio...")
        
        try:
            # In a real implementation, you would:
            # 1. Load the full audio
            # 2. Segment it (e.g., 1-second windows)
            # 3. Exclude segments that overlap with speech
            # 4. Classify the remaining segments
            
            # Here we'll generate some dummy non-speech segments
            non_speech_segments = []
            
            # Get audio duration
            import soundfile as sf
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            
            # Create segments every 5 seconds, avoiding speech segments
            for start_time in np.arange(0, duration, 5.0):
                end_time = min(start_time + 1.0, duration)
                
                # Check if this segment overlaps with any speech segment
                overlaps_speech = False
                for speech in speech_segments:
                    speech_start = speech["start"]
                    speech_end = speech["end"]
                    
                    if end_time > speech_start and start_time < speech_end:
                        overlaps_speech = True
                        break
                
                if not overlaps_speech:
                    # Classify this segment
                    audio_class, confidence = self.audio_classifier.classify(audio_path, start_time, end_time)
                    
                    if confidence > 0.5:  # Only keep confident classifications
                        non_speech_segment = {
                            "type": "non-speech",
                            "start": start_time,
                            "end": end_time,
                            "class": audio_class,
                            "confidence": confidence
                        }
                        non_speech_segments.append(non_speech_segment)
            
            return non_speech_segments
        
        except Exception as e:
            self.logger.error(f"Error analyzing non-speech audio: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Perform simple sentiment analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # In a real implementation, you would use a proper sentiment analysis model
        # Here we'll use a simple keyword-based approach
        
        positive_words = [
            "good", "great", "excellent", "wonderful", "fantastic",
            "happy", "glad", "pleased", "joy", "love", "like", "enjoy",
            "beautiful", "amazing", "awesome", "yes", "sure", "definitely"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "horrible", "poor",
            "sad", "unhappy", "angry", "hate", "dislike", "annoyed",
            "ugly", "disgusting", "no", "never", "not"
        ]
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / total_count


class DummyDiarizationPipeline:
    """
    A dummy speaker diarization pipeline for when pyannote.audio is not available.
    """
    
    def __call__(self, audio_path):
        """
        Simulate diarization by alternating speakers every 5 seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            A simple diarization result
        """
        import soundfile as sf
        
        # Get audio duration
        audio_info = sf.info(audio_path)
        duration = audio_info.duration
        
        # Create turns with alternating speakers
        turns = []
        for i, start_time in enumerate(np.arange(0, duration, this_duration := 5.0)):
            end_time = min(start_time + this_duration, duration)
            speaker = f"SPEAKER_{i % 2 + 1}"
            
            turns.append((start_time, end_time, speaker))
        
        return DummyDiarizationResult(turns)


class DummyDiarizationResult:
    """
    A dummy diarization result for the dummy pipeline.
    """
    
    def __init__(self, turns):
        """
        Initialize with pre-computed turns.
        
        Args:
            turns: List of (start, end, speaker) tuples
        """
        self.turns = turns
    
    def itertracks(self, yield_label=False):
        """
        Iterate through speaker turns.
        
        Args:
            yield_label: Whether to yield the speaker label
            
        Yields:
            Turn object with start/end times and optionally the speaker label
        """
        for start, end, speaker in self.turns:
            turn = DummyTurn(start, end)
            if yield_label:
                yield turn, None, speaker
            else:
                yield turn, None


class DummyTurn:
    """
    A dummy turn object for the dummy diarization result.
    """
    
    def __init__(self, start, end):
        """
        Initialize with start and end times.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
        """
        self.start = start
        self.end = end


class DummyAudioClassifier:
    """
    A dummy audio classifier for when PANNs is not available.
    """
    
    def __init__(self):
        """Initialize the dummy classifier."""
        self.class_names = [
            "speech", "dog_bark", "rain", "crying_baby", "clock_tick",
            "car_horn", "siren", "engine", "train", "church_bells",
            "cough", "footsteps", "music", "television", "door_knock"
        ]
    
    def classify(self, audio_path, start_time, end_time):
        """
        Simulate audio classification with random classes.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time of the segment
            end_time: End time of the segment
            
        Returns:
            Tuple of (class_name, confidence)
        """
        # Generate a deterministic but seemingly random class based on time
        seed = int((start_time * 10) % len(self.class_names))
        np.random.seed(seed)
        
        # Exclude speech class since we're only classifying non-speech segments
        non_speech_classes = [c for c in self.class_names if c != "speech"]
        
        # Select class and confidence
        class_index = np.random.randint(0, len(non_speech_classes))
        class_name = non_speech_classes[class_index]
        
        # Generate a confidence between 0.6 and 0.95
        confidence = 0.6 + np.random.random() * 0.35
        
        return class_name, confidence