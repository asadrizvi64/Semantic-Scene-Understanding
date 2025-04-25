# tests/test_audio.py
"""
Tests for the audio processing module.
"""

import os
import pytest
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock

from modules.audio import AudioProcessor, DummyDiarizationPipeline, DummyAudioClassifier

# Test fixtures
@pytest.fixture
def test_config():
    return {
        "device": "cpu",
        "model_paths": {
            "whisper": "base"
        },
        "output_dir": tempfile.mkdtemp()
    }

@pytest.fixture
def test_audio_path():
    """Create a test audio file."""
    # Create a temporary audio file
    audio_path = os.path.join(tempfile.mkdtemp(), "test_audio.wav")
    
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a 440 Hz sine wave
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    # Save to WAV file
    sf.write(audio_path, audio_data, sample_rate)
    
    return audio_path

@pytest.mark.parametrize("has_whisper", [True, False])
def test_audio_processor_initialization(test_config, has_whisper):
    """Test that the audio processor initializes correctly with or without Whisper."""
    with patch("modules.audio.whisper", autospec=True) as mock_whisper:
        if not has_whisper:
            mock_whisper.load_model.side_effect = ImportError("No whisper available")
        
        processor = AudioProcessor(test_config)
        
        assert processor is not None
        assert processor.config == test_config
        
        # Check that Whisper status is correctly reflected
        if has_whisper:
            assert processor.whisper_model is not None
        else:
            assert processor.whisper_model is None

def test_process_audio(test_config, test_audio_path):
    """Test that audio is processed correctly."""
    # Mock Whisper to return predictable transcription
    with patch("modules.audio.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "This is a test.",
                    "confidence": 0.95
                },
                {
                    "start": 1.8,
                    "end": 3.0,
                    "text": "More test audio.",
                    "confidence": 0.9
                }
            ]
        }
        mock_whisper.load_model.return_value = mock_model
        
        processor = AudioProcessor(test_config)
        
        # Process the audio
        results = processor.process_audio(test_audio_path)
        
        # Check that we got results
        assert len(results) > 0
        
        # Check structure of the results - should be speech and non-speech segments
        has_speech = False
        has_non_speech = False
        
        for segment in results:
            assert "type" in segment
            assert "start" in segment
            assert "end" in segment
            
            if segment["type"] == "speech":
                has_speech = True
                assert "text" in segment
                assert "confidence" in segment
                assert "speaker" in segment
                assert "sentiment" in segment
            
            elif segment["type"] == "non-speech":
                has_non_speech = True
                assert "class" in segment
                assert "confidence" in segment
        
        # Should have both types of segments
        assert has_speech
        assert has_non_speech

def test_transcribe_speech(test_config, test_audio_path):
    """Test speech transcription specifically."""
    # Mock Whisper to return predictable transcription
    with patch("modules.audio.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "This is a test.",
                    "confidence": 0.95
                }
            ]
        }
        mock_whisper.load_model.return_value = mock_model
        
        processor = AudioProcessor(test_config)
        
        # Process speech
        speech_segments = processor._transcribe_speech(test_audio_path)
        
        # Check result
        assert len(speech_segments) == 1
        assert speech_segments[0]["text"] == "This is a test."
        assert speech_segments[0]["start"] == 0.0
        assert speech_segments[0]["end"] == 1.5
        assert speech_segments[0]["confidence"] == 0.95
        assert "sentiment" in speech_segments[0]

def test_diarize_speakers(test_config, test_audio_path):
    """Test speaker diarization."""
    # Create test speech segments
    speech_segments = [
        {
            "type": "speech",
            "start": 0.0,
            "end": 1.5,
            "text": "This is speaker one.",
            "confidence": 0.95,
            "speaker": "unknown"
        },
        {
            "type": "speech",
            "start": 2.0,
            "end": 3.5,
            "text": "This is speaker two.",
            "confidence": 0.9,
            "speaker": "unknown"
        }
    ]
    
    processor = AudioProcessor(test_config)
    
    # Use the dummy diarization pipeline
    processor.diarization_pipeline = DummyDiarizationPipeline()
    
    # Diarize speakers
    diarized_segments = processor._diarize_speakers(test_audio_path, speech_segments)
    
    # Check that speakers were assigned
    assert len(diarized_segments) == 2
    assert diarized_segments[0]["speaker"] != "unknown"
    assert diarized_segments[1]["speaker"] != "unknown"
    
    # Different segments should have different speakers
    assert diarized_segments[0]["speaker"] != diarized_segments[1]["speaker"]

def test_analyze_non_speech(test_config, test_audio_path):
    """Test non-speech audio analysis."""
    # Create test speech segments (to be excluded from non-speech analysis)
    speech_segments = [
        {
            "type": "speech",
            "start": 1.0,
            "end": 2.0,
            "text": "This is speech.",
            "confidence": 0.9,
            "speaker": "SPEAKER_1"
        }
    ]
    
    processor = AudioProcessor(test_config)
    
    # Use the dummy audio classifier
    processor.audio_classifier = DummyAudioClassifier()
    
    # Analyze non-speech
    non_speech_segments = processor._analyze_non_speech(test_audio_path, speech_segments)
    
    # Check that non-speech segments were detected
    assert len(non_speech_segments) > 0
    
    # Check that none of the segments overlap with speech
    for segment in non_speech_segments:
        assert segment["type"] == "non-speech"
        assert "class" in segment
        assert "confidence" in segment
        
        # Verify no overlap with speech
        for speech in speech_segments:
            # Either segment ends before speech starts or starts after speech ends
            assert segment["end"] <= speech["start"] or segment["start"] >= speech["end"]

def test_sentiment_analysis():
    """Test the sentiment analysis function."""
    processor = AudioProcessor({"device": "cpu"})
    
    # Test cases
    positive_text = "I am happy and enjoying this wonderful day."
    negative_text = "I am sad and angry about the terrible situation."
    neutral_text = "The sky is blue and the grass is green."
    
    # Analyze sentiment
    positive_sentiment = processor._analyze_sentiment(positive_text)
    negative_sentiment = processor._analyze_sentiment(negative_text)
    neutral_sentiment = processor._analyze_sentiment(neutral_text)
    
    # Check results
    assert positive_sentiment > 0
    assert negative_sentiment < 0
    assert -0.2 <= neutral_sentiment <= 0.2  # Approximately neutral

def test_dummy_diarization_pipeline(test_audio_path):
    """Test the dummy diarization pipeline."""
    pipeline = DummyDiarizationPipeline()
    
    # Call the pipeline
    result = pipeline(test_audio_path)
    
    # Check that it returns a result with turns
    turns = list(result.itertracks(yield_label=True))
    
    # Should have at least one turn
    assert len(turns) > 0
    
    # Each turn should have start, end times and a speaker label
    for turn, _, speaker in turns:
        assert hasattr(turn, "start")
        assert hasattr(turn, "end")
        assert speaker.startswith("SPEAKER_")

def test_dummy_audio_classifier():
    """Test the dummy audio classifier."""
    classifier = DummyAudioClassifier()
    
    # Generate a predictable result by using a fixed start time
    start_time = 1.0
    end_time = 2.0
    
    # Classify audio
    class_name, confidence = classifier.classify("dummy_path.wav", start_time, end_time)
    
    # Check result
    assert class_name in classifier.class_names
    assert 0.6 <= confidence <= 0.95  # Should be in this range
    
    # Same start time should give the same result (deterministic)
    class_name2, confidence2 = classifier.classify("dummy_path.wav", start_time, end_time)
    assert class_name == class_name2
    assert confidence == confidence2
    
    # Different start time should give a different result
    class_name3, confidence3 = classifier.classify("dummy_path.wav", start_time + 5.0, end_time + 5.0)
    assert class_name != class_name3 or confidence != confidence3

def test_error_handling_nonexistent_file(test_config):
    """Test error handling for nonexistent audio files."""
    processor = AudioProcessor(test_config)
    
    with pytest.raises(FileNotFoundError):
        processor.process_audio("nonexistent_audio.wav")