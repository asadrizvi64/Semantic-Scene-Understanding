# tests/test_ocr.py
"""
Tests for the OCR processing module.
"""

import os
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from modules.ocr import OCRProcessor

# Test fixtures
@pytest.fixture
def test_config():
    return {
        "device": "cpu",
        "ocr_engine": "easyocr",
        "min_text_size": 10,
        "min_ocr_confidence": 0.5,
        "ocr_frequency": 2  # Process every 2nd frame
    }

@pytest.fixture
def test_frames():
    """Generate test frames with text for OCR processing."""
    frames = []
    
    # Create a few test frames
    for i in range(5):
        # Create frame with timestamp
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
        
        # Add text to the frame
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "TEST TEXT", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        if i % 2 == 0:  # Add changing text on even frames
            cv2.putText(frame, "EVEN FRAME", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:  # Different text on odd frames
            cv2.putText(frame, "ODD FRAME", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add timestamp (0.5 seconds apart)
        timestamp = i * 0.5
        
        frames.append((timestamp, frame))
    
    return frames

@pytest.mark.parametrize("has_easyocr", [True, False])
@pytest.mark.parametrize("has_tesseract", [True, False])
def test_ocr_processor_initialization(test_config, has_easyocr, has_tesseract):
    """Test that the OCR processor initializes correctly with available OCR engines."""
    with patch("modules.ocr.easyocr", autospec=True) as mock_easyocr:
        if not has_easyocr:
            mock_easyocr.Reader.side_effect = ImportError("No EasyOCR available")
        
        with patch("modules.ocr.pytesseract", autospec=True) as mock_tesseract:
            if not has_tesseract:
                mock_tesseract.image_to_data.side_effect = ImportError("No Tesseract available")
            
            processor = OCRProcessor(test_config)
            
            assert processor is not None
            assert processor.config == test_config
            
            # Check OCR engine based on availability
            if has_easyocr:
                assert processor.reader is not None
            elif has_tesseract:
                assert processor.reader is not None
            else:
                assert processor.reader is None

def test_process_frames(test_config, test_frames):
    """Test that frames are processed for OCR."""
    # Mock EasyOCR to return predictable results
    with patch("modules.ocr.easyocr") as mock_easyocr:
        # Create a mock reader that returns some text detections
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (
                [[200, 180], [400, 180], [400, 220], [200, 220]],  # Box points
                "TEST TEXT",  # Text
                0.95  # Confidence
            ),
            (
                [[300, 280], [500, 280], [500, 320], [300, 320]],  # Box points
                "FRAME",  # Text
                0.85  # Confidence
            )
        ]
        mock_easyocr.Reader.return_value = mock_reader
        
        processor = OCRProcessor(test_config)
        
        # Process frames
        results = processor.process_frames(test_frames)
        
        # Check that we got results for each frame
        assert len(results) == len(test_frames)
        
        # Check structure of the results
        for frame_data in results:
            assert "timestamp" in frame_data
            assert "frame_idx" in frame_data
            assert "text_detections" in frame_data
            
            # Every even frame should be directly processed (given ocr_frequency=2)
            if int(frame_data["frame_idx"]) % 2 == 0:
                # Should have detected text
                assert len(frame_data["text_detections"]) > 0
                
                # Check text detection structure
                text_det = frame_data["text_detections"][0]
                assert "text" in text_det
                assert "confidence" in text_det
                assert "box" in text_det
                assert "polygon" in text_det

def test_text_tracking(test_config, test_frames):
    """Test that text is tracked across frames."""
    # Mock text detection to return consistent text with slight movement
    with patch.object(OCRProcessor, "_detect_text") as mock_detect:
        # Frame 0 detections
        mock_detect.side_effect = [
            [
                {"text": "TEST TEXT", "confidence": 0.95, "box": [200, 180, 400, 220], 
                 "polygon": [[200, 180], [400, 180], [400, 220], [200, 220]]},
                {"text": "EVEN FRAME", "confidence": 0.9, "box": [300, 280, 500, 320], 
                 "polygon": [[300, 280], [500, 280], [500, 320], [300, 320]]}
            ],
            # Frame 1 detections (propagated from frame 0 with confidence drop)
            [],  # Will be populated by _track_text
            # Frame 2 detections (text moved slightly)
            [
                {"text": "TEST TEXT", "confidence": 0.95, "box": [205, 180, 405, 220], 
                 "polygon": [[205, 180], [405, 180], [405, 220], [205, 220]]},
                {"text": "EVEN FRAME", "confidence": 0.9, "box": [305, 280, 505, 320], 
                 "polygon": [[305, 280], [505, 280], [505, 320], [305, 320]]}
            ],
            # Frame 3 (propagated again)
            [],
            # Frame 4 (text moved more)
            [
                {"text": "TEST TEXT", "confidence": 0.95, "box": [210, 180, 410, 220], 
                 "polygon": [[210, 180], [410, 180], [410, 220], [210, 220]]},
                {"text": "EVEN FRAME", "confidence": 0.9, "box": [310, 280, 510, 320], 
                 "polygon": [[310, 280], [510, 280], [510, 320], [310, 320]]}
            ]
        ]
        
        processor = OCRProcessor(test_config)
        
        # Process frames
        results = processor.process_frames(test_frames)
        
        # Find frames with the tracked text
        text_occurrences = {}
        
        for frame_data in results:
            for text_det in frame_data["text_detections"]:
                text = text_det["text"]
                if text not in text_occurrences:
                    text_occurrences[text] = []
                text_occurrences[text].append(frame_data["frame_idx"])
        
        # "TEST TEXT" should appear in all processed frames
        assert "TEST TEXT" in text_occurrences
        assert len(text_occurrences["TEST TEXT"]) > 0
        
        # Check for tracking information
        has_tracking_info = False
        
        for frame_data in results:
            for text_det in frame_data["text_detections"]:
                if "track_id" in text_det and "first_seen" in text_det and "occurrences" in text_det:
                    has_tracking_info = True
                    break
            if has_tracking_info:
                break
        
        assert has_tracking_info, "Text tracking information not found"

def test_detect_with_easyocr(test_config):
    """Test text detection with EasyOCR specifically."""
    with patch("modules.ocr.easyocr") as mock_easyocr:
        # Create a mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (
                [[100, 100], [200, 100], [200, 150], [100, 150]],  # Box points
                "SAMPLE TEXT",  # Text
                0.9  # Confidence
            )
        ]
        mock_easyocr.Reader.return_value = mock_reader
        
        processor = OCRProcessor(test_config)
        processor.reader = mock_reader
        
        # Create a simple test frame
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "SAMPLE TEXT", (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Detect text
        detections = processor._detect_with_easyocr(frame)
        
        # Check results
        assert len(detections) == 1
        assert detections[0]["text"] == "SAMPLE TEXT"
        assert detections[0]["confidence"] == 0.9
        assert "box" in detections[0]
        assert "polygon" in detections[0]

def test_detect_with_tesseract(test_config):
    """Test text detection with Tesseract specifically."""
    with patch("modules.ocr.pytesseract") as mock_tesseract:
        # Create mock Tesseract output
        mock_data = {
            'text': ['SAMPLE', 'TEXT', ''],
            'conf': [90, 85, -1],
            'left': [100, 170, 0],
            'top': [100, 100, 0],
            'width': [60, 40, 0],
            'height': [30, 30, 0]
        }
        
        # Setup Output structure to match what's expected
        class Output:
            DICT = 'dict'
        mock_tesseract.Output = Output()
        mock_tesseract.image_to_data.return_value = mock_data
        
        processor = OCRProcessor(test_config)
        processor.reader = mock_tesseract
        
        # Create a simple test frame
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "SAMPLE TEXT", (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Detect text
        detections = processor._detect_with_tesseract(frame)
        
        # Check results
        assert len(detections) == 2  # "SAMPLE" and "TEXT"
        
        # Combine detections to check the full text (they might be in any order)
        combined_text = ' '.join([d["text"] for d in detections])
        assert "SAMPLE" in combined_text
        assert "TEXT" in combined_text
        
        # Check detection structure
        for detection in detections:
            assert "text" in detection
            assert "confidence" in detection
            assert "box" in detection
            assert "polygon" in detection

def test_ocr_frequency(test_config, test_frames):
    """Test that OCR processing respects the ocr_frequency setting."""
    with patch.object(OCRProcessor, "_process_single_frame", side_effect=lambda ts, frame, idx: {
        "timestamp": ts,
        "frame_idx": idx,
        "text_detections": [{"text": f"Frame {idx}", "confidence": 0.9, "box": [0, 0, 50, 20]}]
    }):
        # Set OCR frequency to 2 (process every 2nd frame)
        config = test_config.copy()
        config["ocr_frequency"] = 2
        
        processor = OCRProcessor(config)
        
        # Process frames
        results = processor.process_frames(test_frames)
        
        # Check which frames were directly processed vs. propagated
        directly_processed = []
        propagated = []
        
        for i, frame_data in enumerate(results):
            # Hack to detect if this was direct processing or propagation:
            # Directly processed frames call _process_single_frame which adds one detection
            # Propagated frames copy from the previous frame and might adjust confidence
            if "_process_single_frame" in [m[0] for m in processor._process_single_frame.call_args_list]:
                directly_processed.append(i)
            else:
                propagated.append(i)
        
        # Only frames at indices 0, 2, 4 should be directly processed
        expected_direct = [i for i in range(len(test_frames)) if i % 2 == 0]
        assert set(directly_processed) == set(expected_direct)
        
        # The rest should be propagated
        expected_propagated = [i for i in range(len(test_frames)) if i % 2 != 0]
        assert set(propagated) == set(expected_propagated)

def test_error_handling_no_ocr_engine(test_config):
    """Test error handling when no OCR engine is available."""
    with patch("modules.ocr.easyocr", None):
        with patch("modules.ocr.pytesseract", None):
            # Create processor without OCR engines
            processor = OCRProcessor(test_config)
            
            # Process frames should return empty detections but not fail
            results = processor.process_frames([(0.0, np.zeros((100, 100, 3), dtype=np.uint8))])
            
            assert len(results) == 1
            assert results[0]["text_detections"] == []