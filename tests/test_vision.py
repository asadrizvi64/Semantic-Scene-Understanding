# tests/test_vision.py
"""
Tests for the visual processing module.
"""

import os
import pytest
import numpy as np
import cv2
import tempfile
from unittest.mock import patch, MagicMock

from modules.vision import VisualProcessor, SimpleTracker, SimpleTrack

# Test fixtures
@pytest.fixture
def test_config():
    return {
        "device": "cpu",
        "model_paths": {
            "sam": "path/to/mock/sam_model.pth",
            "face_recognition": "path/to/mock/face_model"
        },
        "min_confidence": 0.5
    }

@pytest.fixture
def test_frames():
    """Generate test frames for processing."""
    frames = []
    
    # Create a few test frames
    for i in range(3):
        # Create frame with timestamp
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a "person"
        x_pos = 100 + i * 50
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 100, 300), (0, 255, 0), -1)
        
        # Add a face
        cv2.circle(frame, (x_pos + 50, 150), 30, (255, 255, 255), -1)
        
        # Add an "object"
        cv2.rectangle(frame, (400, 200), (450, 250), (0, 0, 255), -1)
        
        # Add timestamp (0.5 seconds apart)
        timestamp = i * 0.5
        
        frames.append((timestamp, frame))
    
    return frames

@pytest.mark.parametrize("has_sam", [True, False])
def test_visual_processor_initialization(test_config, has_sam):
    """Test that the visual processor initializes correctly with or without SAM."""
    with patch("modules.vision.SamPredictor", autospec=True) as mock_sam:
        if not has_sam:
            mock_sam.side_effect = ImportError("No SAM available")
            
        with patch("torch.hub.load", return_value=MagicMock()) as mock_yolo:
            processor = VisualProcessor(test_config)
            
            assert processor is not None
            assert processor.config == test_config
            
            # Check that fallback to YOLO works when SAM is not available
            if has_sam:
                assert processor.sam_predictor is not None
                mock_yolo.assert_not_called()
            else:
                assert processor.sam_predictor is None
                mock_yolo.assert_called_once()

def test_process_frames(test_config, test_frames):
    """Test that frames are processed and objects are detected."""
    # Mock SAM and BLIP to return predictable results
    with patch("modules.vision.SamPredictor", autospec=True) as mock_sam:
        # Create a mock SAM predictor
        mock_sam_instance = MagicMock()
        mock_sam_instance.predict.return_value = (
            [np.ones((480, 640), dtype=bool)],  # One mask covering the whole image
            [0.9],  # Confidence score
            None   # Logits
        )
        mock_sam.return_value = mock_sam_instance
        
        # Mock BLIP to return predictable captions
        with patch.object(VisualProcessor, "_generate_frame_caption", return_value="A person in a room"):
            with patch.object(VisualProcessor, "_generate_object_caption", return_value="A person"):
                processor = VisualProcessor(test_config)
                
                # Process frames
                results = processor.process_frames(test_frames)
                
                # Check that we got results for each frame
                assert len(results) == len(test_frames)
                
                # Check structure of the results
                for frame_data in results:
                    assert "timestamp" in frame_data
                    assert "overall_caption" in frame_data
                    assert "objects" in frame_data
                    
                    # Should have detected at least one object
                    assert len(frame_data["objects"]) > 0
                    
                    # Check object structure
                    obj = frame_data["objects"][0]
                    assert "id" in obj
                    assert "type" in obj
                    assert "box" in obj
                    assert "score" in obj
                    assert "caption" in obj

def test_object_tracking(test_config, test_frames):
    """Test that objects are tracked across frames."""
    with patch.object(VisualProcessor, "_detect_objects") as mock_detect:
        # Return consistent objects across frames with slight movement
        mock_detect.side_effect = [
            # Frame 1 objects
            [
                {"id": "0_0", "type": "person", "box": [100, 100, 200, 300], "score": 0.9, "caption": "A person"},
                {"id": "0_1", "type": "object", "box": [400, 200, 450, 250], "score": 0.8, "caption": "An object"}
            ],
            # Frame 2 objects (person moved slightly)
            [
                {"id": "1_0", "type": "person", "box": [150, 100, 250, 300], "score": 0.9, "caption": "A person"},
                {"id": "1_1", "type": "object", "box": [400, 200, 450, 250], "score": 0.8, "caption": "An object"}
            ],
            # Frame 3 objects (person moved more)
            [
                {"id": "2_0", "type": "person", "box": [200, 100, 300, 300], "score": 0.9, "caption": "A person"},
                {"id": "2_1", "type": "object", "box": [400, 200, 450, 250], "score": 0.8, "caption": "An object"}
            ]
        ]
        
        # Create processor with SimpleTracker
        processor = VisualProcessor(test_config)
        processor.tracker = SimpleTracker()
        
        # Process frames
        results = processor.process_frames(test_frames)
        
        # Check that tracking IDs were assigned
        for frame_data in results:
            for obj in frame_data["objects"]:
                if obj["type"] == "person":
                    assert obj["track_id"] is not None

def test_face_recognition(test_config, test_frames):
    """Test that faces are recognized in frames."""
    with patch.object(VisualProcessor, "_detect_objects") as mock_detect:
        # Return a person with a face
        mock_detect.return_value = [
            {"id": "0_0", "type": "person", "box": [100, 100, 200, 300], "score": 0.9, "caption": "A person"}
        ]
        
        # Mock face recognition to return a face
        with patch.object(VisualProcessor, "_recognize_faces") as mock_recognize:
            mock_recognize.return_value = [
                {"box": [120, 120, 180, 180], "score": 0.8, "embedding": np.random.rand(512)}
            ]
            
            processor = VisualProcessor(test_config)
            
            # Process a single frame
            frame_data = processor._process_single_frame(0.0, test_frames[0][1], 0)
            
            # Check that faces were recognized
            assert "faces" in frame_data
            assert len(frame_data["faces"]) == 1

def test_action_detection(test_config, test_frames):
    """Test that actions are detected in frames."""
    with patch.object(VisualProcessor, "_detect_objects") as mock_detect:
        # Return a person
        mock_detect.return_value = [
            {"id": "0_0", "type": "person", "box": [100, 100, 200, 300], "score": 0.9, "caption": "A person"}
        ]
        
        # Create a mock tracker that returns velocity information
        mock_track = MagicMock()
        mock_track.is_confirmed.return_value = True
        mock_track.track_id = 1
        mock_track.velocity = (5.0, 0.0)  # Moving right
        
        with patch.object(VisualProcessor, "_track_objects") as mock_track_objects:
            mock_track_objects.return_value = [mock_track]
            
            processor = VisualProcessor(test_config)
            
            # Process a single frame
            frame_data = processor._process_single_frame(0.0, test_frames[0][1], 0)
            
            # Check that actions were detected
            assert "actions" in frame_data
            assert len(frame_data["actions"]) > 0
            
            # Check action structure
            action = frame_data["actions"][0]
            assert "type" in action
            assert "subject_id" in action
            assert "confidence" in action

def test_simple_tracker():
    """Test the SimpleTracker fallback functionality."""
    tracker = SimpleTracker()
    
    # Create some test detections (boxes in format [x1, y1, x2, y2, score, class_id])
    detections1 = [
        [100, 100, 200, 200, 0.9, 0]  # Person
    ]
    
    detections2 = [
        [110, 100, 210, 200, 0.9, 0]  # Same person, moved slightly
    ]
    
    # Track the detections
    tracks1 = tracker.update_tracks(detections1)
    
    # Check that a track was created
    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1
    
    # Update with second frame
    tracks2 = tracker.update_tracks(detections2)
    
    # Check that the track was continued
    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1
    
    # Check that position was updated
    assert tracks2[0].box[0] == 110  # x position updated

def test_calculate_iou():
    """Test the IoU calculation function."""
    processor = VisualProcessor({"device": "cpu"})
    
    # Test cases:
    # 1. No overlap
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert processor._calculate_iou(box1, box2) == 0.0
    
    # 2. Perfect overlap
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert processor._calculate_iou(box1, box2) == 1.0
    
    # 3. Partial overlap
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = processor._calculate_iou(box1, box2)
    assert 0.0 < iou < 1.0
    
    # The exact value should be 25 / 175 = 0.14285...
    assert abs(iou - 0.14285) < 0.001