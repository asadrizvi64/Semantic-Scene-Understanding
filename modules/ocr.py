# modules/ocr.py
"""
OCR Processing Module for Narrative Scene Understanding.
This module handles text detection and recognition in video frames.
"""

import os
import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

class OCRProcessor:
    """
    Processes video frames for text detection and recognition.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OCR processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize OCR engine
        self._init_ocr_engine()
        
        # Track text across frames
        self.text_tracker = {}  # Map text content to tracking info
    
    def _init_ocr_engine(self):
        """Initialize the OCR engine (EasyOCR or Tesseract)."""
        self.logger.info("Initializing OCR engine...")
        
        ocr_engine = self.config.get("ocr_engine", "easyocr")
        
        if ocr_engine == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=self.device.type=="cuda")
                self.logger.info("EasyOCR engine initialized")
            except ImportError:
                self.logger.warning("EasyOCR not available, using Tesseract")
                ocr_engine = "tesseract"
        
        if ocr_engine == "tesseract":
            try:
                import pytesseract
                self.reader = pytesseract
                self.logger.info("Tesseract OCR engine initialized")
            except ImportError:
                self.logger.warning("Neither EasyOCR nor Tesseract available, OCR will be disabled")
                self.reader = None
        
        # OCR processing settings
        self.min_text_size = self.config.get("min_text_size", 10)  # Minimum height in pixels
        self.min_confidence = self.config.get("min_ocr_confidence", 0.5)  # Minimum confidence
        self.process_frequency = self.config.get("ocr_frequency", 5)  # Process every N frames
    
    def process_frames(self, frames: List[Tuple[float, np.ndarray]]) -> List[Dict]:
        """
        Process a list of video frames for text.
        
        Args:
            frames: List of (timestamp, frame) tuples
            
        Returns:
            List of processed OCR data dictionaries
        """
        self.logger.info(f"Processing {len(frames)} frames for OCR...")
        
        if self.reader is None:
            self.logger.warning("OCR engine not available, skipping text detection")
            return [{"timestamp": ts, "text_detections": []} for ts, _ in frames]
        
        results = []
        for frame_idx, (timestamp, frame) in enumerate(frames):
            # Process only every Nth frame to save computation
            if frame_idx % self.process_frequency == 0:
                self.logger.debug(f"Processing OCR for frame {frame_idx} at timestamp {timestamp:.2f}s")
                
                # Process frame
                frame_data = self._process_single_frame(timestamp, frame, frame_idx)
                results.append(frame_data)
            else:
                # For skipped frames, propagate text from previous frame with adjustment
                if results:
                    prev_data = results[-1].copy()
                    prev_data["timestamp"] = timestamp
                    
                    # Adjust positions based on estimated motion
                    for detection in prev_data.get("text_detections", []):
                        # Add small position uncertainty for skipped frames
                        detection["confidence"] = max(0.1, detection.get("confidence", 0.5) - 0.1)
                    
                    results.append(prev_data)
                else:
                    # If no previous frame, create empty data
                    results.append({
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                        "text_detections": []
                    })
            
            # Status update for long videos
            if (frame_idx + 1) % 50 == 0:
                self.logger.info(f"OCR processed {frame_idx + 1}/{len(frames)} frames")
        
        self.logger.info("OCR processing complete")
        return results
    
    def _process_single_frame(self, timestamp: float, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process a single video frame for text.
        
        Args:
            timestamp: Frame timestamp in seconds
            frame: NumPy array containing the frame image
            frame_idx: Index of the frame in the sequence
            
        Returns:
            Dictionary containing OCR analysis data
        """
        # Detect and recognize text
        text_detections = self._detect_text(frame)
        
        # Track text across frames
        text_detections = self._track_text(text_detections, timestamp)
        
        # Create frame data dictionary
        frame_data = {
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "text_detections": text_detections
        }
        
        return frame_data
    
    def _detect_text(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize text in a frame.
        
        Args:
            frame: Frame image
            
        Returns:
            List of text detection dictionaries
        """
        if isinstance(self.reader, type(None)):
            return []
        
        # Depending on the OCR engine
        if hasattr(self.reader, 'readtext'):  # EasyOCR
            return self._detect_with_easyocr(frame)
        else:  # Tesseract
            return self._detect_with_tesseract(frame)
    
    def _detect_with_easyocr(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect text using EasyOCR.
        
        Args:
            frame: Frame image
            
        Returns:
            List of text detection dictionaries
        """
        try:
            # EasyOCR detection
            detections = self.reader.readtext(frame)
            
            text_detections = []
            for detection in detections:
                # Extract information
                box_points = detection[0]  # List of 4 corner points
                text = detection[1]
                confidence = detection[2]
                
                # Skip if confidence is too low
                if confidence < self.min_confidence:
                    continue
                
                # Skip if text is too short (likely noise)
                if len(text.strip()) < 2:
                    continue
                
                # Convert polygon to rectangle
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                # Skip if text is too small
                if (y_max - y_min) < self.min_text_size:
                    continue
                
                # Create detection dictionary
                text_detection = {
                    "text": text,
                    "confidence": float(confidence),
                    "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "polygon": [[int(p[0]), int(p[1])] for p in box_points]
                }
                
                text_detections.append(text_detection)
            
            return text_detections
        
        except Exception as e:
            self.logger.error(f"Error in EasyOCR detection: {e}")
            return []
    
    def _detect_with_tesseract(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect text using Tesseract OCR.
        
        Args:
            frame: Frame image
            
        Returns:
            List of text detection dictionaries
        """
        try:
            # Convert to RGB for Tesseract
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get text and bounding box data
            data = self.reader.image_to_data(frame, output_type=self.reader.Output.DICT)
            
            text_detections = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                # Skip empty text or low confidence
                if not text or confidence < self.min_confidence:
                    continue
                
                # Extract bounding box
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Skip if text is too small
                if h < self.min_text_size:
                    continue
                
                # Create detection dictionary
                text_detection = {
                    "text": text,
                    "confidence": confidence,
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "polygon": [[int(x), int(y)], [int(x + w), int(y)], 
                                [int(x + w), int(y + h)], [int(x), int(y + h)]]
                }
                
                text_detections.append(text_detection)
            
            return text_detections
        
        except Exception as e:
            self.logger.error(f"Error in Tesseract detection: {e}")
            return []
    
    def _track_text(self, detections: List[Dict], timestamp: float) -> List[Dict]:
        """
        Track text across frames to maintain consistency.
        
        Args:
            detections: List of text detections in current frame
            timestamp: Current timestamp
            
        Returns:
            List of text detections with tracking information
        """
        current_texts = {d['text']: d for d in detections}
        
        # Update existing tracked texts
        for text, track_info in list(self.text_tracker.items()):
            if text in current_texts:
                # Update existing track
                detection = current_texts[text]
                
                # Update track info
                track_info['last_seen'] = timestamp
                track_info['occurrences'] += 1
                track_info['last_position'] = detection['box']
                track_info['confidence'] = max(track_info['confidence'], detection['confidence'])
                
                # Add track info to detection
                detection['track_id'] = track_info['track_id']
                detection['first_seen'] = track_info['first_seen']
                detection['occurrences'] = track_info['occurrences']
            else:
                # Text not detected in current frame
                time_since_seen = timestamp - track_info['last_seen']
                
                if time_since_seen > 5.0:  # Remove if not seen for 5 seconds
                    del self.text_tracker[text]
        
        # Add new tracks for new texts
        for text, detection in current_texts.items():
            if text not in self.text_tracker:
                track_id = f"text_{len(self.text_tracker)}"
                
                # Create new track
                self.text_tracker[text] = {
                    'track_id': track_id,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'occurrences': 1,
                    'last_position': detection['box'],
                    'confidence': detection['confidence']
                }
                
                # Add track info to detection
                detection['track_id'] = track_id
                detection['first_seen'] = timestamp
                detection['occurrences'] = 1
        
        return detections