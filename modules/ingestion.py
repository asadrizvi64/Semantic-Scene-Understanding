# modules/ingestion.py
"""
Video Preprocessing Module for Narrative Scene Understanding.
This module handles frame extraction, scene boundary detection, and audio separation.
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path
import ffmpeg
from typing import Dict, List, Tuple, Any, Optional

class VideoPreprocessor:
    """
    Preprocesses video files for narrative scene understanding.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Import TransNetV2 for scene boundary detection if available
        try:
            import transnetv2
            self.scene_detector = transnetv2.TransNetV2()
        except ImportError:
            self.logger.warning("TransNetV2 not available. Using fallback scene detection.")
            self.scene_detector = None
    
    def process(self, video_path: str) -> Tuple[List[Tuple[float, np.ndarray]], List[float], str]:
        """
        Process a video file to extract frames, detect scene boundaries, and separate audio.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames, scene_boundaries, audio_path)
            - frames: List of (timestamp, frame) tuples
            - scene_boundaries: List of scene boundary timestamps
            - audio_path: Path to the extracted audio file
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        self.logger.info(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        # Extract frames
        frames = self._extract_frames(video_path, fps)
        
        # Detect scene boundaries
        scene_boundaries = self._detect_scene_boundaries(video_path, fps)
        
        # Extract audio
        audio_path = self._extract_audio(video_path)
        
        return frames, scene_boundaries, audio_path
    
    def _extract_frames(self, video_path: str, fps: float) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from the video.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second of the video
            
        Returns:
            List of (timestamp, frame) tuples
        """
        self.logger.info("Extracting frames...")
        
        # Get target frame rate for extraction
        target_fps = self.config.get("frame_rate", 2.0)
        adaptive_sampling = self.config.get("adaptive_sampling", True)
        max_frames = self.config.get("max_frames", 500)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame sampling parameters
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps if video_fps > 0 else 0
        
        # If video is too long, adjust target_fps to stay under max_frames
        estimated_frames = duration * target_fps
        if estimated_frames > max_frames:
            target_fps = max_frames / duration
            self.logger.info(f"Adjusting frame rate to {target_fps:.2f} FPS to stay under {max_frames} frames")
        
        # Calculate frame sampling interval
        if adaptive_sampling:
            # Will calculate dynamically based on motion
            base_interval = int(video_fps / target_fps)
        else:
            # Fixed interval
            interval = int(video_fps / target_fps)
        
        frames = []
        frame_idx = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / video_fps
            
            if adaptive_sampling and prev_frame is not None:
                # Calculate motion between frames
                motion = self._calculate_motion(prev_frame, frame)
                
                # Adjust interval based on motion (more motion = more frames)
                interval = max(1, int(base_interval * (1.0 - motion * 0.5)))
            else:
                interval = base_interval if adaptive_sampling else interval
            
            # Extract frame if it's time
            if frame_idx % interval == 0:
                frames.append((timestamp, frame))
                prev_frame = frame.copy()
            
            frame_idx += 1
            
            # Status update for long videos
            if frame_idx % 1000 == 0:
                self.logger.debug(f"Processed {frame_idx}/{frame_count} frames...")
        
        cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Calculate the amount of motion between two frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            Motion value between 0 and 1
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate motion magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = np.mean(mag) / 10.0  # Normalize
        
        return min(1.0, motion)
    
    def _detect_scene_boundaries(self, video_path: str, fps: float) -> List[float]:
        """
        Detect scene boundaries in the video.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second of the video
            
        Returns:
            List of scene boundary timestamps
        """
        self.logger.info("Detecting scene boundaries...")
        
        if self.scene_detector:
            # Use TransNetV2 for scene detection
            video_frames, single_frame_predictions, all_frame_predictions = \
                self.scene_detector.process_video(video_path)
            
            scenes = self.scene_detector.predictions_to_scenes(
                single_frame_predictions, threshold=0.5)
            
            # Convert frame indices to timestamps
            scene_boundaries = [scene[0] / fps for scene in scenes]
        else:
            # Fallback: simple shot detection based on frame differences
            cap = cv2.VideoCapture(video_path)
            prev_frame = None
            scene_boundaries = [0.0]  # Always include the start
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Calculate frame difference
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    diff = cv2.absdiff(gray1, gray2)
                    non_zero_count = np.count_nonzero(diff > 30)
                    
                    # If difference is large enough, mark as scene boundary
                    if non_zero_count > (diff.size * 0.2):  # 20% threshold
                        timestamp = frame_idx / fps
                        scene_boundaries.append(timestamp)
                
                prev_frame = frame.copy()
                frame_idx += 1
            
            cap.release()
        
        self.logger.info(f"Detected {len(scene_boundaries)} scene boundaries")
        return scene_boundaries
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        self.logger.info("Extracting audio...")
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output audio path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
        
        try:
            # Use ffmpeg-python
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
        except Exception as e:
            self.logger.warning(f"Error extracting audio with ffmpeg-python: {e}")
            
            # Fallback to subprocess
            try:
                cmd = [
                    "ffmpeg", "-i", video_path, 
                    "-vn", "-acodec", "pcm_s16le", 
                    "-ac", "1", "-ar", "16000",
                    "-y", audio_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                self.logger.error(f"Error extracting audio with subprocess: {e}")
                # Create empty audio file
                with open(audio_path, 'wb') as f:
                    f.write(b'')
        
        self.logger.info(f"Audio extracted to {audio_path}")
        return audio_path