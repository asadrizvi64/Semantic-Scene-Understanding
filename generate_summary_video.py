#!/usr/bin/env python3
"""
Generate Summary Video with annotations based on Narrative Scene Understanding analysis.
"""

import os
import sys
import argparse
import logging
import json
import tempfile
import subprocess
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import utility functions
from modules.utils import setup_logging, load_graph_from_json

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate Summary Video with Annotations")
    parser.add_argument("video_path", help="Path to the original video file")
    parser.add_argument("--graph", "-g", required=True, help="Path to the narrative graph JSON file")
    parser.add_argument("--output", "-o", default="summary_video.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=720, help="Output video height")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--highlights", action="store_true", help="Include only highlight segments")
    parser.add_argument("--speedup", type=float, default=1.0, help="Speed up factor for non-highlight segments")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input video
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Load narrative graph
    try:
        graph = load_graph_from_json(args.graph)
        logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        return
    
    # Generate summary video
    try:
        generate_summary_video(args.video_path, graph, args.output, args)
    except Exception as e:
        logger.error(f"Error generating summary video: {e}")
        return

def generate_summary_video(video_path: str, graph: Any, output_path: str, args):
    """
    Generate a summary video with annotations based on analysis.
    
    Args:
        video_path: Path to the original video
        graph: Narrative knowledge graph
        output_path: Path to save the summary video
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating summary video from {video_path}")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output dimensions (maintain aspect ratio)
    target_height = args.height
    target_width = int(original_width * target_height / original_height)
    
    logger.info(f"Input video: {original_width}x{original_height} @ {original_fps} FPS, {frame_count} frames")
    logger.info(f"Output video: {target_width}x{target_height} @ {args.fps} FPS")
    
    # Extract key moments from analysis
    key_moments = extract_key_moments(graph)
    logger.info(f"Extracted {len(key_moments)} key moments")
    
    # Create highlight segments
    highlight_segments = []
    
    for moment in key_moments:
        # Create segment window around each key moment
        timestamp = moment["timestamp"]
        start_time = max(0, timestamp - 2.0)  # 2 seconds before moment
        end_time = timestamp + 3.0            # 3 seconds after moment
        
        highlight_segments.append({
            "start": start_time,
            "end": end_time,
            "moment": moment
        })
    
    # Sort and merge overlapping segments
    highlight_segments.sort(key=lambda x: x["start"])
    merged_segments = merge_overlapping_segments(highlight_segments)
    
    logger.info(f"Created {len(merged_segments)} highlight segments")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, args.fps, (target_width, target_height))
    
    # Initialize variables
    frame_idx = 0
    processed_frames = 0
    
    # Process segments
    if args.highlights:
        # Process only highlight segments
        for segment in merged_segments:
            process_segment(cap, out, segment, original_fps, args.fps, target_width, target_height)
            processed_frames += int((segment["end"] - segment["start"]) * args.fps)
    else:
        # Process full video with speedup for non-highlight parts
        current_time = 0.0
        
        while frame_idx < frame_count:
            # Check if current time is in a highlight segment
            in_highlight = False
            current_segment = None
            
            for segment in merged_segments:
                if segment["start"] <= current_time <= segment["end"]:
                    in_highlight = True
                    current_segment = segment
                    break
            
            if in_highlight:
                # Process highlight segment at normal speed
                segment_start_frame = int(current_segment["start"] * original_fps)
                segment_end_frame = int(current_segment["end"] * original_fps)
                
                # Seek to start frame if needed
                if frame_idx < segment_start_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)
                    frame_idx = segment_start_frame
                
                # Process frames in segment
                while frame_idx < segment_end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Add annotations
                    annotated_frame = add_annotations(resized_frame, current_segment, current_time)
                    
                    # Write frame
                    out.write(annotated_frame)
                    
                    frame_idx += 1
                    processed_frames += 1
                    current_time = frame_idx / original_fps
                    
                    # Status update
                    if processed_frames % 100 == 0:
                        logger.debug(f"Processed {processed_frames} frames")
            else:
                # Process non-highlight segment with speedup
                # Find next highlight segment
                next_segment = None
                for segment in merged_segments:
                    if segment["start"] > current_time:
                        next_segment = segment
                        break
                
                # Determine end of non-highlight segment
                if next_segment:
                    end_time = next_segment["start"]
                else:
                    end_time = frame_count / original_fps
                
                # Apply speedup
                if args.speedup > 1.0:
                    # Skip frames based on speedup factor
                    skip_frames = int(args.speedup - 1.0)
                    
                    # Process segment with frame skipping
                    segment_start_frame = frame_idx
                    segment_end_frame = int(end_time * original_fps)
                    
                    while frame_idx < segment_end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Only keep 1 out of every (skip_frames + 1) frames
                        if (frame_idx - segment_start_frame) % (skip_frames + 1) == 0:
                            # Resize frame
                            resized_frame = cv2.resize(frame, (target_width, target_height))
                            
                            # Write frame
                            out.write(resized_frame)
                            processed_frames += 1
                        
                        frame_idx += 1
                        current_time = frame_idx / original_fps
                        
                        # Status update
                        if processed_frames % 100 == 0:
                            logger.debug(f"Processed {processed_frames} frames")
                else:
                    # No speedup, process normally
                    segment_end_frame = int(end_time * original_fps)
                    
                    while frame_idx < segment_end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize frame
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                        
                        # Write frame
                        out.write(resized_frame)
                        
                        frame_idx += 1
                        processed_frames += 1
                        current_time = frame_idx / original_fps
                        
                        # Status update
                        if processed_frames % 100 == 0:
                            logger.debug(f"Processed {processed_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    logger.info(f"Generated summary video with {processed_frames} frames saved to {output_path}")

def extract_key_moments(graph: Any) -> List[Dict[str, Any]]:
    """
    Extract key moments from the narrative graph.
    
    Args:
        graph: Narrative knowledge graph
        
    Returns:
        List of key moments with timestamps and descriptions
    """
    key_moments = []
    
    # Look for key moments:
    # 1. Significant character emotions
    # 2. Important actions
    # 3. Causal chains
    # 4. Character interactions
    
    # Find significant emotions
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") == "emotion" and attrs.get("intensity", 0.0) > 0.7:
            # Get timestamp
            timestamp = attrs.get("timestamp", 0.0)
            
            # Get character
            character = None
            for pred in graph.predecessors(node_id):
                if graph.nodes[pred].get("type") == "character":
                    character = pred
                    break
            
            if character:
                char_name = graph.nodes[character].get("name", character)
                emotion = attrs.get("emotion", "unknown")
                
                key_moments.append({
                    "type": "emotion",
                    "timestamp": timestamp,
                    "character": character,
                    "character_name": char_name,
                    "emotion": emotion,
                    "description": f"{char_name} feels {emotion}"
                })
    
    # Find important actions
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") == "action":
            # Check if this action is part of a causal chain
            is_causal = False
            for _, target, edge_attrs in graph.out_edges(node_id, data=True):
                if edge_attrs.get("relation") == "causes":
                    is_causal = True
                    break
            
            if is_causal:
                # Get timestamp
                timestamp = attrs.get("timestamp", 0.0)
                
                # Get character
                character = None
                for pred in graph.predecessors(node_id):
                    if graph.nodes[pred].get("type") == "character":
                        character = pred
                        break
                
                description = attrs.get("description", "unknown action")
                
                if character:
                    char_name = graph.nodes[character].get("name", character)
                    key_moments.append({
                        "type": "action",
                        "timestamp": timestamp,
                        "character": character,
                        "character_name": char_name,
                        "description": f"{char_name} {description}"
                    })
                else:
                    key_moments.append({
                        "type": "action",
                        "timestamp": timestamp,
                        "description": description
                    })
    
    # Find character interactions (spatial relationships)
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("relation") == "spatial":
            # Check if both nodes are characters
            if (graph.nodes[source].get("type") == "character" and 
                graph.nodes[target].get("type") == "character"):
                
                # Get spatial relationship
                spatial = attrs.get("spatial", "near")
                timestamp = attrs.get("first_seen", 0.0)
                
                source_name = graph.nodes[source].get("name", source)
                target_name = graph.nodes[target].get("name", target)
                
                key_moments.append({
                    "type": "interaction",
                    "timestamp": timestamp,
                    "source_character": source,
                    "source_name": source_name,
                    "target_character": target,
                    "target_name": target_name,
                    "spatial": spatial,
                    "description": f"{source_name} is {spatial} {target_name}"
                })
    
    # Sort by timestamp
    key_moments.sort(key=lambda x: x["timestamp"])
    
    return key_moments

def merge_overlapping_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge overlapping time segments.
    
    Args:
        segments: List of segments with start and end times
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    
    merged = [sorted_segments[0]]
    
    for segment in sorted_segments[1:]:
        current = merged[-1]
        
        # Check if current segment overlaps with the last merged segment
        if segment["start"] <= current["end"]:
            # Merge by extending the end time and combining moments
            current["end"] = max(current["end"], segment["end"])
            
            # Merge moment info if present
            if "moment" in segment and "moment" in current:
                # Keep both moments as a list
                if not isinstance(current["moment"], list):
                    current["moment"] = [current["moment"]]
                current["moment"].append(segment["moment"])
            elif "moment" in segment:
                current["moment"] = segment["moment"]
        else:
            # No overlap, add as a new segment
            merged.append(segment)
    
    return merged

def process_segment(cap: cv2.VideoCapture, out: cv2.VideoWriter, segment: Dict[str, Any], 
                   original_fps: float, target_fps: float, target_width: int, target_height: int):
    """
    Process a video segment with annotations.
    
    Args:
        cap: OpenCV VideoCapture object
        out: OpenCV VideoWriter object
        segment: Segment information
        original_fps: Original video FPS
        target_fps: Target output FPS
        target_width: Target frame width
        target_height: Target frame height
    """
    # Calculate frame indices
    start_frame = int(segment["start"] * original_fps)
    end_frame = int(segment["end"] * original_fps)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current time
        current_time = current_frame / original_fps
        
        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # Add annotations
        annotated_frame = add_annotations(resized_frame, segment, current_time)
        
        # Write frame
        out.write(annotated_frame)
        
        current_frame += 1

def add_annotations(frame: np.ndarray, segment: Dict[str, Any], current_time: float) -> np.ndarray:
    """
    Add annotations to a video frame based on analysis.
    
    Args:
        frame: Input frame
        segment: Segment information
        current_time: Current timestamp
        
    Returns:
        Annotated frame
    """
    # Create a copy of the frame to avoid modifying the original
    annotated = frame.copy()
    
    # Get frame dimensions
    height, width = annotated.shape[:2]
    
    # Draw semi-transparent overlay at the top for text
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # Add timestamp
    cv2.putText(annotated, f"Time: {current_time:.2f}s", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add moment description(s)
    if "moment" in segment:
        moments = segment["moment"] if isinstance(segment["moment"], list) else [segment["moment"]]
        
        for i, moment in enumerate(moments):
            if i >= 2:  # Limit to 2 descriptions to avoid clutter
                break
                
            description = moment.get("description", "")
            y_pos = 60 + i * 30
            
            # Choose color based on moment type
            if moment.get("type") == "emotion":
                color = (255, 255, 0)  # Yellow
            elif moment.get("type") == "action":
                color = (0, 255, 255)  # Cyan
            else:
                color = (255, 255, 255)  # White
            
            cv2.putText(annotated, description, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add bounding boxes for characters if available
    if "moment" in segment:
        moments = segment["moment"] if isinstance(segment["moment"], list) else [segment["moment"]]
        
        for moment in moments:
            if "character" in moment and abs(moment["timestamp"] - current_time) < 0.5:
                # In a full implementation, this would use character positions from the frame analysis
                # Here we'll just demonstrate with placeholder boxes
                
                # Draw a placeholder box in the center
                box_width = width // 3
                box_height = height // 3
                box_x = (width - box_width) // 2
                box_y = (height - box_height) // 2
                
                # Draw box
                cv2.rectangle(annotated, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                             (0, 255, 0), 2)
                
                # Add character name
                char_name = moment.get("character_name", "Character")
                cv2.putText(annotated, char_name, (box_x, box_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add emotion if available
                if moment.get("type") == "emotion":
                    emotion = moment.get("emotion", "")
                    cv2.putText(annotated, emotion, (box_x, box_y + box_height + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return annotated

if __name__ == "__main__":
    main()