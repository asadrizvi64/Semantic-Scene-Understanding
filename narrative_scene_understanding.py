# narrative_scene_understanding.py
"""
Example implementation of the Narrative Scene Understanding pipeline.
This script shows how to use the pipeline for analyzing a video file
and answering natural language queries about its content.
"""

import os
import argparse
import logging
import torch
from pathlib import Path
import cv2
import numpy as np
import networkx as nx
import subprocess
from datetime import datetime
from tqdm import tqdm
import json

# Import core modules
from modules.ingestion import VideoPreprocessor
from modules.vision import VisualProcessor
from modules.audio import AudioProcessor
from modules.ocr import OCRProcessor
from modules.knowledge_graph import NarrativeGraphBuilder
from modules.analysis import NarrativeAnalyzer
from modules.query import SceneQueryEngine
from modules.utils import setup_logging, generate_scene_summary

# Global configuration
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'output',
    'frame_rate': 2,  # Extract 2 frames per second by default
    'adaptive_sampling': True,  # Use adaptive sampling based on motion
    'max_frames': 500,  # Limit total frames for memory reasons
    'model_paths': {
        'sam': 'models/sam_vit_h_4b8939.pth',
        'whisper': 'base',  # Whisper model size
        'face_recognition': 'models/insightface_model',
        'ocr': 'models/easyocr_model',
    }
}

def process_video(video_path, config=None):
    """
    Process a video file through the complete pipeline.
    
    Args:
        video_path: Path to the video file
        config: Optional configuration overrides
        
    Returns:
        Dict containing narrative graph, summary, and results
    """
    if config:
        # Update global config with any overrides
        CONFIG.update(config)
    
    logging.info(f"Processing video: {video_path}")
    logging.info(f"Using device: {CONFIG['device']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Step 1: Data Ingestion
    logging.info("Step 1: Data Ingestion")
    preprocessor = VideoPreprocessor(CONFIG)
    frames, scene_boundaries, audio_path = preprocessor.process(video_path)
    
    logging.info(f"Extracted {len(frames)} frames, {len(scene_boundaries)} scene boundaries")
    
    # Step 2: Multi-modal Processing
    logging.info("Step 2: Visual Processing")
    visual_processor = VisualProcessor(CONFIG)
    visual_data = visual_processor.process_frames(frames)
    
    logging.info("Step 3: Audio Processing")
    audio_processor = AudioProcessor(CONFIG)
    audio_data = audio_processor.process_audio(audio_path)
    
    logging.info("Step 4: OCR Processing")
    ocr_processor = OCRProcessor(CONFIG)
    ocr_data = ocr_processor.process_frames(frames)
    
    # Step 5: Knowledge Graph Construction
    logging.info("Step 5: Narrative Graph Construction")
    graph_builder = NarrativeGraphBuilder(CONFIG)
    narrative_graph = graph_builder.build_graph(
        visual_data=visual_data,
        audio_data=audio_data,
        ocr_data=ocr_data,
        scene_boundaries=scene_boundaries
    )
    
    # Step 6: Narrative Analysis
    logging.info("Step 6: Narrative Analysis")
    analyzer = NarrativeAnalyzer(CONFIG)
    analysis_results = analyzer.analyze(narrative_graph)
    
    # Generate scene summary
    scene_summary = generate_scene_summary(narrative_graph, analysis_results)
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(CONFIG['output_dir'], f"narrative_graph_{timestamp}.json")
    
    logging.info(f"Saving narrative graph to {graph_path}")
    
    # Convert NetworkX graph to a serializable format
    graph_data = {
        'nodes': [
            {'id': node, **attrs} for node, attrs in narrative_graph.nodes(data=True)
        ],
        'edges': [
            {'source': u, 'target': v, **attrs} for u, v, attrs in narrative_graph.edges(data=True)
        ]
    }
    
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(CONFIG['output_dir'], f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(scene_summary)
    
    return {
        'graph': narrative_graph,
        'summary': scene_summary,
        'analysis': analysis_results,
        'graph_path': graph_path,
        'summary_path': summary_path
    }

def query_scene(question, results=None, video_path=None, config=None):
    """
    Query a processed scene with natural language.
    
    Args:
        question: Natural language question about the scene
        results: Optional results from previous processing
        video_path: Path to video file (if results not provided)
        config: Optional configuration overrides
        
    Returns:
        Dict with answer, reasoning, and supporting evidence
    """
    if results is None:
        if video_path is None:
            raise ValueError("Either results or video_path must be provided")
        results = process_video(video_path, config)
    
    logging.info(f"Querying scene: '{question}'")
    
    # Initialize the query engine
    query_engine = SceneQueryEngine(
        model_name="llama-7b-chat",  # Or other model as needed
        temperature=0.2
    )
    
    # Process the query
    response = query_engine.query_scene(
        question=question,
        narrative_graph=results['graph']
    )
    
    # Log the response
    logging.info(f"Answer: {response['answer']}")
    
    return response

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Narrative Scene Understanding Pipeline")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--query", "-q", help="Optional query about the scene")
    parser.add_argument("--frame_rate", type=float, default=2.0, help="Frame sampling rate")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Processing device")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Update config based on arguments
    config_overrides = {
        'output_dir': args.output,
        'frame_rate': args.frame_rate
    }
    
    if args.device:
        config_overrides['device'] = args.device
    
    # Process the video
    results = process_video(args.video_path, config_overrides)
    print(f"\nScene Summary:\n{results['summary']}\n")
    
    # Process query if provided
    if args.query:
        response = query_scene(args.query, results)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {response['answer']}")
        
        # Optionally show reasoning
        if args.debug:
            print(f"\nReasoning Process:\n{response['reasoning']}")

if __name__ == "__main__":
    main()