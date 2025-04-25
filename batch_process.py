#!/usr/bin/env python3
"""
Batch video processing script for Narrative Scene Understanding.
This script processes multiple video files and generates analysis for each.
"""

import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import core modules
from modules.utils import setup_logging
from narrative_scene_understanding import process_video

def find_video_files(directory: str) -> List[str]:
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def process_videos(video_files: List[str], output_dir: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process multiple video files.
    
    Args:
        video_files: List of video file paths
        output_dir: Directory to save outputs
        config: Optional configuration overrides
        
    Returns:
        Dictionary of results keyed by video path
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video
    results = {}
    for i, video_path in enumerate(video_files):
        try:
            logger.info(f"Processing video {i+1}/{len(video_files)}: {video_path}")
            
            # Generate output subdirectory based on video name
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Update config with video-specific output directory
            video_config = config.copy() if config else {}
            video_config['output_dir'] = video_output_dir
            
            # Process video
            start_time = time.time()
            video_results = process_video(video_path, video_config)
            process_time = time.time() - start_time
            
            # Add processing time and save summary
            video_results['process_time'] = process_time
            
            with open(os.path.join(video_output_dir, 'summary.txt'), 'w') as f:
                f.write(video_results['summary'])
            
            # Save results metadata (exclude large objects)
            metadata = {
                'video_path': video_path,
                'output_dir': video_output_dir,
                'graph_path': video_results.get('graph_path'),
                'summary_path': video_results.get('summary_path'),
                'process_time': process_time,
                'processed_at': datetime.now().isoformat(),
                'frame_count': len(video_results.get('visual_data', [])),
                'speech_segments': len(video_results.get('audio_data', [])),
            }
            
            with open(os.path.join(video_output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Store in results dict
            results[video_path] = metadata
            
            logger.info(f"Completed processing {video_path} in {process_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            results[video_path] = {'error': str(e)}
    
    # Generate batch summary
    batch_summary = {
        'total_videos': len(video_files),
        'successful': sum(1 for r in results.values() if 'error' not in r),
        'failed': sum(1 for r in results.values() if 'error' in r),
        'total_process_time': sum(r.get('process_time', 0) for r in results.values() if 'process_time' in r),
        'processed_at': datetime.now().isoformat(),
        'videos': results
    }
    
    # Save batch summary
    with open(os.path.join(output_dir, 'batch_summary.json'), 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Batch Video Processing for Narrative Scene Understanding")
    parser.add_argument("--directory", "-d", required=True, help="Directory containing video files")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--config", "-c", help="Path to configuration JSON file")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search directory recursively")
    parser.add_argument("--pattern", "-p", help="File pattern to match (e.g., '*_raw.mp4')")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of videos to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Find video files
    logger.info(f"Searching for video files in {args.directory}")
    video_files = find_video_files(args.directory)
    
    # Apply pattern filter if provided
    if args.pattern:
        import fnmatch
        video_files = [v for v in video_files if fnmatch.fnmatch(os.path.basename(v), args.pattern)]
    
    # Apply limit if provided
    if args.limit and args.limit > 0:
        video_files = video_files[:args.limit]
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process videos
    if video_files:
        results = process_videos(video_files, args.output, config)
        logger.info(f"Batch processing complete. Processed {len(results)} videos.")
        
        # Print summary
        successful = sum(1 for r in results.values() if 'error' not in r)
        failed = sum(1 for r in results.values() if 'error' in r)
        
        logger.info(f"Summary: {successful} successful, {failed} failed")
        if failed > 0:
            logger.info("Failed videos:")
            for video_path, result in results.items():
                if 'error' in result:
                    logger.info(f"  - {video_path}: {result['error']}")
    else:
        logger.warning("No video files found to process")

if __name__ == "__main__":
    main()