#!/usr/bin/env python3
"""
Script to download InsightFace models for Narrative Scene Understanding.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import tempfile
import zipfile
import shutil
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("insightface-downloader")

# Default models directory
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/insightface_model"))

# InsightFace model URL - using buffalo_l model which is good for face recognition
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MODEL_SIZE = 134021266  # ~128 MB

def download_file(url, filename, expected_size=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        filename: Path to save the file
        expected_size: Expected file size in bytes (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Check if file already exists with the expected size
        if os.path.exists(filename):
            if expected_size and os.path.getsize(filename) == expected_size:
                logger.info(f"File already exists and has the correct size: {filename}")
                return True
            else:
                logger.info(f"File exists but may be incomplete: {filename}")
                # Rename the existing file as backup
                backup_name = f"{filename}.bak"
                shutil.move(filename, backup_name)
                logger.info(f"Moved existing file to {backup_name}")
        
        # Start the download
        logger.info(f"Downloading {url} to {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get content length if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize the progress bar
        progress_bar = tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            desc=filename
        )
        
        # Download the file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # Verify file size if expected size is provided
        if expected_size and os.path.getsize(filename) != expected_size:
            logger.warning(f"Downloaded file size doesn't match expected size: {filename}")
            return False
        
        logger.info(f"Download completed: {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_path):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_path: Path to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_path}")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(info.file_size for info in zip_ref.infolist())
            
            # Extract with progress bar
            extracted_size = 0
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_path)
                    extracted_size += file.file_size
                    pbar.update(file.file_size)
        
        logger.info(f"Extraction completed: {extract_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def verify_insightface_model(model_dir):
    """
    Verify that the InsightFace model was downloaded and extracted correctly.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        True if the model files exist, False otherwise
    """
    expected_files = [
        "1k5d_final.onnx",    # Face recognition model
        "2d106det.onnx",      # Facial landmark detection model
        "det_10g.onnx",       # Face detection model
        "genderage.onnx",     # Gender and age estimation model
        "w600k_r50.onnx"      # Face recognition feature extraction model
    ]
    
    for file in expected_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing model file: {file_path}")
            return False
    
    logger.info("All InsightFace model files verified successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download InsightFace models for Narrative Scene Understanding")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save models")
    
    args = parser.parse_args()
    output_dir = args.output_dir
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download zip file
        zip_path = os.path.join(temp_dir, "insightface_model.zip")
        if not download_file(MODEL_URL, zip_path, MODEL_SIZE):
            logger.error("Failed to download InsightFace model")
            sys.exit(1)
        
        # Extract zip file
        if not extract_zip(zip_path, output_dir):
            logger.error("Failed to extract InsightFace model")
            sys.exit(1)
        
        # Verify model files
        if not verify_insightface_model(output_dir):
            logger.error("InsightFace model files are incomplete")
            sys.exit(1)
        
        logger.info(f"InsightFace model downloaded and extracted to {output_dir}")
        logger.info("Done!")

if __name__ == "__main__":
    main()