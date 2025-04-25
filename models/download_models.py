#!/usr/bin/env python3
"""
Script to download required models for Narrative Scene Understanding.
"""

import os
import sys
import argparse
import logging
import requests
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model-downloader")

# Default models directory
DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

# Model URLs and information
SAM_MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size": 2564217609,  # ~2.4 GB
        "filename": "sam_vit_h_4b8939.pth",
        "description": "SAM ViT-H (Highest quality)"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size": 1249446430,  # ~1.2 GB
        "filename": "sam_vit_l_0b3195.pth",
        "description": "SAM ViT-L (Balanced quality and speed)"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size": 375042949,   # ~375 MB
        "filename": "sam_vit_b_01ec64.pth",
        "description": "SAM ViT-B (Fastest)"
    }
}

YOLO_MODELS = {
    "yolov5s": {
        "description": "YOLOv5 Small (Fallback for SAM)",
        "note": "Downloaded on-demand via torch.hub"
    }
}

WHISPER_MODELS = {
    "base": {
        "description": "Whisper Base (Balance of accuracy and speed)",
        "note": "Downloaded on-demand by whisper library"
    },
    "small": {
        "description": "Whisper Small (Higher accuracy)",
        "note": "Downloaded on-demand by whisper library"
    },
    "tiny": {
        "description": "Whisper Tiny (Fastest)",
        "note": "Downloaded on-demand by whisper library"
    }
}

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

def download_sam_model(model_type, models_dir):
    """
    Download SAM model.
    
    Args:
        model_type: Type of SAM model to download (vit_h, vit_l, or vit_b)
        models_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    if model_type not in SAM_MODELS:
        logger.error(f"Unknown SAM model type: {model_type}")
        return False
    
    model_info = SAM_MODELS[model_type]
    filename = os.path.join(models_dir, model_info["filename"])
    
    logger.info(f"Downloading SAM {model_type} model: {model_info['description']}")
    return download_file(model_info["url"], filename, model_info["size"])

def download_yolo_model(models_dir):
    """
    Download YOLOv5 model (on-demand).
    
    Args:
        models_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Testing YOLOv5 model download (will be cached by torch.hub)")
        # This will download and cache the model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        logger.info("YOLOv5 model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {e}")
        return False

def download_insightface_model(models_dir):
    """
    Download InsightFace model.
    
    Args:
        models_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Call the dedicated script for InsightFace download
        insightface_script = os.path.join(os.path.dirname(__file__), "download_insightface.py")
        
        if not os.path.exists(insightface_script):
            logger.error(f"InsightFace download script not found: {insightface_script}")
            return False
        
        logger.info("Running InsightFace download script")
        result = subprocess.run(
            [sys.executable, insightface_script, "--output_dir", models_dir],
            check=True
        )
        
        if result.returncode == 0:
            logger.info("InsightFace model downloaded successfully")
            return True
        else:
            logger.error("InsightFace download script failed")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading InsightFace model: {e}")
        return False

def download_whisper_model(model_type):
    """
    Download Whisper model (on-demand).
    
    Args:
        model_type: Type of Whisper model to download (base, small, or tiny)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import whisper
        logger.info(f"Testing Whisper {model_type} model download")
        # This will download and cache the model
        model = whisper.load_model(model_type)
        logger.info(f"Whisper {model_type} model loaded successfully")
        return True
    except ImportError:
        logger.warning("Whisper package not installed. Install with: pip install whisper")
        return False
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return False

def download_llama_models(models_dir):
    """
    Inform about Llama model download process.
    
    Args:
        models_dir: Directory to save the model
    """
    logger.info("\nLlama Model Download Information:")
    logger.info("===================================")
    logger.info("Llama models require manual download due to license restrictions.")
    logger.info("To use Llama models:")
    logger.info("1. Request access at: https://ai.meta.com/llama/")
    logger.info("2. Download the model weights")
    logger.info("3. Convert to GGUF format using llama.cpp")
    logger.info("4. Place the resulting GGUF file in the models/ directory")
    logger.info("")
    logger.info("You can use the convert_models.py script after downloading:")
    logger.info(f"python scripts/convert_models.py --input /path/to/downloaded/model --output {models_dir}/llama-7b.gguf")
    logger.info("")
    logger.info("Alternatively, you can configure the system to use OpenAI's API instead.")

def main():
    parser = argparse.ArgumentParser(description="Download models for Narrative Scene Understanding")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Directory to save models")
    parser.add_argument("--sam_model", choices=["vit_h", "vit_l", "vit_b"], default="vit_h", help="SAM model variant to download")
    parser.add_argument("--whisper_model", choices=["tiny", "base", "small"], default="base", help="Whisper model variant to test")
    parser.add_argument("--skip_sam", action="store_true", help="Skip SAM model download")
    parser.add_argument("--skip_yolo", action="store_true", help="Skip YOLOv5 model test download")
    parser.add_argument("--skip_insightface", action="store_true", help="Skip InsightFace model download")
    parser.add_argument("--skip_whisper", action="store_true", help="Skip Whisper model test")
    parser.add_argument("--skip_llama_info", action="store_true", help="Skip Llama model information")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Print welcome message
    logger.info("=== Narrative Scene Understanding Model Downloader ===")
    logger.info(f"Models will be saved to: {args.models_dir}")
    
    # Download or check models
    success = True
    
    if not args.skip_sam:
        if not download_sam_model(args.sam_model, args.models_dir):
            success = False
    
    if not args.skip_yolo:
        if not download_yolo_model(args.models_dir):
            success = False
    
    if not args.skip_insightface:
        if not download_insightface_model(args.models_dir):
            success = False
    
    if not args.skip_whisper:
        if not download_whisper_model(args.whisper_model):
            success = False
    
    if not args.skip_llama_info:
        download_llama_models(args.models_dir)
    
    # Final status
    if success:
        logger.info("\n✅ All model downloads completed successfully!")
    else:
        logger.warning("\n⚠️ Some model downloads failed. Check the logs for details.")
    
    logger.info("\nNext Steps:")
    logger.info("1. Install the required Python packages: pip install -r requirements.txt")
    logger.info("2. Configure your system in configs/default.json")
    logger.info("3. Run the system: python narrative_scene_understanding.py <video_file>")

if __name__ == "__main__":
    main()