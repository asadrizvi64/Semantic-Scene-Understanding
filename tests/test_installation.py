#!/usr/bin/env python3
"""
Installation test script for Narrative Scene Understanding.
This script verifies that all required components are properly installed.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("installation-test")

def check_core_dependencies():
    """Check if core dependencies are installed."""
    core_packages = [
        "numpy",
        "cv2",  # OpenCV
        "torch",
        "networkx",
        "matplotlib",
        "PIL",  # Pillow
        "tqdm",
        "soundfile",
    ]
    
    missing = []
    for package in core_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} is not installed")
    
    return missing

def check_optional_dependencies():
    """Check if optional dependencies are installed."""
    optional_packages = [
        ("segment_anything", "SAM (Segment Anything Model)"),
        ("transformers", "Transformers (for BLIP-2 and others)"),
        ("deep_sort_realtime", "DeepSORT (for object tracking)"),
        ("insightface", "InsightFace (for face recognition)"),
        ("whisper", "Whisper (for speech transcription)"),
        ("pyannote.audio", "pyannote.audio (for speaker diarization)"),
        ("easyocr", "EasyOCR (for text recognition)"),
        ("pytesseract", "PyTesseract (fallback OCR)"),
        ("openai", "OpenAI API (for LLM)"),
        ("llama_cpp", "llama-cpp-python (for local LLM)"),
        ("pygraphviz", "PyGraphviz (for visualization)"),
    ]
    
    missing = []
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {description} is installed")
        except ImportError:
            missing.append(description)
            logger.warning(f"✗ {description} is not installed")
    
    return missing

def check_model_files():
    """Check if model files are available."""
    # Check common model locations
    model_paths = [
        ("models/sam_vit_h_4b8939.pth", "SAM model"),
        ("models/llama-7b.gguf", "Llama model"),
    ]
    
    missing = []
    for path, description in model_paths:
        if os.path.exists(path):
            logger.info(f"✓ {description} file found at {path}")
        else:
            missing.append(description)
            logger.warning(f"✗ {description} file not found at {path}")
    
    return missing

def check_modules():
    """Check if the package modules are properly installed."""
    module_paths = [
        ("modules.ingestion", "Video ingestion module"),
        ("modules.vision", "Visual processing module"),
        ("modules.audio", "Audio processing module"),
        ("modules.ocr", "OCR processing module"),
        ("modules.knowledge_graph", "Knowledge graph module"),
        ("modules.analysis", "Narrative analysis module"),
        ("modules.query", "Scene query module"),
        ("modules.utils", "Utility functions module"),
    ]
    
    missing = []
    for module_path, description in module_paths:
        try:
            importlib.import_module(module_path)
            logger.info(f"✓ {description} is properly installed")
        except ImportError as e:
            missing.append(description)
            logger.warning(f"✗ {description} is not properly installed: {e}")
    
    return missing

def run_minimal_test():
    """Run a minimal test to verify basic functionality."""
    logger.info("Running minimal functionality test...")
    
    try:
        # Import main function
        from narrative_scene_understanding import process_video
        
        # Create a dummy knowledge graph
        import networkx as nx
        from modules.utils import generate_scene_summary
        
        # Create a simple test graph
        g = nx.DiGraph()
        
        # Add some nodes
        g.add_node("character_1", type="character", name="Character A", description="Main character")
        g.add_node("character_2", type="character", name="Character B", description="Secondary character")
        g.add_node("action_1", type="action", description="walks across the room", timestamp=1.0)
        g.add_node("speech_1", type="speech", text="Hello there!", speaker="character_1", start_time=2.0)
        g.add_node("emotion_1", type="emotion", emotion="surprised", timestamp=3.0)
        
        # Add some edges
        g.add_edge("character_1", "action_1", relation="performs")
        g.add_edge("character_1", "speech_1", relation="speaks")
        g.add_edge("character_2", "emotion_1", relation="feels")
        
        # Create a dummy analysis result
        analysis = {
            "characters": {
                "character_1": {
                    "emotions": [{"emotion": "neutral", "timestamp": 0.0}],
                    "goals": [],
                    "relationships": [],
                    "arc": {"type": "flat"}
                },
                "character_2": {
                    "emotions": [{"emotion": "surprised", "timestamp": 3.0}],
                    "goals": [],
                    "relationships": [],
                    "arc": {"type": "flat"}
                }
            },
            "key_events": [],
            "themes": [],
            "causal_chains": []
        }
        
        # Generate a summary
        summary = generate_scene_summary(g, analysis)
        
        if summary:
            logger.info("✓ Successfully generated scene summary")
            logger.info(f"Summary excerpt: {summary[:100]}...")
        else:
            logger.warning("✗ Failed to generate scene summary")
        
        logger.info("Minimal functionality test completed")
        return True
    
    except Exception as e:
        logger.error(f"✗ Minimal functionality test failed: {e}")
        return False

def main():
    """Run the installation tests."""
    logger.info("=== Narrative Scene Understanding Installation Test ===")
    
    # Check Python version
    py_version = sys.version_info
    logger.info(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        logger.warning("⚠️ Python 3.8 or higher is recommended")
    
    # Check dependencies
    logger.info("\n=== Checking Core Dependencies ===")
    missing_core = check_core_dependencies()
    
    logger.info("\n=== Checking Optional Dependencies ===")
    missing_optional = check_optional_dependencies()
    
    logger.info("\n=== Checking Model Files ===")
    missing_models = check_model_files()
    
    logger.info("\n=== Checking Module Installation ===")
    missing_modules = check_modules()
    
    # Run minimal test
    logger.info("\n=== Running Minimal Functionality Test ===")
    test_success = run_minimal_test()
    
    # Summary
    logger.info("\n=== Installation Test Summary ===")
    
    if not missing_core and not missing_modules and test_success:
        logger.info("✅ Core installation is complete and functional")
    else:
        logger.warning("⚠️ Core installation has issues")
    
    if missing_optional:
        logger.info(f"ℹ️ {len(missing_optional)} optional dependencies are not installed")
        logger.info("   Some features may be limited or unavailable")
    else:
        logger.info("✅ All optional dependencies are installed")
    
    if missing_models:
        logger.info(f"ℹ️ {len(missing_models)} model files were not found")
        logger.info("   You may need to download them using scripts/download_models.py")
    else:
        logger.info("✅ All model files are available")
    
    logger.info("\nFor any missing components, please refer to the installation instructions in the README.")

if __name__ == "__main__":
    main()