# Narrative Scene Understanding Documentation

Welcome to the documentation for the Narrative Scene Understanding system. This documentation will guide you through installation, usage, and advanced features of this multi-modal knowledge graph approach to deep semantic video analysis.

## Overview

Narrative Scene Understanding is a comprehensive system that processes video content to extract deep semantic understanding, including:

- **Character motivations** and emotional states
- **Causal relationships** between events
- **Narrative arcs** and thematic elements
- **Spatial-temporal continuity** across scenes

The system integrates multiple modalities:
- **Visual analysis**: Object segmentation, character tracking, facial recognition
- **Audio analysis**: Speech transcription, speaker diarization, sound classification
- **OCR processing**: Text detection and recognition in scenes
- **Scene descriptions**: Natural language descriptions of visual content

All modalities are unified in a dynamic knowledge graph that enables natural language querying using LLM-powered reasoning.

## Key Features

- **Multi-modal integration** of visual, audio, and textual information
- **Dynamic knowledge graph** that evolves throughout the narrative
- **Character-centric analysis** for tracking individuals' journeys
- **Causal chain inference** for understanding "why" events happen
- **Natural language query interface** with LLM-powered reasoning
- **Cross-domain applicability** for film, security, sports, and personal media

## Documentation Sections

- [Installation Guide](installation.md) - How to install the system and its dependencies
- [Usage Guide](usage.md) - Basic usage instructions and examples
- [API Documentation](api/index.md) - Details on the programmatic interface
- [Advanced Usage](advanced/index.md) - Advanced features and customization

## Quickstart

For a quick introduction to using the system, follow these steps:

1. Install the system:
    ```bash
    git clone https://github.com/yourusername/narrative-scene-understanding.git
    cd narrative-scene-understanding
    sudo bash scripts/install_dependencies.sh
    pip install -e .
    ```
2. Download the required models:
    ```bash
    python scripts/download_models.py
    ```
3. Process a video file:
    ```bash
    narrative-scene path/to/video.mp4
    ```
4. Ask questions about the video:
    ```bash
    narrative-scene path/to/video.mp4 --query "Why did Character A appear surprised?"
    ```

See the [Usage Guide](usage.md) for more details.

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux (recommended), macOS, or Windows
- **Hardware**:
  - **CPU**: 4+ cores recommended
  - **RAM**: 16GB+ recommended
  - **GPU**: NVIDIA GPU with CUDA support recommended for optimal performance
  - **Storage**: 10GB+ for models and dependencies

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
