# Narrative Scene Understanding

A comprehensive system for deep semantic understanding of visual narratives through multi-modal analysis and knowledge graph construction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Narrative Scene Understanding processes video content to extract deep semantic understanding, including:

- **Character motivations** and emotional states
- **Causal relationships** between events
- **Narrative arcs** and thematic elements
- **Spatial-temporal continuity** across scenes

The pipeline integrates multiple modalities:
- **Visual analysis**: Object segmentation, character tracking, facial recognition
- **Audio analysis**: Speech transcription, speaker diarization, sound classification
- **OCR processing**: Text detection and recognition in scenes
- **Scene descriptions**: Natural language descriptions of visual content

All modalities are unified in a dynamic knowledge graph that enables natural language querying using LLM-powered reasoning.

## Quick Start: Running Inference on a Video

### Command Line Usage

Process a video file:

```bash
# Basic processing
narrative-scene path/to/video.mp4

# With a specific configuration
narrative-scene path/to/video.mp4 --config configs/film_analysis.json

# Process and query
narrative-scene path/to/video.mp4 --query "Why did Character A appear surprised when Character B entered the room?"
```

### Python API

```python
from narrative_scene_understanding import process_video, query_scene

# Process a video
results = process_video("path/to/video.mp4")

# Access the results
print(results["summary"])

# Query the processed scene
response = query_scene(
    "What was the emotional relationship between the two characters?", 
    results=results
)
print(response["answer"])
```

### Using Ollama for LLM Reasoning

```python
# Process with Ollama integration
from narrative_scene_understanding import process_video

config = {
    "query_engine": {
        "use_ollama": True,
        "ollama_model": "llama3",  # Or your preferred model
        "ollama_url": "http://localhost:11434",
        "temperature": 0.2
    }
}

results = process_video("path/to/video.mp4", config)

# Query using Ollama
response = query_scene(
    "What motivated Character A to hide the object?", 
    results=results
)
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/narrative-scene-understanding.git
cd narrative-scene-understanding

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install -e .
```

### Full Installation (with all optional components)

```bash
# Install system dependencies
sudo bash scripts/install_dependencies.sh

# Install with all optional dependencies
pip install -e ".[full]"

# Download required model weights
python scripts/download_models.py
```

## Key Features

- **Multi-modal integration** of visual, audio, and textual information
- **Dynamic knowledge graph** that evolves throughout the narrative
- **Character-centric analysis** for tracking individuals' journeys
- **Causal chain inference** for understanding "why" events happen
- **Natural language query interface** with LLM-powered reasoning
- **Cross-domain applicability** for film, security, sports, and personal media

## System Architecture

![System Architecture](docs/images/architecture.png)

The system consists of six main components:

1. **Data Ingestion Layer**: Extracts frames, detects scene boundaries, separates audio
2. **Multi-modal Processing**: Handles visual analysis, audio transcription, and OCR
3. **Knowledge Graph Construction**: Builds a dynamic graph of narrative elements
4. **Narrative Analysis**: Infers higher-level concepts like motivations and themes
5. **Query Interface**: Enables natural language questioning with LLM reasoning
6. **Output Visualization**: Presents results through interactive dashboards

## Applications

### Film and Media Analysis

- Character arc tracking and emotional development
- Narrative structure analysis
- Directorial technique identification
- Subtext and thematic exploration

### Security Footage Analysis

- Behavioral anomaly detection
- Incident reconstruction
- Intent and motivation analysis
- Causal chain extraction

### Sports Analysis

- Play pattern recognition
- Strategy assessment
- Player interaction dynamics
- Game-changing moment identification

### Personal Media Organization

- Event clustering and summarization
- Relationship mapping in personal content
- Semantic search capabilities
- Memory-based querying

## Advanced Usage

### Visualizing the Knowledge Graph

```bash
# Visualize the knowledge graph
narrative-visualize path/to/narrative_graph.json

# Filter to show only character relationships
narrative-visualize path/to/narrative_graph.json --filter characters
```

### Character Emotional Arcs

```bash
# Visualize character emotional arcs
narrative-arcs path/to/analysis_results.json

# Focus on specific characters
narrative-arcs path/to/analysis_results.json --characters "Character A" "Character B"
```

### Interactive Exploration

```bash
# Explore results interactively
narrative-explore --input path/to/results/directory
```

### Generating Summary Videos

```bash
# Create an annotated summary video
narrative-summary path/to/video.mp4 --graph path/to/narrative_graph.json
```

### Batch Processing

```bash
# Process multiple videos
narrative-batch --directory path/to/videos/folder
```

## Configuration

The system can be customized with different configurations:

- **default.json**: Balanced settings for general use
- **film_analysis.json**: Optimized for film narrative analysis
- **security.json**: Optimized for security footage
- **sports.json**: Optimized for sports videos
- **high_precision.json**: Maximum quality but slower processing

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Documentation](docs/api/index.md)
- [Advanced Usage](docs/advanced/index.md)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Formatting

```bash
# Format code
black modules/ tests/

# Check style
flake8 modules/ tests/
```

## Citation

If you use this system in your research, please cite:

```
@software{narrative_scene_understanding,
  author = {Narrative Scene Understanding Team},
  title = {Narrative Scene Understanding: Multi-modal Knowledge Graph Approach},
  year = {2025},
  url = {https://github.com/yourusername/narrative-scene-understanding}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SAM model is from Meta AI Research
- Whisper model is from OpenAI
- InsightFace for face recognition
- [Optional] Ollama for local LLM integration
