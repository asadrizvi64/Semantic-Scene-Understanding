# Usage Guide

This guide explains how to use the Narrative Scene Understanding system for analyzing video content and querying it.

## Basic Usage

### Processing a Video

To process a video file and generate a narrative analysis:

```bash
narrative-scene path/to/video.mp4
```

This will:

- Extract frames from the video
- Process video and audio content
- Build a knowledge graph
- Generate a narrative summary
- Save the results to the output directory

The results include:

- A knowledge graph in JSON format
- A text summary of the scene
- Analysis data including character emotions, key events, and themes

### Querying a Scene

To ask questions about a video:

```bash
narrative-scene path/to/video.mp4 --query "Why did Character A appear surprised?"
```

This will:

- Process the video (if not already processed)
- Answer the question based on the narrative analysis
- Show the reasoning process and supporting evidence

### Batch Processing

To process multiple videos:

```bash
narrative-batch --directory path/to/videos/folder
```

Options include:

- `--recursive` to search subfolders
- `--limit` to process only a specific number of videos
- `--config` to specify a configuration file

### Visualizing Results

To visualize the knowledge graph:

```bash
narrative-visualize /path/to/narrative_graph.json
```

To visualize character emotional arcs:

```bash
narrative-arcs /path/to/analysis_results.json
```

## Using with Ollama

### Setting Up Ollama

First, make sure Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### Processing with Ollama Integration

You can use Ollama for the LLM component by specifying it in your config:

```bash
narrative-scene path/to/video.mp4 --config configs/ollama_config.json
```

Where `ollama_config.json` contains:

```json
{
  "query_engine": {
    "use_ollama": true,
    "ollama_model": "llama3",
    "ollama_url": "http://localhost:11434",
    "temperature": 0.2
  }
}
```

### Using Different Ollama Models

You can use different models available in Ollama:

```json
{
  "query_engine": {
    "use_ollama": true,
    "ollama_model": "mistral",  // Or llama3, vicuna, etc.
    "ollama_url": "http://localhost:11434",
    "temperature": 0.2
  }
}
```

## Configuration Options

The system behavior can be customized using configuration files:

```bash
narrative-scene path/to/video.mp4 --config configs/film_analysis.json
```

We provide several pre-configured files:

- `default.json`: Balanced settings for general use
- `film_analysis.json`: Optimized for film narrative analysis
- `security.json`: Optimized for security footage
- `sports.json`: Optimized for sports videos
- `high_precision.json`: Maximum quality but slower processing

### Key Configuration Parameters

- `frame_rate`: How many frames per second to extract (default: 2.0)
- `adaptive_sampling`: Whether to adapt frame rate based on motion (default: true)
- `max_frames`: Maximum number of frames to process (default: 500)
- `device`: Computation device ("cuda" or "cpu")
- `model_paths`: Paths to various model files
- `vision`: Visual processing options
- `audio`: Audio processing options
- `ocr`: OCR processing options
- `knowledge_graph`: Knowledge graph construction options
- `query_engine`: Query engine options including LLM settings

## Interactive Exploration

For an interactive exploration of the results:

```bash
narrative-explore --input /path/to/results/directory
```

This launches a web interface where you can:

- View the scene summary
- Explore the knowledge graph
- Analyze character emotions
- Ask questions about the scene
- Visualize relationships and timelines

## Generating Summary Videos

To create an annotated summary video:

```bash
narrative-summary path/to/video.mp4 --graph path/to/narrative_graph.json
```

Options include:

- `--highlights` to include only key moments
- `--speedup` to accelerate non-highlight segments
- `--height` to set the output video height

## Python API

You can also use the system programmatically:

```python
from narrative_scene_understanding import process_video, query_scene

# Process a video
results = process_video("path/to/video.mp4")

# Print the summary
print(results["summary"])

# Ask a question
response = query_scene("What is the main theme of this scene?", results=results)
print(response["answer"])
```

## Advanced Usage

For more advanced usage, including custom model integration, pipeline customization, and extending the system, see the Advanced Usage Guide.

## Examples

For practical examples, check out:

- Example notebooks in `examples/notebooks/`
- Example outputs in `examples/outputs/`
- Sample scripts in `examples/scripts/`
