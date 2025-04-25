# Installation Guide

This guide walks you through the process of installing the **Narrative Scene Understanding** system and all of its dependencies.

---

## Prerequisites

Before installing, ensure your system meets the following requirements:

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended but not required)
- **FFmpeg** (for video and audio processing)
- **Git**

---

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/narrative-scene-understanding.git
cd narrative-scene-understanding
```

---

### 2. Install System Dependencies

The easiest way to install all system dependencies is to use the provided script:

```bash
sudo bash scripts/install_dependencies.sh
```

This script will install:

- Build tools and utilities
- OpenCV dependencies
- FFmpeg
- Audio processing libraries
- Tesseract OCR
- CUDA dependencies (optional)
- GraphViz for visualization

---

### 3. Set Up Python Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

---

### 4. Install Python Dependencies

To install the basic package:

```bash
pip install -e .
```

For a full installation with all optional dependencies:

```bash
pip install -e ".[full]"
```

---

### 5. Download Pre-trained Models

Download the required model weights by running:

```bash
python scripts/download_models.py
```

---

# You're all set!

Now you can proceed to using the Narrative Scene Understanding system.

---

## Using Ollama for LLM Integration

This system supports Ollama for local LLM integration, which is especially useful for the query engine.

### 1. Install Ollama

Follow the installation instructions from the Ollama website.

### 2. Pull the Required Model

```bash
ollama pull llama3  # Or any other model you prefer
```

### 3. Configure the System to Use Ollama

Either create a configuration file or modify the default configuration to use Ollama:

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

To use this configuration:

```bash
narrative-scene path/to/video.mp4 --config path/to/your/config.json
```

---

## Testing the Installation

To verify that everything is installed correctly:

```bash
python tests/test_installation.py
```

This script will check if all required components are properly installed and functional.

---

## Troubleshooting

### CUDA Issues

If you're having issues with CUDA:

- Verify your CUDA installation:

```bash
nvidia-smi
```

- Check that PyTorch is using CUDA:

```python
import torch
print(torch.cuda.is_available())
```

### Missing Dependencies

If you encounter errors about missing dependencies:

- Ensure all system dependencies are installed:

```bash
sudo apt-get install libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

- Install additional Python packages if needed:

```bash
pip install -r requirements.txt
```

### Model Download Issues

If model download fails:

- Check your internet connection
- Try downloading manually following instructions in `models/README.md`
- Check if you have enough disk space (models can be several GB)

---

## Alternative Installation Methods

### Using Docker

We provide a Docker image with all dependencies pre-installed:

```bash
docker pull yourusername/narrative-scene-understanding:latest
docker run -it --gpus all -v /path/to/videos:/videos yourusername/narrative-scene-understanding
```

### Conda Installation

```bash
conda create -n narrative-scene python=3.9
conda activate narrative-scene
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e .
```

---

# Next Steps

Once installation is complete, head to the Usage Guide to learn how to use the system.
