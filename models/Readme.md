# Pre-trained Models for Narrative Scene Understanding

This directory is intended to store pre-trained models used by the Narrative Scene Understanding system. Due to their large size, these files are not included in the git repository and must be downloaded separately.

## Required Models

The following models are required for full functionality:

### Visual Processing

1. **SAM (Segment Anything Model)**
   - File: `sam_vit_h_4b8939.pth` (2.4 GB)
   - Source: [Meta AI Research](https://github.com/facebookresearch/segment-anything)
   - Used for: High-quality object segmentation

2. **InsightFace**
   - Files: Multiple model files in `insightface_model/` directory
   - Source: [InsightFace](https://github.com/deepinsight/insightface)
   - Used for: Face recognition and analysis

### Audio Processing

1. **Whisper**
   - Files: Downloaded automatically by the Whisper library
   - Source: [OpenAI](https://github.com/openai/whisper)
   - Used for: Speech transcription

2. **Pyannote Audio** (optional)
   - Files: Downloaded automatically by the pyannote.audio library
   - Source: [Pyannote](https://github.com/pyannote/pyannote-audio)
   - Used for: Speaker diarization

### Text Processing (LLM)

1. **Llama Model**
   - File: `llama-7b-chat.gguf` or similar (4-5 GB)
   - Source: Converted from Meta's Llama model
   - Used for: LLM reasoning in the query engine

## Downloading the Models

You can download the required models using the provided script:

```bash
python scripts/download_models.py