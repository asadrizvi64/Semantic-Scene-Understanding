
# Advanced Usage Documentation

This section covers advanced usage scenarios, customization, and extension of the Narrative Scene Understanding system.

## Contents

- [Custom Model Integration](#custom-model-integration)
- [Pipeline Customization](#pipeline-customization)
- [Working with the Knowledge Graph](#working-with-the-knowledge-graph)
- [Advanced Querying](#advanced-querying)
- [Performance Optimization](#performance-optimization)
- [Integration with Other Systems](#integration-with-other-systems)
- [Extending the System](#extending-the-system)

## Custom Model Integration

### Using Custom Vision Models

You can integrate your own vision models by extending the `VisualProcessor` class:

```python
from modules.vision import VisualProcessor

class CustomVisualProcessor(VisualProcessor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your custom model
        self.custom_model = YourCustomModel(...)

    def _detect_objects(self, frame, frame_idx):
        # Override the object detection method with your custom implementation
        # Return a list of detected objects in the same format
        objects = []
        # Your custom detection logic here
        return objects
```

### Custom LLM Integration

Using Ollama with Custom Models

If you have custom models in Ollama, you can specify them in the configuration:

```python
config = {
    "query_engine": {
        "use_ollama": True,
        "ollama_model": "your-custom-model",  # Your custom model in Ollama
        "ollama_url": "http://localhost:11434",
        "temperature": 0.2
    }
}
```

You can also create and use a custom model in Ollama:

```bash
# Create a custom model using Modelfile
echo "FROM llama3" > Modelfile
echo "PARAMETER temperature 0.2" >> Modelfile
echo "SYSTEM You are a narrative analysis expert specialized in understanding stories, character motivations, and plot elements." >> Modelfile

# Create the model
ollama create narrative-expert -f Modelfile

# Use it in your configuration
config = {
    "query_engine": {
        "use_ollama": True,
        "ollama_model": "narrative-expert",
        "ollama_url": "http://localhost:11434"
    }
}
```

### Implementing a Custom LLM Backend

You can create your own LLM backend by extending the SceneQueryEngine class:

```python
from modules.query import SceneQueryEngine

class CustomQueryEngine(SceneQueryEngine):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        # Initialize your custom LLM
        self.custom_llm = YourCustomLLM(model_path)

    def _generate_response(self, prompt):
        # Implement your custom generation logic
        return self.custom_llm.generate(prompt)
```
... (continued content as before)
