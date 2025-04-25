# modules/__init__.py
"""
Narrative Scene Understanding package.
"""

from .ingestion import VideoPreprocessor
from .vision import VisualProcessor
from .audio import AudioProcessor
from .ocr import OCRProcessor
from .knowledge_graph import NarrativeGraphBuilder
from .analysis import NarrativeAnalyzer
from .query import SceneQueryEngine
from .utils import (
    setup_logging,
    generate_scene_summary,
    save_graph_to_json,
    load_graph_from_json,
    visualize_graph,
    find_character_by_name,
    find_node_by_content
)

__version__ = "0.1.0"
__author__ = "Narrative Scene Understanding Team"
__license__ = "MIT"
__description__ = "A system for deep semantic understanding of visual narratives"

# Set up logging with default settings
setup_logging()