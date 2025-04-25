# tests/test_knowledge_graph.py
"""
Tests for the knowledge graph module.
"""

import os
import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock

from modules.knowledge_graph import NarrativeGraphBuilder

# Test fixtures
@pytest.fixture
def test_config():
    return {
        "device": "cpu",
        "frame_rate": 2.0
    }

@pytest.fixture
def sample_visual_data():
    """Generate sample visual processing data."""
    return [
        {
            "timestamp": 0.0,
            "overall_caption": "A person standing in a room",
            "objects": [
                {
                    "id": "0_0",
                    "type": "person",
                    "box": [100, 100, 200, 300],
                    "score": 0.95,
                    "caption": "A person wearing a red shirt",
                    "track_id": 1
                },
                {
                    "id": "0_1",
                    "type": "object",
                    "box": [400, 200, 450, 250],
                    "score": 0.8,
                    "caption": "A book on a table",
                    "track_id": None
                }
            ],
            "actions": [
                {
                    "type": "standing",
                    "subject_id": "0_0",
                    "subject_type": "person",
                    "confidence": 0.9,
                    "description": "Person 1 is standing"
                }
            ],
            "faces": [
                {
                    "box": [120, 110, 180, 170],
                    "score": 0.85,
                    "embedding": np.random.rand(512),
                    "gender": "male",
                    "age": 30,
                    "person_id": None
                }
            ]
        },
        {
            "timestamp": 0.5,
            "overall_caption": "A person walking across the room",
            "objects": [
                {
                    "id": "1_0",
                    "type": "person",
                    "box": [150, 100, 250, 300],
                    "score": 0.95,
                    "caption": "A person wearing a red shirt",
                    "track_id": 1
                },
                {
                    "id": "1_1",
                    "type": "object",
                    "box": [400, 200, 450, 250],
                    "score": 0.8,
                    "caption": "A book on a table",
                    "track_id": None
                }
            ],
            "actions": [
                {
                    "type": "walking",
                    "subject_id": "1_0",
                    "subject_type": "person",
                    "confidence": 0.85,
                    "description": "Person 1 is walking"
                }
            ],
            "faces": [
                {
                    "box": [170, 110, 230, 170],
                    "score": 0.85,
                    "embedding": np.random.rand(512),
                    "gender": "male",
                    "age": 30,
                    "person_id": None
                }
            ]
        }
    ]

@pytest.fixture
def sample_audio_data():
    """Generate sample audio processing data."""
    return [
        {
            "type": "speech",
            "start": 0.2,
            "end": 1.0,
            "text": "Hello there",
            "confidence": 0.9,
            "speaker": "SPEAKER_1",
            "sentiment": 0.3
        },
        {
            "type": "non-speech",
            "start": 1.2,
            "end": 1.5,
            "class": "footsteps",
            "confidence": 0.75
        }
    ]

@pytest.fixture
def sample_ocr_data():
    """Generate sample OCR processing data."""
    return [
        {
            "timestamp": 0.0,
            "frame_idx": 0,
            "text_detections": [
                {
                    "text": "EXIT",
                    "confidence": 0.95,
                    "box": [500, 50, 580, 80],
                    "polygon": [[500, 50], [580, 50], [580, 80], [500, 80]],
                    "track_id": "text_0",
                    "first_seen": 0.0,
                    "occurrences": 1
                }
            ]
        },
        {
            "timestamp": 0.5,
            "frame_idx": 1,
            "text_detections": [
                {
                    "text": "EXIT",
                    "confidence": 0.93,
                    "box": [500, 50, 580, 80],
                    "polygon": [[500, 50], [580, 50], [580, 80], [500, 80]],
                    "track_id": "text_0",
                    "first_seen": 0.0,
                    "occurrences": 2
                }
            ]
        }
    ]

@pytest.fixture
def sample_scene_boundaries():
    """Generate sample scene boundaries."""
    return [0.0]  # Just one scene in this test

def test_graph_builder_initialization(test_config):
    """Test that the graph builder initializes correctly."""
    builder = NarrativeGraphBuilder(test_config)
    
    assert builder is not None
    assert builder.config == test_config
    assert isinstance(builder.graph, nx.DiGraph)

def test_build_graph(test_config, sample_visual_data, sample_audio_data, sample_ocr_data, sample_scene_boundaries):
    """Test that a graph is built correctly from sample data."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Build the graph
    graph = builder.build_graph(
        visual_data=sample_visual_data,
        audio_data=sample_audio_data,
        ocr_data=sample_ocr_data,
        scene_boundaries=sample_scene_boundaries
    )
    
    # Check that the graph is a NetworkX DiGraph
    assert isinstance(graph, nx.DiGraph)
    
    # Check that the graph has nodes
    assert len(graph.nodes) > 0
    
    # Check that the graph has expected node types
    node_types = set()
    for _, attrs in graph.nodes(data=True):
        if "type" in attrs:
            node_types.add(attrs["type"])
    
    # Should have at least some of these node types
    expected_types = {"scene", "character", "object", "action", "speech", "time"}
    assert len(node_types.intersection(expected_types)) > 0
    
    # Check that the graph has edges
    assert len(graph.edges) > 0

def test_scene_boundary_addition(test_config):
    """Test that scene boundaries are properly added to the graph."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Add scene boundaries
    scene_boundaries = [0.0, 5.0, 10.0]
    builder._add_scene_boundaries(scene_boundaries)
    
    # Check that scene nodes were created
    scene_nodes = [node for node, attrs in builder.graph.nodes(data=True) 
                  if attrs.get("type") == "scene"]
    assert len(scene_nodes) == len(scene_boundaries)
    
    # Check that time nodes were created
    time_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                 if attrs.get("type") == "time"]
    assert len(time_nodes) == len(scene_boundaries)
    
    # Check that scenes are connected sequentially
    for i in range(len(scene_nodes) - 1):
        assert builder.graph.has_edge(f"scene_{i}", f"scene_{i+1}")
        assert builder.graph.get_edge_data(f"scene_{i}", f"scene_{i+1}")["relation"] == "precedes"

def test_visual_data_processing(test_config, sample_visual_data, sample_scene_boundaries):
    """Test processing of visual data with character and object tracking."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Add scene boundaries
    builder._add_scene_boundaries(sample_scene_boundaries)
    
    # Process visual data
    builder._process_visual_data(sample_visual_data)
    
    # Check that character nodes were created
    character_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                      if attrs.get("type") == "character"]
    assert len(character_nodes) > 0
    
    # Check that object nodes were created
    object_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                   if attrs.get("type") == "object"]
    assert len(object_nodes) > 0
    
    # Check that action nodes were created
    action_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                   if attrs.get("type") == "action"]
    assert len(action_nodes) > 0
    
    # Check that characters have positions
    for node in character_nodes:
        positions = builder.graph.nodes[node].get("positions", [])
        assert len(positions) > 0
        for pos in positions:
            assert "timestamp" in pos
            assert "box" in pos
            assert "scene_id" in pos

def test_audio_data_processing(test_config, sample_audio_data, sample_scene_boundaries):
    """Test processing of audio data with speech and non-speech segments."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Add scene boundaries
    builder._add_scene_boundaries(sample_scene_boundaries)
    
    # Process audio data
    builder._process_audio_data(sample_audio_data)
    
    # Check that speech nodes were created
    speech_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                   if attrs.get("type") == "speech"]
    assert len(speech_nodes) > 0
    
    # Check that sound nodes were created
    sound_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                  if attrs.get("type") == "sound"]
    assert len(sound_nodes) > 0
    
    # Check speech node properties
    for node in speech_nodes:
        attrs = builder.graph.nodes[node]
        assert "text" in attrs
        assert "start_time" in attrs
        assert "end_time" in attrs
        assert "speaker_id" in attrs

def test_ocr_data_processing(test_config, sample_ocr_data, sample_scene_boundaries):
    """Test processing of OCR data with text tracking."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Add scene boundaries
    builder._add_scene_boundaries(sample_scene_boundaries)
    
    # Process OCR data
    builder._process_ocr_data(sample_ocr_data)
    
    # Check that text nodes were created
    text_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                 if attrs.get("type") == "text"]
    assert len(text_nodes) > 0
    
    # Check text node properties
    for node in text_nodes:
        attrs = builder.graph.nodes[node]
        assert "content" in attrs
        assert "confidence" in attrs
        assert "timestamp" in attrs

def test_spatial_relationship_processing(test_config, sample_visual_data, sample_scene_boundaries):
    """Test processing of spatial relationships between entities."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Add scene boundaries
    builder._add_scene_boundaries(sample_scene_boundaries)
    
    # Process visual data
    builder._process_visual_data(sample_visual_data)
    
    # Process spatial relationships for the first frame
    builder._process_spatial_relationships(sample_visual_data[0], 0.0, "scene_0")
    
    # Find spatial relationship edges
    spatial_edges = []
    for u, v, attrs in builder.graph.edges(data=True):
        if attrs.get("relation") == "spatial":
            spatial_edges.append((u, v, attrs))
    
    # Should have found at least one spatial relationship
    assert len(spatial_edges) > 0
    
    # Check spatial edge properties
    for u, v, attrs in spatial_edges:
        assert "spatial" in attrs
        assert attrs["spatial"] in ["left_of", "right_of", "above", "below"]
        assert "first_seen" in attrs
        assert "last_updated" in attrs

def test_entity_resolution(test_config):
    """Test entity resolution for character identity tracking."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Create character nodes with similar attributes but different positions
    builder.graph.add_node(
        "character_1",
        type="character",
        description="Person in red shirt",
        first_seen=0.0,
        last_seen=1.0,
        positions=[
            {"timestamp": 0.0, "box": [100, 100, 200, 300], "scene_id": "scene_0"}
        ]
    )
    
    builder.graph.add_node(
        "character_2",
        type="character",
        description="Person in red shirt",
        first_seen=5.0,
        last_seen=6.0,
        positions=[
            {"timestamp": 5.0, "box": [150, 100, 250, 300], "scene_id": "scene_1"}
        ]
    )
    
    # Run entity resolution
    builder._resolve_entities()
    
    # Check if one node was merged into another
    # Either character_1 should have both positions or character_2 should be gone
    if "character_1" in builder.graph.nodes and "character_2" in builder.graph.nodes:
        # Both exist, so character_1 should have both positions
        positions = builder.graph.nodes["character_1"].get("positions", [])
        assert len(positions) == 2
    else:
        # character_2 was merged into character_1
        assert "character_1" in builder.graph.nodes
        assert "character_2" not in builder.graph.nodes
        
        # character_1 should have both positions
        positions = builder.graph.nodes["character_1"].get("positions", [])
        assert len(positions) == 2

def test_relationship_inference(test_config):
    """Test inference of relationships between entities."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Create character and event nodes
    builder.graph.add_node("character_1", type="character")
    builder.graph.add_node("character_2", type="character")
    
    # Create action nodes
    builder.graph.add_node(
        "action_1",
        type="action",
        action_type="greet",
        description="greeting",
        timestamp=1.0
    )
    
    builder.graph.add_node(
        "action_2",
        type="action",
        action_type="wave",
        description="waving",
        timestamp=1.1
    )
    
    # Connect actions to characters
    builder.graph.add_edge("character_1", "action_1", relation="performs")
    builder.graph.add_edge("action_1", "character_2", relation="targets")
    
    builder.graph.add_edge("character_2", "action_2", relation="performs")
    builder.graph.add_edge("action_2", "character_1", relation="targets")
    
    # Create speech nodes
    builder.graph.add_node(
        "speech_1",
        type="speech",
        text="Hello there",
        start_time=1.2,
        end_time=1.5,
        speaker_id="SPEAKER_1"
    )
    
    builder.graph.add_node(
        "speech_2",
        type="speech",
        text="Hi, how are you?",
        start_time=1.6,
        end_time=2.0,
        speaker_id="SPEAKER_2"
    )
    
    # Connect speech to characters
    builder.graph.add_edge("character_1", "speech_1", relation="speaks")
    builder.graph.add_edge("character_2", "speech_2", relation="speaks")
    
    # Run relationship inference
    builder._infer_relationships()
    
    # Check that character relationships were inferred
    relationship_edges = []
    for u, v, attrs in builder.graph.edges(data=True):
        if attrs.get("relation") == "relationship":
            relationship_edges.append((u, v, attrs))
    
    # Should have found at least one relationship
    assert len(relationship_edges) > 0
    
    # Check relationship edge properties
    for u, v, attrs in relationship_edges:
        assert "type" in attrs
        assert "sentiment" in attrs
        assert "confidence" in attrs

def test_causal_relationship_inference(test_config):
    """Test inference of causal relationships between events."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Create character node
    builder.graph.add_node("character_1", type="character")
    
    # Create action and emotion nodes with close timestamps
    builder.graph.add_node(
        "action_1",
        type="action",
        action_type="surprise",
        description="surprising action",
        timestamp=1.0
    )
    
    builder.graph.add_node(
        "emotion_1",
        type="emotion",
        emotion="surprised",
        timestamp=1.2,
        intensity=0.8
    )
    
    # Connect character to action and emotion
    builder.graph.add_edge("character_1", "action_1", relation="performs")
    builder.graph.add_edge("character_1", "emotion_1", relation="feels")
    
    # Run causal inference
    builder._infer_causal_relationships()
    
    # Check that causal relationships were inferred
    causal_edges = []
    for u, v, attrs in builder.graph.edges(data=True):
        if attrs.get("relation") == "causes":
            causal_edges.append((u, v, attrs))
    
    # Should have found at least one causal relationship
    assert len(causal_edges) > 0
    
    # Check if the action causes the emotion
    action_causes_emotion = False
    for u, v, attrs in causal_edges:
        if u == "action_1" and v == "emotion_1":
            action_causes_emotion = True
            break
    
    assert action_causes_emotion, "Should infer that action causes emotion"

def test_character_goal_inference(test_config):
    """Test inference of character goals based on actions."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Create character node
    builder.graph.add_node("character_1", type="character")
    
    # Create object node
    builder.graph.add_node("object_1", type="object", name="book")
    
    # Create multiple actions targeting the same object
    for i in range(3):
        action_node = f"action_{i}"
        builder.graph.add_node(
            action_node,
            type="action",
            action_type="reach_for",
            description="reaching for the book",
            timestamp=1.0 + i * 0.5
        )
        
        # Connect character to action
        builder.graph.add_edge("character_1", action_node, relation="performs")
        
        # Connect action to object
        builder.graph.add_edge(action_node, "object_1", relation="targets")
    
    # Run goal inference
    builder._infer_character_goals()
    
    # Check that goal nodes were created
    goal_nodes = [node for node, attrs in builder.graph.nodes(data=True)
                 if attrs.get("type") == "goal"]
    
    # Should have inferred at least one goal
    assert len(goal_nodes) > 0
    
    # Check if the character is connected to the goal
    character_has_goal = False
    for goal_node in goal_nodes:
        if builder.graph.has_edge("character_1", goal_node):
            if builder.graph.get_edge_data("character_1", goal_node)["relation"] == "has_goal":
                character_has_goal = True
                break
    
    assert character_has_goal, "Character should have a goal"
    
    # Check goal properties
    for goal_node in goal_nodes:
        goal_attrs = builder.graph.nodes[goal_node]
        assert "description" in goal_attrs
        assert "target" in goal_attrs
        assert "confidence" in goal_attrs
        
        # Goal should be related to the object
        assert goal_attrs["target"] == "object_1"

def test_emotional_response_inference(test_config):
    """Test inference of emotional responses to events."""
    builder = NarrativeGraphBuilder(test_config)
    
    # Create character node
    builder.graph.add_node("character_1", type="character")
    
    # Create action node
    builder.graph.add_node(
        "action_1",
        type="action",
        action_type="scare",
        description="scary action",
        timestamp=1.0
    )
    
    # Create emotion node appearing after the action
    builder.graph.add_node(
        "emotion_1",
        type="emotion",
        emotion="afraid",
        timestamp=1.3,
        intensity=0.9
    )
    
    # Connect character to emotion
    builder.graph.add_edge("character_1", "emotion_1", relation="feels")
    
    # Run emotional response inference
    builder._infer_emotional_responses()
    
    # Check if a causal relationship was created between action and emotion
    has_causal_edge = builder.graph.has_edge("action_1", "emotion_1")
    
    if has_causal_edge:
        edge_data = builder.graph.get_edge_data("action_1", "emotion_1")
        assert edge_data["relation"] == "causes"
    else:
        # If no direct edge, check if the action indirectly causes the emotion
        # through some intermediate node
        paths = list(nx.all_simple_paths(builder.graph, "action_1", "emotion_1"))
        assert len(paths) > 0, "Action should cause emotion directly or indirectly"

def test_boxed_overlap_calculation():
    """Test the calculation of bounding box overlap."""
    builder = NarrativeGraphBuilder({"device": "cpu"})
    
    # Test cases:
    # 1. No overlap
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert not builder._boxes_overlap_or_close(box1, box2)
    
    # 2. Overlap
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    assert builder._boxes_overlap_or_close(box1, box2)
    
    # 3. Close but no overlap
    box1 = [0, 0, 10, 10]
    box2 = [15, 15, 25, 25]
    # Default threshold is 50, so these should be considered "close"
    assert builder._boxes_overlap_or_close(box1, box2)
    
    # 4. Not close
    box1 = [0, 0, 10, 10]
    box2 = [100, 100, 110, 110]
    assert not builder._boxes_overlap_or_close(box1, box2)

def test_find_current_scene():
    """Test finding the current scene based on timestamp."""
    builder = NarrativeGraphBuilder({"device": "cpu"})
    
    # Add scene nodes with timestamps
    builder.graph.add_node("scene_0", type="scene", timestamp=0.0)
    builder.graph.add_node("scene_1", type="scene", timestamp=5.0)
    builder.graph.add_node("scene_2", type="scene", timestamp=10.0)
    
    # Test finding scenes
    assert builder._find_current_scene(0.0) == "scene_0"
    assert builder._find_current_scene(2.5) == "scene_0"
    assert builder._find_current_scene(5.0) == "scene_1"
    assert builder._find_current_scene(7.5) == "scene_1"
    assert builder._find_current_scene(10.0) == "scene_2"
    assert builder._find_current_scene(15.0) == "scene_2"

def test_find_scenes_in_timespan():
    """Test finding scenes that overlap with a given timespan."""
    builder = NarrativeGraphBuilder({"device": "cpu"})
    
    # Add scene nodes with timestamps
    builder.graph.add_node("scene_0", type="scene", timestamp=0.0)
    builder.graph.add_node("scene_1", type="scene", timestamp=5.0)
    builder.graph.add_node("scene_2", type="scene", timestamp=10.0)
    
    # Test finding scenes in various timespans
    assert builder._find_scenes_in_timespan(0.0, 2.0) == ["scene_0"]
    assert builder._find_scenes_in_timespan(4.0, 6.0) == ["scene_0", "scene_1"]
    assert builder._find_scenes_in_timespan(7.0, 12.0) == ["scene_1", "scene_2"]
    
    # Test with larger timespan covering all scenes
    assert set(builder._find_scenes_in_timespan(0.0, 15.0)) == {"scene_0", "scene_1", "scene_2"}