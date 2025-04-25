# tests/test_analysis.py
"""
Tests for the narrative analysis module.
"""

import os
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from modules.analysis import NarrativeAnalyzer

# Test fixtures
@pytest.fixture
def test_config():
    return {
        "device": "cpu"
    }

@pytest.fixture
def sample_graph():
    """Create a sample narrative graph for testing."""
    graph = nx.DiGraph()
    
    # Add scene and time nodes
    graph.add_node("scene_0", type="scene", timestamp=0.0)
    graph.add_node("time_0", type="time", timestamp=0.0)
    graph.add_edge("scene_0", "time_0", relation="has_time")
    
    # Add character nodes
    graph.add_node(
        "character_1",
        type="character",
        name="Character A",
        description="Main character",
        first_seen=0.0,
        last_seen=10.0
    )
    
    graph.add_node(
        "character_2",
        type="character",
        name="Character B",
        description="Secondary character",
        first_seen=1.0,
        last_seen=10.0
    )
    
    # Add emotion nodes for Character A
    graph.add_node(
        "emotion_1",
        type="emotion",
        emotion="neutral",
        timestamp=0.5,
        intensity=0.5
    )
    
    graph.add_node(
        "emotion_2",
        type="emotion",
        emotion="angry",
        timestamp=5.0,
        intensity=0.8
    )
    
    graph.add_node(
        "emotion_3",
        type="emotion",
        emotion="sad",
        timestamp=8.0,
        intensity=0.7
    )
    
    # Connect character to emotions
    graph.add_edge("character_1", "emotion_1", relation="feels")
    graph.add_edge("character_1", "emotion_2", relation="feels")
    graph.add_edge("character_1", "emotion_3", relation="feels")
    
    # Add emotion for Character B
    graph.add_node(
        "emotion_4",
        type="emotion",
        emotion="afraid",
        timestamp=6.0,
        intensity=0.6
    )
    
    graph.add_edge("character_2", "emotion_4", relation="feels")
    
    # Add action nodes
    graph.add_node(
        "action_1",
        type="action",
        action_type="enter",
        description="enters the room",
        timestamp=1.0
    )
    
    graph.add_node(
        "action_2",
        type="action",
        action_type="throw",
        description="throws an object",
        timestamp=5.0
    )
    
    # Connect characters to actions
    graph.add_edge("character_2", "action_1", relation="performs")
    graph.add_edge("character_1", "action_2", relation="performs")
    
    # Add speech nodes
    graph.add_node(
        "speech_1",
        type="speech",
        text="Hello there",
        start_time=2.0,
        end_time=3.0,
        speaker="character_1"
    )
    
    graph.add_node(
        "speech_2",
        type="speech",
        text="What are you doing here?",
        start_time=4.0,
        end_time=5.0,
        speaker="character_1"
    )
    
    graph.add_node(
        "speech_3",
        type="speech",
        text="I'm sorry for intruding",
        start_time=7.0,
        end_time=8.0,
        speaker="character_2"
    )
    
    # Connect characters to speeches
    graph.add_edge("character_1", "speech_1", relation="speaks")
    graph.add_edge("character_1", "speech_2", relation="speaks")
    graph.add_edge("character_2", "speech_3", relation="speaks")
    
    # Add causal relationships
    graph.add_edge("action_1", "emotion_2", relation="causes", probability=0.7)
    graph.add_edge("action_2", "emotion_4", relation="causes", probability=0.8)
    
    # Connect everything to the scene
    graph.add_edge("scene_0", "character_1", relation="contains")
    graph.add_edge("scene_0", "character_2", relation="contains")
    
    for node in ["emotion_1", "emotion_2", "emotion_3", "emotion_4",
                "action_1", "action_2",
                "speech_1", "speech_2", "speech_3"]:
        graph.add_edge("scene_0", node, relation="contains")
    
    return graph

def test_analyzer_initialization(test_config):
    """Test that the narrative analyzer initializes correctly."""
    analyzer = NarrativeAnalyzer(test_config)
    
    assert analyzer is not None
    assert analyzer.config == test_config

def test_analyze(test_config, sample_graph):
    """Test the complete analysis process."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Analyze the graph
    results = analyzer.analyze(sample_graph)
    
    # Check that results contains the expected sections
    assert "characters" in results
    assert "causal_chains" in results
    assert "narrative_arcs" in results
    assert "key_events" in results
    assert "themes" in results
    
    # Check character analysis
    assert "character_1" in results["characters"]
    assert "character_2" in results["characters"]
    
    char1 = results["characters"]["character_1"]
    assert "emotions" in char1
    assert "goals" in char1
    assert "relationships" in char1
    assert "arc" in char1
    
    # Check that there are emotional states
    assert len(char1["emotions"]) == 3
    
    # Check character arc
    assert char1["arc"]["type"] in ["positive", "negative", "flat", "complex"]
    assert "start_state" in char1["arc"]
    assert "end_state" in char1["arc"]
    
    # Check causal chains
    assert len(results["causal_chains"]) > 0
    
    # Check key events
    assert len(results["key_events"]) > 0

def test_analyze_characters(test_config, sample_graph):
    """Test character analysis specifically."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Analyze characters
    character_analysis = analyzer._analyze_characters(sample_graph)
    
    # Check results
    assert "character_1" in character_analysis
    assert "character_2" in character_analysis
    
    # Check character 1 analysis
    char1 = character_analysis["character_1"]
    assert char1["id"] == "character_1"
    assert char1["name"] == "Character A"
    assert len(char1["emotions"]) == 3
    
    # Check emotional progression
    emotions = char1["emotions"]
    assert emotions[0]["emotion"] == "neutral"
    assert emotions[1]["emotion"] == "angry"
    assert emotions[2]["emotion"] == "sad"
    
    # Check character arc
    arc = char1["arc"]
    assert arc["type"] in ["negative", "complex", "flat"]  # Based on neutral->sad
    assert arc["start_state"] == "neutral"
    assert arc["end_state"] == "sad"

def test_analyze_character_emotions(test_config, sample_graph):
    """Test analysis of character emotions specifically."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Analyze character 1's emotions
    emotions = analyzer._analyze_character_emotions(sample_graph, "character_1")
    
    # Check results
    assert len(emotions) == 3
    
    # Check emotion ordering by timestamp
    assert emotions[0]["emotion"] == "neutral"
    assert emotions[0]["timestamp"] == 0.5
    
    assert emotions[1]["emotion"] == "angry"
    assert emotions[1]["timestamp"] == 5.0
    
    assert emotions[2]["emotion"] == "sad"
    assert emotions[2]["timestamp"] == 8.0

def test_analyze_character_goals(test_config, sample_graph):
    """Test analysis of character goals."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Add a goal node to the graph
    sample_graph.add_node(
        "goal_1",
        type="goal",
        description="find an object",
        target="object_1",
        confidence=0.8
    )
    
    # Connect character to goal
    sample_graph.add_edge("character_1", "goal_1", relation="has_goal")
    
    # Analyze character 1's goals
    goals = analyzer._analyze_character_goals(sample_graph, "character_1")
    
    # Check results
    assert len(goals) == 1
    assert goals[0]["description"] == "find an object"
    assert goals[0]["target"] == "object_1"
    assert goals[0]["confidence"] == 0.8

def test_analyze_character_relationships(test_config, sample_graph):
    """Test analysis of character relationships."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Add a relationship between characters
    sample_graph.add_edge(
        "character_1", "character_2",
        relation="relationship",
        type="antagonistic",
        sentiment=-0.7,
        confidence=0.8
    )
    
    # Analyze character 1's relationships
    relationships = analyzer._analyze_character_relationships(sample_graph, "character_1")
    
    # Check results
    assert len(relationships) == 1
    assert relationships[0]["character"] == "character_2"
    assert relationships[0]["type"] == "antagonistic"
    assert relationships[0]["sentiment"] == -0.7
    assert relationships[0]["confidence"] == 0.8

def test_analyze_character_arc(test_config, sample_graph):
    """Test analysis of character arcs."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Get character 1's emotions
    emotions = analyzer._analyze_character_emotions(sample_graph, "character_1")
    
    # Analyze character 1's arc
    arc = analyzer._analyze_character_arc(sample_graph, "character_1", emotions)
    
    # Check results
    assert "type" in arc
    assert "confidence" in arc
    assert "start_state" in arc
    assert "end_state" in arc
    assert "key_moments" in arc
    
    # Check start and end states
    assert arc["start_state"] == "neutral"
    assert arc["end_state"] == "sad"
    
    # Check key moments - should have at least 2 (neutral->angry, angry->sad)
    assert len(arc["key_moments"]) >= 2

def test_extract_causal_chains(test_config, sample_graph):
    """Test extraction of causal chains."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Extract causal chains
    causal_chains = analyzer._extract_causal_chains(sample_graph)
    
    # Check results
    assert len(causal_chains) > 0
    
    # Check structure of a causal chain
    chain = causal_chains[0]
    assert "events" in chain
    assert "confidence" in chain
    
    # Check events in the chain
    events = chain["events"]
    assert len(events) >= 2
    
    # First event should be an action, last event should be an emotion
    assert events[0]["type"] in ["action", "speech"]
    assert events[-1]["type"] == "emotion"

def test_extract_narrative_arcs(test_config, sample_graph):
    """Test extraction of narrative arcs."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Extract narrative arcs
    narrative_arcs = analyzer._extract_narrative_arcs(sample_graph)
    
    # Check results
    assert len(narrative_arcs) > 0
    
    # Check structure of a narrative arc
    arc = narrative_arcs[0]
    assert "type" in arc
    assert "start_time" in arc
    assert "end_time" in arc
    assert "scenes" in arc
    assert "key_characters" in arc
    
    # First arc should be of type "exposition"
    assert arc["type"] == "exposition"
    
    # Arc should include both characters
    assert "character_1" in arc["key_characters"]
    assert "character_2" in arc["key_characters"]

def test_extract_key_events(test_config, sample_graph):
    """Test extraction of key events."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Extract key events
    key_events = analyzer._extract_key_events(sample_graph)
    
    # Check results
    assert len(key_events) > 0
    
    # Events should be in chronological order
    for i in range(1, len(key_events)):
        assert key_events[i]["timestamp"] >= key_events[i-1]["timestamp"]
    
    # Check structure of a key event
    event = key_events[0]
    assert "id" in event
    assert "type" in event
    assert "timestamp" in event
    
    # Check that events include actions, emotions, and speeches
    event_types = [e["type"] for e in key_events]
    assert "action" in event_types or "speech" in event_types or "emotion" in event_types

def test_extract_themes(test_config, sample_graph):
    """Test extraction of themes."""
    analyzer = NarrativeAnalyzer(test_config)
    
    # Extract themes
    themes = analyzer._extract_themes(sample_graph)
    
    # Check results
    assert len(themes) > 0
    
    # Check structure of a theme
    theme = themes[0]
    assert "name" in theme
    assert "confidence" in theme
    assert "keyword_matches" in theme
    
    # Themes should be sorted by confidence (descending)
    for i in range(1, len(themes)):
        assert themes[i]["confidence"] <= themes[i-1]["confidence"]