# tests/test_query.py
"""
Tests for the scene query engine.
"""

import os
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

from modules.query import SceneQueryEngine, QueryContext

# Test fixtures
@pytest.fixture
def test_graph():
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
    
    # Connect character to emotions
    graph.add_edge("character_1", "emotion_1", relation="feels")
    graph.add_edge("character_1", "emotion_2", relation="feels")
    
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
    
    # Connect characters to speeches
    graph.add_edge("character_1", "speech_1", relation="speaks")
    graph.add_edge("character_1", "speech_2", relation="speaks")
    
    # Add causal relationships
    graph.add_edge("action_1", "emotion_2", relation="causes", probability=0.7)
    
    # Add an object
    graph.add_node(
        "object_1",
        type="object",
        description="A book",
        first_seen=0.0,
        last_seen=10.0
    )
    
    # Connect object to scene
    graph.add_edge("scene_0", "object_1", relation="contains")
    
    # Add a spatial relationship
    graph.add_edge(
        "character_1", "object_1",
        relation="spatial",
        spatial="near",
        first_seen=0.0,
        last_updated=5.0
    )
    
    # Connect everything to the scene
    graph.add_edge("scene_0", "character_1", relation="contains")
    graph.add_edge("scene_0", "character_2", relation="contains")
    
    for node in ["emotion_1", "emotion_2", "action_1", "action_2", "speech_1", "speech_2"]:
        graph.add_edge("scene_0", node, relation="contains")
    
    return graph

@pytest.mark.parametrize("llm_type", ["llama", "openai", None])
def test_query_engine_initialization(llm_type):
    """Test that the query engine initializes correctly with different LLM backends."""
    with patch("modules.query.Llama", autospec=True) if llm_type == "llama" else patch("modules.query.openai.OpenAI", autospec=True) if llm_type == "openai" else patch("builtins.open"):
        # Mock llama-cpp or openai package import
        if llm_type == "llama":
            with patch("os.path.exists", return_value=True):
                engine = SceneQueryEngine(model_name="llama-7b-chat")
                assert engine.llm_type == "llama"
        elif llm_type == "openai":
            engine = SceneQueryEngine(model_name="gpt-4o")
            assert engine.llm_type == "openai"
        else:
            # Both packages unavailable
            with patch("modules.query.Llama", side_effect=ImportError()), \
                 patch("modules.query.openai", side_effect=ImportError()):
                engine = SceneQueryEngine()
                assert engine.llm_type is None
                assert engine.llm is None

def test_query_scene(test_graph):
    """Test the main query_scene method."""
    # Create a mock LLM that returns a predictable response
    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [{"text": "This is a test response with reasoning.\n\nFinal answer."}]
    }
    
    # Create engine with mocked LLM
    with patch.object(SceneQueryEngine, "_generate_with_llama", return_value=("Test reasoning", "Test answer")):
        with patch.object(SceneQueryEngine, "_generate_with_openai", return_value=("Test reasoning", "Test answer")):
            engine = SceneQueryEngine()
            engine.llm = mock_llm
            engine.llm_type = "llama"  # Use llama for this test
            
            # Process a query
            question = "Why did Character A get angry?"
            response = engine.query_scene(question, test_graph)
            
            # Check response structure
            assert "answer" in response
            assert "reasoning" in response
            assert "evidence" in response
            assert "query_type" in response
            
            # Check response content
            assert response["answer"] == "Test answer"
            assert response["reasoning"] == "Test reasoning"

def test_parse_query():
    """Test the query parsing function."""
    engine = SceneQueryEngine()
    
    # Test motivation query
    question1 = "Why did Character A get angry at Character B?"
    query_type1, targets1 = engine._parse_query(question1)
    
    assert query_type1 == "motivation"
    assert "Character" in targets1
    assert "A" in targets1
    assert "B" in targets1
    
    # Test emotional query
    question2 = "How did Character A feel when Character B entered?"
    query_type2, targets2 = engine._parse_query(question2)
    
    assert query_type2 == "emotional"
    assert "Character" in targets2
    assert "A" in targets2
    assert "B" in targets2
    
    # Test relationship query
    question3 = "What is the relationship between Character A and Character B?"
    query_type3, targets3 = engine._parse_query(question3)
    
    assert query_type3 == "relationship"
    assert "Character" in targets3
    assert "A" in targets3
    assert "B" in targets3
    
    # Test general query
    question4 = "Describe the scene."
    query_type4, targets4 = engine._parse_query(question4)
    
    assert query_type4 == "general"
    assert len(targets4) == 0  # No specific targets

def test_retrieve_relevant_nodes(test_graph):
    """Test retrieval of relevant nodes based on query targets."""
    engine = SceneQueryEngine()
    
    # Test with Character A as target
    targets1 = {"Character", "A"}
    nodes1 = engine._retrieve_relevant_nodes(test_graph, targets1)
    
    # Should retrieve character_1, its emotions, actions, and speeches
    assert len(nodes1["characters"]) > 0
    assert any(node_id == "character_1" for node_id, _ in nodes1["characters"])
    assert len(nodes1["emotions"]) > 0
    assert len(nodes1["speeches"]) > 0
    
    # Test with Character B as target
    targets2 = {"Character", "B"}
    nodes2 = engine._retrieve_relevant_nodes(test_graph, targets2)
    
    # Should retrieve character_2 and its actions
    assert len(nodes2["characters"]) > 0
    assert any(node_id == "character_2" for node_id, _ in nodes2["characters"])
    assert len(nodes2["actions"]) > 0
    
    # Test with "book" as target
    targets3 = {"book"}
    nodes3 = engine._retrieve_relevant_nodes(test_graph, targets3)
    
    # Should retrieve the book object
    assert len(nodes3["objects"]) > 0
    assert any("book" in attrs.get("description", "").lower() for _, attrs in nodes3["objects"])
    
    # Test with empty targets (should retrieve core nodes)
    targets4 = set()
    nodes4 = engine._retrieve_relevant_nodes(test_graph, targets4)
    
    # Should retrieve core nodes like characters and major events
    assert len(nodes4["characters"]) > 0
    assert len(nodes4["events"]) == 0  # No event nodes in test_graph

def test_build_query_context(test_graph):
    """Test building a query context from relevant nodes."""
    engine = SceneQueryEngine()
    
    # Retrieve nodes related to Character A
    targets = {"Character", "A"}
    relevant_nodes = engine._retrieve_relevant_nodes(test_graph, targets)
    
    # Build context
    context = engine._build_query_context(relevant_nodes, test_graph, "emotional")
    
    # Check context structure
    assert hasattr(context, "characters")
    assert hasattr(context, "objects")
    assert hasattr(context, "actions")
    assert hasattr(context, "dialogue")
    assert hasattr(context, "spatial_info")
    assert hasattr(context, "emotional_states")
    assert hasattr(context, "events")
    
    # Check character info
    assert "character_1" in context.characters
    
    # Check emotion info
    assert "character_1" in context.emotional_states
    assert len(context.emotional_states["character_1"]) > 0
    
    # Check dialogue info
    assert len(context.dialogue) > 0
    assert any(speech["text"] == "Hello there" for speech in context.dialogue)
    
    # Check spatial info
    assert len(context.spatial_info) > 0
    assert any(info["subject"] == "character_1" and info["object"] == "object_1" for info in context.spatial_info)

def test_format_context_for_prompt():
    """Test formatting the query context into a prompt string."""
    engine = SceneQueryEngine()
    
    # Create a sample context
    context = QueryContext(
        characters={
            "character_1": {
                "description": "Main character",
                "traits": ["brave", "impulsive"],
                "first_seen": 0.0,
                "last_seen": 10.0
            }
        },
        objects={
            "object_1": {
                "description": "A book",
                "location": None,
                "state": None
            }
        },
        actions=[
            {
                "id": "action_1",
                "action": "throws an object",
                "subject": "character_1",
                "object": "object_1",
                "time": 5.0
            }
        ],
        dialogue=[
            {
                "id": "speech_1",
                "text": "Hello there",
                "speaker": "character_1",
                "time": 2.0,
                "sentiment": 0.3
            }
        ],
        spatial_info=[
            {
                "subject": "character_1",
                "object": "object_1",
                "relation": "near",
                "time": 5.0
            }
        ],
        emotional_states={
            "character_1": [
                {
                    "emotion": "angry",
                    "intensity": 0.8,
                    "time": 5.0,
                    "cause": None
                }
            ]
        },
        events=[]
    )
    
    # Format context for different query types
    prompt_emotional = engine._format_context_for_prompt(context, "emotional")
    prompt_motivation = engine._format_context_for_prompt(context, "motivation")
    
    # Check that sections are included
    assert "CHARACTERS:" in prompt_emotional
    assert "ACTIONS:" in prompt_emotional
    assert "DIALOGUE:" in prompt_emotional
    assert "SPATIAL RELATIONSHIPS:" in prompt_emotional
    assert "EMOTIONAL STATES:" in prompt_emotional
    
    # Character info should be included
    assert "character_1: Main character" in prompt_emotional
    
    # Emotional states are emphasized in emotional queries
    assert "EMOTIONAL STATES:" in prompt_emotional
    assert "character_1" in prompt_emotional
    assert "angry" in prompt_emotional

def test_generate_reasoning_and_answer():
    """Test the reasoning and answer generation."""
    # Create a mock LLM that returns a predictable response
    with patch.object(SceneQueryEngine, "_generate_with_llama", return_value=("Test reasoning", "Test answer")):
        with patch.object(SceneQueryEngine, "_generate_with_openai", return_value=("Test reasoning", "Test answer")):
            engine = SceneQueryEngine()
            engine.llm_type = "llama"  # Use llama for this test
            
            # Create a minimal context
            context = QueryContext(
                characters={"character_1": {"description": "Main character"}},
                objects={},
                actions=[],
                dialogue=[],
                spatial_info=[],
                emotional_states={},
                events=[]
            )
            
            # Generate reasoning and answer
            reasoning, answer = engine._generate_reasoning_and_answer(
                "Why did Character A get angry?",
                context,
                "motivation"
            )
            
            # Check results
            assert reasoning == "Test reasoning"
            assert answer == "Test answer"

def test_extract_supporting_evidence(test_graph):
    """Test extraction of supporting evidence for answers."""
    engine = SceneQueryEngine()
    
    # Retrieve nodes related to Character A
    targets = {"Character", "A"}
    relevant_nodes = engine._retrieve_relevant_nodes(test_graph, targets)
    
    # Extract evidence
    evidence = engine._extract_supporting_evidence(relevant_nodes, targets)
    
    # Check evidence structure
    assert len(evidence) > 0
    
    # Check evidence details
    for item in evidence:
        assert "type" in item
        assert "id" in item
        assert "relevance" in item
        
        # Evidence types should be character, action, or dialogue
        assert item["type"] in ["character", "action", "dialogue"]
        
        # Evidence relevance should be direct or contextual
        assert item["relevance"] in ["direct", "contextual"]

def test_llm_generation_error_handling():
    """Test error handling during LLM generation."""
    # Create an engine with a non-existent model
    engine = SceneQueryEngine(model_name="nonexistent-model")
    
    # Mock LLM to raise an exception
    with patch.object(SceneQueryEngine, "_generate_with_llama", side_effect=Exception("Test error")):
        with patch.object(SceneQueryEngine, "_generate_with_openai", side_effect=Exception("Test error")):
            engine.llm_type = "llama"  # Use llama for this test
            
            # Create a minimal context
            context = QueryContext(
                characters={},
                objects={},
                actions=[],
                dialogue=[],
                spatial_info=[],
                emotional_states={},
                events=[]
            )
            
            # Generate reasoning and answer
            reasoning, answer = engine._generate_reasoning_and_answer(
                "Test question",
                context,
                "general"
            )
            
            # Check that error handling returned a fallback message
            assert "Error generating reasoning" in reasoning
            assert "error analyzing this scene" in answer.lower()

def test_no_llm_error_handling(test_graph):
    """Test handling the case where no LLM is available."""
    # Create an engine with no LLM
    engine = SceneQueryEngine()
    engine.llm = None
    engine.llm_type = None
    
    # Process a query
    response = engine.query_scene("Test question", test_graph)
    
    # Check that an appropriate error message is returned
    assert "Cannot process query" in response["answer"]
    assert response["reasoning"] == ""
    assert response["evidence"] == []