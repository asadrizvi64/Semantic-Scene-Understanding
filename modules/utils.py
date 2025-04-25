# modules/utils.py
"""
Utility functions for Narrative Scene Understanding.
"""

import os
import logging
import networkx as nx
import json
from typing import Dict, List, Tuple, Any, Optional

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def generate_scene_summary(graph: nx.DiGraph, analysis_results: Dict) -> str:
    """
    Generate a natural language summary of a scene.
    
    Args:
        graph: Narrative knowledge graph
        analysis_results: Results from the narrative analyzer
        
    Returns:
        String containing the scene summary
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating scene summary")
    
    # Extract key information for the summary
    characters = analysis_results.get("characters", {})
    key_events = analysis_results.get("key_events", [])
    themes = analysis_results.get("themes", [])
    causal_chains = analysis_results.get("causal_chains", [])
    
    # Start building the summary
    summary_parts = []
    
    # Scene overview
    setting_desc = "This scene takes place"
    
    # Try to determine location
    locations = [
        node for node, attrs in graph.nodes(data=True)
        if attrs.get("type") == "location"
    ]
    
    if locations:
        location_name = graph.nodes[locations[0]].get("name", "an unspecified location")
        setting_desc += f" in {location_name}"
    else:
        setting_desc += " in an unspecified location"
    
    # Try to determine time
    time_nodes = [
        node for node, attrs in graph.nodes(data=True)
        if attrs.get("type") == "time"
    ]
    
    if time_nodes:
        # Use the first time node as a reference
        time_attrs = graph.nodes[time_nodes[0]]
        if "time_of_day" in time_attrs:
            setting_desc += f" during the {time_attrs['time_of_day']}"
    
    summary_parts.append(setting_desc + ".")
    
    # Character overview
    if characters:
        char_overview = "The scene involves "
        char_names = list(characters.keys())
        
        if len(char_names) == 1:
            char_overview += f"{char_names[0]}"
        elif len(char_names) == 2:
            char_overview += f"{char_names[0]} and {char_names[1]}"
        else:
            char_overview += ", ".join(char_names[:-1]) + f", and {char_names[-1]}"
        
        summary_parts.append(char_overview + ".")
    
    # Key events summary
    if key_events:
        events_summary = "The following key events occur: "
        event_descriptions = []
        
        for event in key_events:
            if event.get("type") == "action":
                # Action event
                subject = event.get("subject", "Someone")
                description = event.get("description", "does something")
                event_descriptions.append(f"{subject} {description}")
            elif event.get("type") == "speech":
                # Speech event
                speaker = event.get("speaker", "Someone")
                text = event.get("text", "says something")
                event_descriptions.append(f"{speaker} says \"{text}\"")
            elif event.get("type") == "emotion":
                # Emotion event
                character = event.get("character", "Someone")
                emotion = event.get("emotion", "feels something")
                event_descriptions.append(f"{character} feels {emotion}")
        
        if event_descriptions:
            events_summary += "; ".join(event_descriptions) + "."
            summary_parts.append(events_summary)
    
    # Causal chains (for narrative flow)
    if causal_chains and len(causal_chains) > 0:
        # Use the most confident causal chain
        top_chain = causal_chains[0]
        chain_events = top_chain.get("events", [])
        
        if len(chain_events) >= 2:
            cause_effect = "Notably, "
            
            # Format first event (cause)
            first = chain_events[0]
            if first.get("type") == "action":
                cause_effect += f"when {first.get('subject', 'someone')} {first.get('description', 'acts')}"
            elif first.get("type") == "speech":
                cause_effect += f"when {first.get('speaker', 'someone')} says \"{first.get('text', 'something')}\""
            else:
                cause_effect += f"when {first.get('id', 'an event')} occurs"
            
            # Format last event (effect)
            last = chain_events[-1]
            if last.get("type") == "action":
                cause_effect += f", this leads to {last.get('subject', 'someone')} {last.get('description', 'acting')}"
            elif last.get("type") == "speech":
                cause_effect += f", this prompts {last.get('speaker', 'someone')} to say \"{last.get('text', 'something')}\""
            elif last.get("type") == "emotion":
                cause_effect += f", this causes {last.get('character', 'someone')} to feel {last.get('emotion', 'an emotion')}"
            else:
                cause_effect += f", this results in {last.get('id', 'an event')}"
            
            summary_parts.append(cause_effect + ".")
    
    # Character emotions and relationships
    char_dynamics = []
    for char_id, char_info in characters.items():
        # Get emotions
        emotions = char_info.get("emotions", [])
        if emotions:
            latest_emotion = emotions[-1]
            emotion_text = f"{char_id} appears {latest_emotion.get('emotion', 'neutral')}"
            char_dynamics.append(emotion_text)
        
        # Get relationships
        relationships = char_info.get("relationships", [])
        for rel in relationships:
            other_char = rel.get("character")
            rel_type = rel.get("type")
            if other_char and rel_type:
                rel_text = f"{char_id} has a {rel_type} relationship with {other_char}"
                char_dynamics.append(rel_text)
    
    if char_dynamics:
        summary_parts.append("Character dynamics include: " + ". ".join(char_dynamics) + ".")
    
    # Themes
    if themes:
        top_themes = [theme.get("name") for theme in themes[:3]]
        if top_themes:
            theme_text = "The scene explores themes of "
            if len(top_themes) == 1:
                theme_text += f"{top_themes[0]}"
            elif len(top_themes) == 2:
                theme_text += f"{top_themes[0]} and {top_themes[1]}"
            else:
                theme_text += f"{top_themes[0]}, {top_themes[1]}, and {top_themes[2]}"
            summary_parts.append(theme_text + ".")
    
    # Character arcs
    arc_insights = []
    for char_id, char_info in characters.items():
        arc = char_info.get("arc", {})
        arc_type = arc.get("type")
        if arc_type and arc_type != "flat":
            start = arc.get("start_state", "neutral")
            end = arc.get("end_state", "neutral")
            arc_insights.append(f"{char_id} undergoes a {arc_type} arc from {start} to {end}")
    
    if arc_insights:
        summary_parts.append("Character development: " + ". ".join(arc_insights) + ".")
    
    # Combine all parts into the final summary
    summary = "\n\n".join(summary_parts)
    
    return summary

def save_graph_to_json(graph: nx.DiGraph, output_path: str):
    """
    Save a NetworkX graph to a JSON file.
    
    Args:
        graph: NetworkX DiGraph to save
        output_path: Path to save the JSON file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert NetworkX graph to a serializable format
    serializable_graph = {
        "nodes": [],
        "edges": []
    }
    
    # Convert nodes
    for node_id, attrs in graph.nodes(data=True):
        node_data = {
            "id": node_id
        }
        
        # Process node attributes
        for key, value in attrs.items():
            # Handle non-serializable types
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            
            node_data[key] = value
        
        serializable_graph["nodes"].append(node_data)
    
    # Convert edges
    for source, target, attrs in graph.edges(data=True):
        edge_data = {
            "source": source,
            "target": target
        }
        
        # Process edge attributes
        for key, value in attrs.items():
            # Handle non-serializable types
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            
            edge_data[key] = value
        
        serializable_graph["edges"].append(edge_data)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_graph, f, indent=2)
    
    logger.info(f"Graph saved to {output_path}")

def load_graph_from_json(input_path: str) -> nx.DiGraph:
    """
    Load a NetworkX graph from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        NetworkX DiGraph
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading graph from {input_path}")
    
    # Read JSON file
    with open(input_path, 'r') as f:
        graph_data = json.load(f)
    
    # Create new graph
    graph = nx.DiGraph()
    
    # Add nodes
    for node_data in graph_data.get("nodes", []):
        node_id = node_data.pop("id")
        graph.add_node(node_id, **node_data)
    
    # Add edges
    for edge_data in graph_data.get("edges", []):
        source = edge_data.pop("source")
        target = edge_data.pop("target")
        graph.add_edge(source, target, **edge_data)
    
    logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def visualize_graph(graph: nx.DiGraph, output_path: str = None):
    """
    Visualize a knowledge graph.
    
    Args:
        graph: NetworkX DiGraph to visualize
        output_path: Path to save the visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pygraphviz as pgv
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Visualization requires matplotlib and pygraphviz packages. Please install them.")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create node positions using Graphviz layout
    pos = graphviz_layout(graph, prog="dot")
    
    # Define node colors based on type
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        
        if node_type == 'character':
            node_colors.append('skyblue')
            node_sizes.append(300)
        elif node_type == 'object':
            node_colors.append('lightgreen')
            node_sizes.append(200)
        elif node_type == 'location':
            node_colors.append('orange')
            node_sizes.append(250)
        elif node_type == 'action':
            node_colors.append('salmon')
            node_sizes.append(200)
        elif node_type == 'speech':
            node_colors.append('plum')
            node_sizes.append(200)
        elif node_type == 'emotion':
            node_colors.append('pink')
            node_sizes.append(150)
        elif node_type == 'time':
            node_colors.append('lightgray')
            node_sizes.append(100)
        elif node_type == 'scene':
            node_colors.append('gold')
            node_sizes.append(350)
        else:
            node_colors.append('white')
            node_sizes.append(100)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=1, arrowsize=15)
    
    # Draw labels
    node_labels = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', '')
        if node_type == 'character':
            label = graph.nodes[node].get('name', node)
        elif node_type in ['action', 'speech', 'emotion']:
            # Keep labels short for these types
            label = node_type
        else:
            label = str(node)[:10]  # Truncate long labels
        node_labels[node] = label
    
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
    
    # Add title and adjust layout
    plt.title("Narrative Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def find_character_by_name(graph: nx.DiGraph, name: str) -> Optional[str]:
    """
    Find a character node by name.
    
    Args:
        graph: Narrative knowledge graph
        name: Character name to search for
        
    Returns:
        Character node ID if found, None otherwise
    """
    for node, attrs in graph.nodes(data=True):
        if attrs.get('type') == 'character':
            node_name = attrs.get('name', '')
            if node_name.lower() == name.lower() or name.lower() in node_name.lower():
                return node
    return None

def find_node_by_content(graph: nx.DiGraph, content: str, node_type: str = None) -> Optional[str]:
    """
    Find a node containing specific content.
    
    Args:
        graph: Narrative knowledge graph
        content: Content to search for
        node_type: Optional node type to filter by
        
    Returns:
        Node ID if found, None otherwise
    """
    for node, attrs in graph.nodes(data=True):
        # Filter by type if specified
        if node_type and attrs.get('type') != node_type:
            continue
        
        # Search in common text fields
        for field in ['text', 'description', 'name']:
            if field in attrs and content.lower() in str(attrs[field]).lower():
                return node
    
    return None