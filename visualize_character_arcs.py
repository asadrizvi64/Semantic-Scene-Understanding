#!/usr/bin/env python3
"""
Visualize Character Emotional Arcs.
This script creates visualizations of character emotional arcs over time based on narrative analysis.
"""

import os
import argparse
import logging
import networkx as nx
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

# Import utility functions
from modules.utils import setup_logging, load_graph_from_json

# Emotion to valence mapping
EMOTION_VALENCE = {
    "happy": 1.0,
    "joyful": 1.0,
    "excited": 0.8,
    "content": 0.6,
    "calm": 0.4,
    "neutral": 0.0,
    "bored": -0.2,
    "sad": -0.6,
    "angry": -0.8,
    "furious": -0.9,
    "afraid": -0.7,
    "terrified": -0.9,
    "disgusted": -0.7,
    "surprised": 0.3,
    "shocked": -0.4,
    "confused": -0.3,
    "anxious": -0.5,
    "nervous": -0.4,
    "embarrassed": -0.5,
    "ashamed": -0.7,
    "guilt": -0.7,
    "proud": 0.7,
    "loving": 0.9,
    "grateful": 0.8,
    "hopeful": 0.7,
    "relieved": 0.6,
    "satisfied": 0.7,
    "frustrated": -0.6,
    "disappointed": -0.5,
    "jealous": -0.6,
    "envious": -0.5,
    "suspicious": -0.4,
    "lonely": -0.7,
    "nostalgic": 0.3
}

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize Character Emotional Arcs")
    parser.add_argument("input_path", help="Path to the graph JSON file or analysis results")
    parser.add_argument("--output", "-o", default="character_arcs.png", help="Output file path")
    parser.add_argument("--characters", "-c", nargs="+", help="Specific characters to visualize (all if not specified)")
    parser.add_argument("--format", choices=["png", "svg", "pdf", "html"], default="png", 
                        help="Output file format")
    parser.add_argument("--width", type=int, default=10, help="Figure width in inches")
    parser.add_argument("--height", type=int, default=6, help="Figure height in inches")
    parser.add_argument("--interactive", "-i", action="store_true", help="Create interactive visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Check if plotly is available for interactive visualization
    if args.interactive and not HAVE_PLOTLY:
        logger.warning("Plotly not available, falling back to static visualization")
        args.interactive = False
    
    # Set output format
    output_path = args.output
    if not output_path.endswith(f".{args.format}"):
        output_path = f"{os.path.splitext(output_path)[0]}.{args.format}"
    
    # Load data
    try:
        if args.input_path.endswith('.json'):
            # Check if it's a graph or analysis results
            with open(args.input_path, 'r') as f:
                data = json.load(f)
            
            if "nodes" in data and "edges" in data:
                # It's a graph
                graph = load_graph_from_json(args.input_path)
                logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
                
                # Extract character emotion data
                character_data = extract_character_emotion_data(graph)
            else:
                # Assume it's analysis results
                character_data = {}
                if "characters" in data:
                    for char_id, char_info in data["characters"].items():
                        if "emotions" in char_info:
                            character_data[char_id] = char_info
                
                logger.info(f"Loaded analysis data with {len(character_data)} characters")
        else:
            logger.error(f"Unsupported input file format: {args.input_path}")
            return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Filter characters if specified
    if args.characters:
        filtered_data = {}
        for char_id, char_info in character_data.items():
            char_name = char_info.get("name", char_id)
            # Check if character name matches any of the specified names
            if any(name.lower() in char_name.lower() for name in args.characters):
                filtered_data[char_id] = char_info
        
        character_data = filtered_data
        logger.info(f"Filtered to {len(character_data)} characters")
    
    # Create visualization
    if len(character_data) == 0:
        logger.warning("No character data found for visualization")
        return
    
    if args.interactive and HAVE_PLOTLY:
        create_interactive_arc_visualization(character_data, output_path, args)
    else:
        create_static_arc_visualization(character_data, output_path, args)

def extract_character_emotion_data(graph: nx.DiGraph) -> Dict[str, Dict]:
    """
    Extract character emotion data from the graph.
    
    Args:
        graph: NetworkX DiGraph
        
    Returns:
        Dictionary mapping character IDs to info including emotions
    """
    character_data = {}
    
    # Find all character nodes
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "character":
            continue
        
        # Create character entry
        character_data[node_id] = {
            "name": attrs.get("name", node_id),
            "description": attrs.get("description", ""),
            "first_seen": attrs.get("first_seen", 0.0),
            "last_seen": attrs.get("last_seen", float('inf')),
            "emotions": []
        }
    
    # Find all emotion nodes linked to characters
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "emotion":
            continue
        
        # Find character that feels this emotion
        for pred in graph.predecessors(node_id):
            if graph.nodes[pred].get("type") == "character":
                char_id = pred
                
                if char_id in character_data:
                    # Add emotion to character data
                    character_data[char_id]["emotions"].append({
                        "emotion": attrs.get("emotion", "unknown"),
                        "timestamp": attrs.get("timestamp", 0.0),
                        "intensity": attrs.get("intensity", 0.5)
                    })
    
    # Sort emotions by timestamp for each character
    for char_id in character_data:
        character_data[char_id]["emotions"].sort(key=lambda e: e["timestamp"])
    
    return character_data

def emotion_to_valence(emotion: str) -> float:
    """
    Convert emotion name to valence value.
    
    Args:
        emotion: Emotion name
        
    Returns:
        Valence value between -1.0 and 1.0
    """
    emotion = emotion.lower()
    
    # Return exact match if available
    if emotion in EMOTION_VALENCE:
        return EMOTION_VALENCE[emotion]
    
    # Try to find partial match
    for known_emotion, valence in EMOTION_VALENCE.items():
        if known_emotion in emotion or emotion in known_emotion:
            return valence
    
    # Default to neutral
    return 0.0

def create_static_arc_visualization(character_data: Dict[str, Dict], output_path: str, args):
    """
    Create a static visualization of character emotional arcs.
    
    Args:
        character_data: Dictionary of character data including emotions
        output_path: Output file path
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating static character arc visualization")
    
    # Create figure
    plt.figure(figsize=(args.width, args.height))
    
    # Set up colors for characters
    color_map = plt.cm.get_cmap("tab10", len(character_data))
    
    # Plot each character's emotional arc
    for i, (char_id, char_info) in enumerate(character_data.items()):
        char_name = char_info.get("name", char_id)
        emotions = char_info.get("emotions", [])
        
        if not emotions:
            continue
        
        # Convert emotions to timestamps and valence values
        timestamps = [e["timestamp"] for e in emotions]
        valence = [emotion_to_valence(e["emotion"]) * e.get("intensity", 1.0) for e in emotions]
        
        # Plot the arc
        color = color_map(i)
        plt.plot(timestamps, valence, 'o-', label=char_name, color=color, linewidth=2, markersize=6)
        
        # Annotate emotions
        for j, (t, v, e) in enumerate(zip(timestamps, valence, emotions)):
            # Only annotate some points to avoid clutter
            if j % max(1, len(emotions) // 5) == 0 or j == len(emotions) - 1:
                plt.annotate(e["emotion"], (t, v), 
                           textcoords="offset points", 
                           xytext=(0, 10), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set up plot
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Neutral line
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotional Valence")
    plt.title("Character Emotional Arcs")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add markers for key events
    for char_id, char_info in character_data.items():
        if "arc" in char_info and "key_moments" in char_info["arc"]:
            for moment in char_info["arc"]["key_moments"]:
                timestamp = moment.get("timestamp", 0)
                plt.axvline(x=timestamp, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved character arc visualization to {output_path}")
    
    plt.close()

def create_interactive_arc_visualization(character_data: Dict[str, Dict], output_path: str, args):
    """
    Create an interactive visualization of character emotional arcs using Plotly.
    
    Args:
        character_data: Dictionary of character data including emotions
        output_path: Output file path
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    
    if not HAVE_PLOTLY:
        logger.error("Plotly is required for interactive visualization")
        return
    
    logger.info("Creating interactive character arc visualization")
    
    # Create figure with subplots (one shared plot and individual character plots)
    num_chars = len(character_data)
    fig = make_subplots(
        rows=num_chars + 1, 
        cols=1,
        shared_xaxes=True,
        row_heights=[2] + [1] * num_chars,
        subplot_titles=["All Characters"] + [info.get("name", char_id) for char_id, info in character_data.items()]
    )
    
    # Set up colors for characters
    colorscale = px.colors.qualitative.Plotly

    # Plot all characters in the top subplot
    for i, (char_id, char_info) in enumerate(character_data.items()):
        char_name = char_info.get("name", char_id)
        emotions = char_info.get("emotions", [])
        
        if not emotions:
            continue
        
        # Convert emotions to timestamps and valence values
        timestamps = [e["timestamp"] for e in emotions]
        valence = [emotion_to_valence(e["emotion"]) * e.get("intensity", 1.0) for e in emotions]
        emotion_names = [e["emotion"] for e in emotions]
        
        # Create hover text
        hover_text = [f"Character: {char_name}<br>Emotion: {e['emotion']}<br>Time: {e['timestamp']:.2f}s<br>Intensity: {e.get('intensity', 1.0):.2f}" 
                     for e in emotions]
        
        # Add line to the combined plot
        color = colorscale[i % len(colorscale)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=valence,
                mode='lines+markers',
                name=char_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                text=hover_text,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Add individual character plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=valence,
                mode='lines+markers+text',
                name=char_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                text=emotion_names,
                textposition="top center",
                textfont=dict(size=10),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ),
            row=i+2, col=1
        )
        
        # Add events if available
        if "arc" in char_info and "key_moments" in char_info["arc"]:
            for moment in char_info["arc"]["key_moments"]:
                timestamp = moment.get("timestamp", 0)
                from_emotion = moment.get("from_emotion", "")
                to_emotion = moment.get("to_emotion", "")
                
                # Add vertical line for event in individual character plot
                fig.add_shape(
                    type="line",
                    x0=timestamp, x1=timestamp,
                    y0=-1, y1=1,
                    line=dict(color="gray", width=1, dash="dot"),
                    row=i+2, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=timestamp,
                    y=0,
                    text=f"{from_emotion} → {to_emotion}",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="gray",
                    font=dict(size=8),
                    row=i+2, col=1
                )
    
    # Add horizontal lines at neutral (0) for all plots
    for i in range(num_chars + 1):
        fig.add_shape(
            type="line",
            x0=0, x1=max([max([e["timestamp"] for e in char_info.get("emotions", [])], default=10) 
                         for char_info in character_data.values()]),
            y0=0, y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Interactive Character Emotional Arcs",
        xaxis_title="Time (seconds)",
        yaxis_title="Emotional Valence",
        hovermode="closest",
        template="plotly_white",
        height=300 * (num_chars + 1),
        width=1000,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update y-axis ranges
    for i in range(num_chars + 1):
        fig.update_yaxes(range=[-1.1, 1.1], row=i+1, col=1)
        
        # Add y-axis labels for individual plots
        if i > 0:
            fig.update_yaxes(title_text="Valence", row=i+1, col=1)
    
    # Add emotion scale annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text="Emotion Scale: -1 (Negative) to +1 (Positive)",
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    # Save as HTML file
    fig.write_html(output_path)
    logger.info(f"Saved interactive character arc visualization to {output_path}")

if __name__ == "__main__":
    main()