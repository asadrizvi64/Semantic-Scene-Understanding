#!/usr/bin/env python3
"""
Interactive exploration of Narrative Scene Understanding results.
This script provides a web-based interface for exploring analysis results.
"""

import os
import argparse
import logging
import json
import webbrowser
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from modules.utils import setup_logging, load_graph_from_json

# Try to import optional dependencies
try:
    import networkx as nx
    import dash
    from dash import dcc, html, ctx
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    HAVE_DASH = True
except ImportError:
    HAVE_DASH = False

# Define emotion to valence mapping (for emotional analysis)
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
    "nervous": -0.4
}

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

def main():
    """Main function for command-line usage."""
    if not HAVE_DASH:
        print("This script requires Dash and Plotly. Please install with:")
        print("pip install dash dash-bootstrap-components plotly pandas networkx")
        return
    
    parser = argparse.ArgumentParser(description="Interactive Exploration of Narrative Scene Understanding Results")
    parser.add_argument("--input", "-i", required=True, help="Path to results directory or specific result file")
    parser.add_argument("--port", type=int, default=8050, help="Port for web server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Load data
    try:
        data_sources = find_data_sources(args.input)
        if not data_sources:
            logger.error(f"No valid data sources found in {args.input}")
            return
        
        logger.info(f"Found {len(data_sources)} data sources")
    except Exception as e:
        logger.error(f"Error finding data sources: {e}")
        return
    
    # Create and run the Dash app
    app = create_dash_app(data_sources)
    
    # Open browser automatically
    if not args.no_browser:
        # Open browser in a separate thread
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    
    # Run the server
    app.run_server(debug=args.debug, port=args.port)

def find_data_sources(input_path: str) -> List[Dict[str, Any]]:
    """
    Find valid data sources in the input path.
    
    Args:
        input_path: Path to results directory or specific result file
        
    Returns:
        List of data source dictionaries
    """
    data_sources = []
    
    # Check if input path is a directory
    if os.path.isdir(input_path):
        # Look for graph.json, metadata.json, and summary.txt files
        for root, _, files in os.walk(input_path):
            metadata_file = None
            graph_file = None
            summary_file = None
            
            for file in files:
                if file.lower() == "metadata.json":
                    metadata_file = os.path.join(root, file)
                elif file.lower().endswith("graph.json") or "graph" in file.lower() and file.lower().endswith(".json"):
                    graph_file = os.path.join(root, file)
                elif file.lower() == "summary.txt" or file.lower().endswith("summary.txt"):
                    summary_file = os.path.join(root, file)
            
            # If we have at least a graph or metadata file, add as source
            if graph_file or metadata_file:
                source = {
                    "name": os.path.basename(root),
                    "path": root
                }
                
                if metadata_file:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        source["metadata"] = metadata
                    except Exception as e:
                        print(f"Error loading metadata file {metadata_file}: {e}")
                
                if graph_file:
                    source["graph_path"] = graph_file
                
                if summary_file:
                    try:
                        with open(summary_file, 'r') as f:
                            summary = f.read()
                        source["summary"] = summary
                    except Exception as e:
                        print(f"Error loading summary file {summary_file}: {e}")
                
                data_sources.append(source)
    else:
        # Input is a specific file
        if input_path.lower().endswith(".json"):
            # Check if it's a graph or analysis results
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                source = {
                    "name": os.path.basename(os.path.dirname(input_path)),
                    "path": os.path.dirname(input_path)
                }
                
                if "nodes" in data and "edges" in data:
                    # It's a graph
                    source["graph_path"] = input_path
                elif "characters" in data:
                    # It's analysis results
                    source["analysis"] = data
                
                # Look for summary file in same directory
                summary_file = os.path.join(os.path.dirname(input_path), "summary.txt")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r') as f:
                            summary = f.read()
                        source["summary"] = summary
                    except Exception as e:
                        print(f"Error loading summary file {summary_file}: {e}")
                
                data_sources.append(source)
            except Exception as e:
                print(f"Error loading JSON file {input_path}: {e}")
    
    return data_sources

def create_dash_app(data_sources: List[Dict[str, Any]]) -> dash.Dash:
    """
    Create a Dash app for exploring analysis results.
    
    Args:
        data_sources: List of data source dictionaries
        
    Returns:
        Dash app instance
    """
    # Initialize the app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Narrative Scene Understanding Explorer"
    
    # Create source selector options
    source_options = [{"label": source["name"], "value": i} for i, source in enumerate(data_sources)]
    
    # Create layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Narrative Scene Understanding Explorer", className="mt-4 mb-4"),
                html.P("Select a data source and view different visualizations"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Source"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="source-selector",
                            options=source_options,
                            value=0 if source_options else None,
                            clearable=False
                        ),
                        html.Div(id="source-metadata", className="mt-3")
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Scene Summary"),
                    dbc.CardBody([
                        html.Div(id="summary-content", style={"white-space": "pre-line"})
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Query Scene"),
                    dbc.CardBody([
                        dbc.Input(
                            id="query-input",
                            placeholder="Ask a question about the scene...",
                            type="text",
                            className="mb-2"
                        ),
                        dbc.Button("Ask", id="query-button", color="primary", className="mb-3"),
                        html.Div(id="query-result")
                    ])
                ])
            ], md=4),
            
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab([
                        dcc.Graph(id="character-arcs-graph", style={"height": "600px"})
                    ], label="Character Arcs"),
                    
                    dbc.Tab([
                        dcc.Graph(id="knowledge-graph", style={"height": "600px"})
                    ], label="Knowledge Graph"),
                    
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Character Selection"),
                                dcc.Dropdown(
                                    id="character-selector",
                                    multi=True
                                )
                            ], md=6),
                            dbc.Col([
                                html.H5("Relationship Type"),
                                dcc.Checklist(
                                    id="relationship-type",
                                    options=[
                                        {"label": "Spatial", "value": "spatial"},
                                        {"label": "Causal", "value": "causes"},
                                        {"label": "Social", "value": "relationship"}
                                    ],
                                    value=["spatial", "causes", "relationship"],
                                    inline=True
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        dcc.Graph(id="relationship-graph", style={"height": "500px"})
                    ], label="Relationships"),
                    
                    dbc.Tab([
                        html.H5("Timeline Visualization", className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Event Types"),
                                dcc.Checklist(
                                    id="event-types",
                                    options=[
                                        {"label": "Actions", "value": "action"},
                                        {"label": "Speech", "value": "speech"},
                                        {"label": "Emotions", "value": "emotion"}
                                    ],
                                    value=["action", "speech", "emotion"],
                                    inline=True
                                )
                            ], md=6)
                        ], className="mb-3"),
                        
                        dcc.Graph(id="timeline-graph", style={"height": "500px"})
                    ], label="Timeline")
                ])
            ], md=8)
        ])
    ], fluid=True)
    
    # Define callbacks
    @app.callback(
        Output("source-metadata", "children"),
        Input("source-selector", "value")
    )
    def update_source_metadata(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return "No source selected"
        
        source = data_sources[source_idx]
        metadata = source.get("metadata", {})
        
        # Create metadata display
        metadata_items = []
        
        if "video_path" in metadata:
            metadata_items.append(html.P(f"Video: {metadata['video_path']}"))
        
        if "process_time" in metadata:
            metadata_items.append(html.P(f"Processing Time: {metadata['process_time']:.2f} seconds"))
        
        if "frame_count" in metadata:
            metadata_items.append(html.P(f"Frames Processed: {metadata['frame_count']}"))
        
        if "speech_segments" in metadata:
            metadata_items.append(html.P(f"Speech Segments: {metadata['speech_segments']}"))
        
        if "processed_at" in metadata:
            metadata_items.append(html.P(f"Processed At: {metadata['processed_at']}"))
        
        return metadata_items if metadata_items else "No metadata available"
    
    @app.callback(
        Output("summary-content", "children"),
        Input("source-selector", "value")
    )
    def update_summary(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return "No source selected"
        
        source = data_sources[source_idx]
        summary = source.get("summary", "No summary available")
        
        return summary
    
    @app.callback(
        Output("character-arcs-graph", "figure"),
        Input("source-selector", "value")
    )
    def update_character_arcs(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return go.Figure()
        
        source = data_sources[source_idx]
        
        # Load graph if available
        graph = None
        if "graph_path" in source:
            try:
                graph = load_graph_from_json(source["graph_path"])
            except Exception as e:
                print(f"Error loading graph: {e}")
        
        # Extract character emotion data
        character_data = {}
        
        if graph:
            # Extract from graph
            # Find all character nodes
            for node_id, attrs in graph.nodes(data=True):
                if attrs.get("type") != "character":
                    continue
                
                # Create character entry
                character_data[node_id] = {
                    "name": attrs.get("name", node_id),
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
        elif "analysis" in source and "characters" in source["analysis"]:
            # Extract from analysis results
            for char_id, char_info in source["analysis"]["characters"].items():
                if "emotions" in char_info:
                    character_data[char_id] = {
                        "name": char_info.get("name", char_id),
                        "emotions": char_info["emotions"]
                    }
        
        # Create the arcs visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add a trace for each character
        for char_id, char_info in character_data.items():
            emotions = char_info.get("emotions", [])
            
            if not emotions:
                continue
            
            char_name = char_info.get("name", char_id)
            
            # Convert emotions to timestamps and valence values
            timestamps = [e.get("timestamp", 0.0) for e in emotions]
            valence = [emotion_to_valence(e.get("emotion", "neutral")) * e.get("intensity", 1.0) for e in emotions]
            
            # Create hover text
            hover_text = [f"Character: {char_name}<br>Emotion: {e.get('emotion', 'unknown')}<br>Time: {e.get('timestamp', 0.0):.2f}s<br>Intensity: {e.get('intensity', 0.5):.2f}" 
                         for e in emotions]
            
            # Add line
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=valence,
                    mode='lines+markers',
                    name=char_name,
                    text=hover_text,
                    hoverinfo='text'
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Character Emotional Arcs",
            xaxis_title="Time (seconds)",
            yaxis_title="Emotional Valence",
            yaxis=dict(range=[-1.1, 1.1]),
            hovermode="closest",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add neutral line
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max([max([e.get("timestamp", 0.0) for e in char_info.get("emotions", [])], default=10) 
                  for char_info in character_data.values()]),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        return fig
    
    @app.callback(
        Output("knowledge-graph", "figure"),
        Input("source-selector", "value")
    )
    def update_knowledge_graph(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return go.Figure()
        
        source = data_sources[source_idx]
        
        # Load graph if available
        if "graph_path" not in source:
            return go.Figure(layout=dict(title="No graph data available"))
        
        try:
            graph = load_graph_from_json(source["graph_path"])
        except Exception as e:
            print(f"Error loading graph: {e}")
            return go.Figure(layout=dict(title=f"Error loading graph: {e}"))
        
        # Create networkx layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Define node colors based on node type
        node_type_colors = {
            "character": "#3498db",  # Blue
            "object": "#2ecc71",     # Green
            "action": "#e74c3c",     # Red
            "speech": "#9b59b6",     # Purple
            "emotion": "#f1c40f",    # Yellow
            "scene": "#95a5a6",      # Gray
            "goal": "#e67e22",       # Orange
            "time": "#1abc9c",       # Teal
            "text": "#34495e",       # Dark Blue
            "sound": "#fd79a8",      # Pink
            "caption": "#7f8c8d"     # Light Gray
        }
        
        # Group nodes by type
        nodes_by_type = {}
        
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            nodes_by_type[node_type].append((node, attrs))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for u, v, attrs in graph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Edge text
                relation = attrs.get("relation", "unknown")
                edge_text.append(f"Relation: {relation}")
        
        # Add edge trace
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                text=edge_text,
                name='Edges'
            )
        )
        
        # Add node traces for each node type
        for node_type, nodes in nodes_by_type.items():
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            color = node_type_colors.get(node_type, "#bdc3c7")
            
            for node, attrs in nodes:
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Create label
                    if node_type == "character":
                        label = f"Character: {attrs.get('name', node)}"
                    elif node_type == "emotion":
                        label = f"Emotion: {attrs.get('emotion', 'unknown')}"
                    elif node_type == "action":
                        label = f"Action: {attrs.get('description', 'unknown')}"
                    elif node_type == "speech":
                        label = f"Speech: {attrs.get('text', 'unknown')}"
                    else:
                        label = f"{node_type.title()}: {node}"
                    
                    node_text.append(label)
                    
                    # Scale node size based on type
                    if node_type == "character":
                        node_size.append(15)
                    elif node_type == "scene":
                        node_size.append(18)
                    else:
                        node_size.append(10)
            
            # Add node trace
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=node_size,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    text=node_text,
                    hoverinfo='text',
                    name=f"{node_type.title()}"
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Knowledge Graph",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    # Additional callbacks for relationship and timeline tabs
    @app.callback(
        Output("character-selector", "options"),
        Output("character-selector", "value"),
        Input("source-selector", "value")
    )
    def update_character_options(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return
        
    def update_character_options(source_idx):
        if source_idx is None or len(data_sources) <= source_idx:
            return [], []
        
        source = data_sources[source_idx]
        
        # Get characters from graph or analysis results
        characters = []
        
        if "graph_path" in source:
            try:
                graph = load_graph_from_json(source["graph_path"])
                for node, attrs in graph.nodes(data=True):
                    if attrs.get("type") == "character":
                        name = attrs.get("name", node)
                        characters.append({"label": name, "value": node})
            except Exception as e:
                print(f"Error loading graph: {e}")
        elif "analysis" in source and "characters" in source["analysis"]:
            for char_id, char_info in source["analysis"]["characters"].items():
                name = char_info.get("name", char_id)
                characters.append({"label": name, "value": char_id})
        
        # Set default value to all characters
        default_value = [char["value"] for char in characters]
        
        return characters, default_value
    
    @app.callback(
        Output("relationship-graph", "figure"),
        Input("source-selector", "value"),
        Input("character-selector", "value"),
        Input("relationship-type", "value")
    )
    def update_relationship_graph(source_idx, selected_characters, relationship_types):
        if source_idx is None or len(data_sources) <= source_idx or not selected_characters:
            return go.Figure()
        
        source = data_sources[source_idx]
        
        # Load graph if available
        if "graph_path" not in source:
            return go.Figure(layout=dict(title="No graph data available"))
        
        try:
            graph = load_graph_from_json(source["graph_path"])
        except Exception as e:
            print(f"Error loading graph: {e}")
            return go.Figure(layout=dict(title=f"Error loading graph: {e}"))
        
        # Create subgraph with selected characters and relationship types
        subgraph = nx.DiGraph()
        
        # Add selected character nodes
        for node in selected_characters:
            if node in graph.nodes:
                subgraph.add_node(node, **graph.nodes[node])
        
        # Add other nodes that have relationships with selected characters
        for u, v, attrs in graph.edges(data=True):
            relation = attrs.get("relation", "unknown")
            
            if relation not in relationship_types:
                continue
            
            if u in selected_characters and v not in subgraph:
                if v in graph.nodes:
                    subgraph.add_node(v, **graph.nodes[v])
            
            if v in selected_characters and u not in subgraph:
                if u in graph.nodes:
                    subgraph.add_node(u, **graph.nodes[u])
            
            # Add edge if both nodes are in subgraph
            if u in subgraph and v in subgraph:
                subgraph.add_edge(u, v, **attrs)
        
        # Create layout
        pos = nx.spring_layout(subgraph, seed=42)
        
        # Define node colors based on node type
        node_type_colors = {
            "character": "#3498db",  # Blue
            "object": "#2ecc71",     # Green
            "action": "#e74c3c",     # Red
            "speech": "#9b59b6",     # Purple
            "emotion": "#f1c40f",    # Yellow
            "scene": "#95a5a6",      # Gray
            "goal": "#e67e22",       # Orange
            "time": "#1abc9c",       # Teal
            "text": "#34495e",       # Dark Blue
            "sound": "#fd79a8",      # Pink
            "caption": "#7f8c8d"     # Light Gray
        }
        
        # Create edge traces by relation type
        edge_traces = []
        
        for relation in relationship_types:
            edge_x = []
            edge_y = []
            edge_text = []
            
            for u, v, attrs in subgraph.edges(data=True):
                if attrs.get("relation") == relation and u in pos and v in pos:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Edge text
                    if relation == "spatial":
                        spatial = attrs.get("spatial", "near")
                        edge_text.append(f"Spatial: {spatial}")
                    elif relation == "causes":
                        probability = attrs.get("probability", 0.5)
                        edge_text.append(f"Causes (probability: {probability:.2f})")
                    elif relation == "relationship":
                        rel_type = attrs.get("type", "unknown")
                        sentiment = attrs.get("sentiment", 0.0)
                        edge_text.append(f"Relationship: {rel_type} (sentiment: {sentiment:.2f})")
                    else:
                        edge_text.append(f"Relation: {relation}")
            
            if edge_x:
                # Set color based on relation type
                if relation == "spatial":
                    color = "#3498db"  # Blue
                elif relation == "causes":
                    color = "#e67e22"  # Orange
                elif relation == "relationship":
                    color = "#2ecc71"  # Green
                else:
                    color = "#95a5a6"  # Gray
                
                # Add edge trace
                edge_traces.append(
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=2, color=color),
                        hoverinfo='text',
                        text=edge_text,
                        name=f"{relation.title()}"
                    )
                )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node, attrs in subgraph.nodes(data=True):
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_type = attrs.get("type", "unknown")
                
                # Set color based on node type
                node_color.append(node_type_colors.get(node_type, "#bdc3c7"))
                
                # Set size based on node type and whether it's a selected character
                if node in selected_characters:
                    node_size.append(20)  # Larger for selected characters
                elif node_type == "character":
                    node_size.append(15)
                else:
                    node_size.append(10)
                
                # Create label
                if node_type == "character":
                    label = f"Character: {attrs.get('name', node)}"
                elif node_type == "emotion":
                    label = f"Emotion: {attrs.get('emotion', 'unknown')}"
                elif node_type == "action":
                    label = f"Action: {attrs.get('description', 'unknown')}"
                elif node_type == "speech":
                    label = f"Speech: {attrs.get('text', 'unknown')}"
                else:
                    label = f"{node_type.title()}: {node}"
                
                node_text.append(label)
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='white')
            ),
            text=[attrs.get("name", node) if node in selected_characters else "" 
                  for node, attrs in subgraph.nodes(data=True)],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            name='Nodes'
        )
        
        # Create figure with all traces
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title="Character Relationships",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
        
        return fig
    
    @app.callback(
        Output("timeline-graph", "figure"),
        Input("source-selector", "value"),
        Input("event-types", "value")
    )
    def update_timeline(source_idx, event_types):
        if source_idx is None or len(data_sources) <= source_idx:
            return go.Figure()
        
        source = data_sources[source_idx]
        
        # Load graph if available
        if "graph_path" not in source:
            return go.Figure(layout=dict(title="No graph data available"))
        
        try:
            graph = load_graph_from_json(source["graph_path"])
        except Exception as e:
            print(f"Error loading graph: {e}")
            return go.Figure(layout=dict(title=f"Error loading graph: {e}"))
        
        # Create timeline data
        events = []
        
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            
            if node_type not in event_types:
                continue
            
            # Get timestamp based on node type
            timestamp = None
            if node_type == "action":
                timestamp = attrs.get("timestamp", None)
            elif node_type == "speech":
                timestamp = attrs.get("start_time", None)
            elif node_type == "emotion":
                timestamp = attrs.get("timestamp", None)
            
            if timestamp is None:
                continue
            
            # Get character associated with event
            character = None
            character_name = "Unknown"
            
            for pred in graph.predecessors(node):
                pred_attrs = graph.nodes[pred]
                if pred_attrs.get("type") == "character":
                    character = pred
                    character_name = pred_attrs.get("name", pred)
                    break
            
            # Create event data
            event = {
                "timestamp": timestamp,
                "type": node_type,
                "character": character,
                "character_name": character_name
            }
            
            # Add type-specific data
            if node_type == "action":
                event["description"] = attrs.get("description", "unknown action")
            elif node_type == "speech":
                event["text"] = attrs.get("text", "")
            elif node_type == "emotion":
                event["emotion"] = attrs.get("emotion", "unknown")
                event["intensity"] = attrs.get("intensity", 0.5)
            
            events.append(event)
        
        # Sort events by timestamp
        events.sort(key=lambda e: e["timestamp"])
        
        # Group events by character
        events_by_character = {}
        
        for event in events:
            character = event["character"]
            if character not in events_by_character:
                events_by_character[character] = []
            events_by_character[character].append(event)
        
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each character
        y_pos = 0
        for character, char_events in events_by_character.items():
            char_name = char_events[0]["character_name"] if char_events else "Unknown"
            
            # Create events for this character
            for event in char_events:
                event_type = event["type"]
                timestamp = event["timestamp"]
                
                # Set marker properties based on event type
                if event_type == "action":
                    marker_symbol = "circle"
                    marker_color = "#e74c3c"  # Red
                    hover_text = f"Action: {event['description']}"
                elif event_type == "speech":
                    marker_symbol = "square"
                    marker_color = "#9b59b6"  # Purple
                    hover_text = f"Speech: {event['text']}"
                elif event_type == "emotion":
                    marker_symbol = "diamond"
                    marker_color = "#f1c40f"  # Yellow
                    
                    # Scale marker size by emotion intensity
                    intensity = event.get("intensity", 0.5)
                    marker_size = 10 + intensity * 10
                    
                    hover_text = f"Emotion: {event['emotion']} (intensity: {intensity:.2f})"
                else:
                    marker_symbol = "x"
                    marker_color = "#95a5a6"  # Gray
                    hover_text = f"Event: {event_type}"
                
                # Add marker
                fig.add_trace(
                    go.Scatter(
                        x=[timestamp],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(
                            symbol=marker_symbol,
                            color=marker_color,
                            size=12 if event_type != "emotion" else marker_size,
                            line=dict(width=1, color="white")
                        ),
                        name=event_type.title(),
                        text=hover_text,
                        hoverinfo="text",
                        showlegend=False
                    )
                )
            
            # Add character label
            fig.add_annotation(
                x=0,
                y=y_pos,
                text=char_name,
                showarrow=False,
                xshift=-100,
                align="right",
                xanchor="right",
                yanchor="middle",
                font=dict(size=12)
            )
            
            # Increment y position for next character
            y_pos += 1
        
        # Add legend for event types
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="circle", color="#e74c3c", size=10),
                name="Action"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="square", color="#9b59b6", size=10),
                name="Speech"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="diamond", color="#f1c40f", size=10),
                name="Emotion"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Event Timeline",
            xaxis_title="Time (seconds)",
            yaxis=dict(
                showticklabels=False,
                zeroline=False,
                range=[-1, y_pos]
            ),
            hovermode="closest",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add vertical grid lines
        if events:
            min_time = min(e["timestamp"] for e in events)
            max_time = max(e["timestamp"] for e in events)
            
            # Add grid lines at regular intervals
            interval = max(1.0, (max_time - min_time) / 10)
            for t in np.arange(min_time, max_time + interval, interval):
                fig.add_shape(
                    type="line",
                    x0=t, x1=t,
                    y0=-1, y1=y_pos,
                    line=dict(color="lightgray", width=1, dash="dot")
                )
        
        return fig
    
    @app.callback(
        Output("query-result", "children"),
        Input("query-button", "n_clicks"),
        State("query-input", "value"),
        State("source-selector", "value"),
        prevent_initial_call=True
    )
    def process_query(n_clicks, query, source_idx):
        if n_clicks is None or not query or source_idx is None or len(data_sources) <= source_idx:
            return html.Div("No query or source selected")
        
        source = data_sources[source_idx]
        
        # In a real implementation, this would use the query engine to process the query
        # Here we'll just return a placeholder response
        
        return html.Div([
            html.H5("Query Result"),
            html.P(f"Query: {query}"),
            html.P("This is a placeholder response. In a real implementation, this would use the SceneQueryEngine to process your query and return a detailed answer."),
            html.P("To enable real querying, you would need to:"),
            html.Ol([
                html.Li("Implement a REST API for the SceneQueryEngine"),
                html.Li("Make an AJAX call to that API from this callback"),
                html.Li("Parse and display the response")
            ])
        ])
    
    return app

if __name__ == "__main__":
    main()