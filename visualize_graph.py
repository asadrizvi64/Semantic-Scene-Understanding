#!/usr/bin/env python3
"""
Visualize a Narrative Knowledge Graph.
This script creates visualizations of the knowledge graph using various layouts and visualization methods.
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
from typing import Dict, List, Tuple, Any, Optional, Set

# Try to import optional dependencies
try:
    import pygraphviz
    HAVE_GRAPHVIZ = True
except ImportError:
    HAVE_GRAPHVIZ = False

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

# Import utility functions
from modules.utils import setup_logging, load_graph_from_json

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize Narrative Knowledge Graph")
    parser.add_argument("graph_path", help="Path to the graph JSON file")
    parser.add_argument("--output", "-o", default="graph_visualization.png", help="Output file path")
    parser.add_argument("--layout", choices=["graphviz", "spring", "circular", "spectral", "timeline"], 
                        default="graphviz", help="Graph layout algorithm")
    parser.add_argument("--format", choices=["png", "svg", "pdf", "html"], default="png", 
                        help="Output file format")
    parser.add_argument("--filter", choices=["all", "characters", "actions", "emotions", "causal"], 
                        default="all", help="Filter the graph to specific node types")
    parser.add_argument("--label-size", type=int, default=8, help="Font size for node labels")
    parser.add_argument("--node-size", type=int, default=300, help="Base size for nodes")
    parser.add_argument("--edge-width", type=float, default=1.0, help="Width for edges")
    parser.add_argument("--width", type=int, default=12, help="Figure width in inches")
    parser.add_argument("--height", type=int, default=8, help="Figure height in inches")
    parser.add_argument("--interactive", "-i", action="store_true", help="Create interactive visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Check if graphviz is available
    if args.layout == "graphviz" and not HAVE_GRAPHVIZ:
        logger.warning("Graphviz not available, falling back to spring layout")
        args.layout = "spring"
    
    # Check if plotly is available for interactive visualization
    if args.interactive and not HAVE_PLOTLY:
        logger.warning("Plotly not available, falling back to static visualization")
        args.interactive = False
    
    # Load the graph
    try:
        graph = load_graph_from_json(args.graph_path)
        logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        return
    
    # Filter the graph if requested
    if args.filter != "all":
        graph = filter_graph(graph, args.filter)
        logger.info(f"Filtered graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Set output format
    output_path = args.output
    if not output_path.endswith(f".{args.format}"):
        output_path = f"{os.path.splitext(output_path)[0]}.{args.format}"
    
    # Create visualization
    if args.interactive:
        create_interactive_visualization(graph, output_path, args)
    else:
        create_static_visualization(graph, output_path, args)

def filter_graph(graph: nx.DiGraph, filter_type: str) -> nx.DiGraph:
    """
    Filter the graph to specific node types.
    
    Args:
        graph: NetworkX DiGraph
        filter_type: Type of filter to apply
        
    Returns:
        Filtered NetworkX DiGraph
    """
    # Create a new graph for the filtered nodes
    filtered_graph = nx.DiGraph()
    
    # Define filter criteria
    if filter_type == "characters":
        # Include character nodes and their direct connections
        node_types = ["character"]
        include_neighbors = True
    elif filter_type == "actions":
        # Include character, action, and object nodes
        node_types = ["character", "action", "object"]
        include_neighbors = False
    elif filter_type == "emotions":
        # Include character and emotion nodes
        node_types = ["character", "emotion"]
        include_neighbors = False
    elif filter_type == "causal":
        # Include nodes connected by causal relationships
        causal_nodes = set()
        for u, v, attrs in graph.edges(data=True):
            if attrs.get("relation") == "causes":
                causal_nodes.add(u)
                causal_nodes.add(v)
        
        # Add nodes and their attributes
        for node in causal_nodes:
            filtered_graph.add_node(node, **graph.nodes[node])
        
        # Add causal edges
        for u, v, attrs in graph.edges(data=True):
            if attrs.get("relation") == "causes" and u in causal_nodes and v in causal_nodes:
                filtered_graph.add_edge(u, v, **attrs)
        
        return filtered_graph
    else:
        # Unknown filter type, return original graph
        return graph
    
    # Add nodes of the specified types
    nodes_to_include = set()
    
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") in node_types:
            nodes_to_include.add(node)
    
    # If including neighbors, add them too
    if include_neighbors:
        neighbors = set()
        for node in nodes_to_include:
            neighbors.update(graph.predecessors(node))
            neighbors.update(graph.successors(node))
        nodes_to_include.update(neighbors)
    
    # Add nodes and their attributes
    for node in nodes_to_include:
        filtered_graph.add_node(node, **graph.nodes[node])
    
    # Add edges between included nodes
    for u, v, attrs in graph.edges(data=True):
        if u in filtered_graph.nodes and v in filtered_graph.nodes:
            filtered_graph.add_edge(u, v, **attrs)
    
    return filtered_graph

def create_static_visualization(graph: nx.DiGraph, output_path: str, args):
    """
    Create a static visualization of the graph.
    
    Args:
        graph: NetworkX DiGraph to visualize
        output_path: Path to save the visualization
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating static visualization with {args.layout} layout")
    
    # Create figure
    plt.figure(figsize=(args.width, args.height))
    
    # Get node positions based on layout algorithm
    if args.layout == "graphviz":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    elif args.layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif args.layout == "circular":
        pos = nx.circular_layout(graph)
    elif args.layout == "spectral":
        pos = nx.spectral_layout(graph)
    elif args.layout == "timeline":
        pos = create_timeline_layout(graph)
    else:
        logger.warning(f"Unknown layout: {args.layout}, falling back to spring layout")
        pos = nx.spring_layout(graph, seed=42)
    
    # Define node colors and sizes based on node type
    node_colors = []
    node_sizes = []
    
    # Create a colormap for different node types
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
    
    for node in graph.nodes():
        node_type = graph.nodes[node].get("type", "unknown")
        
        # Set color based on node type
        if node_type in node_type_colors:
            node_colors.append(node_type_colors[node_type])
        else:
            node_colors.append("#bdc3c7")  # Default gray
        
        # Set size based on node type
        base_size = args.node_size
        if node_type == "character":
            node_sizes.append(base_size * 1.5)
        elif node_type == "scene":
            node_sizes.append(base_size * 2.0)
        elif node_type == "emotion":
            # Scale emotion node size by intensity if available
            intensity = graph.nodes[node].get("intensity", 0.5)
            node_sizes.append(base_size * (0.5 + intensity))
        else:
            node_sizes.append(base_size)
    
    # Define edge colors based on edge type
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v, attrs in graph.edges(data=True):
        relation = attrs.get("relation", "unknown")
        
        if relation == "contains":
            edge_colors.append("#95a5a6")  # Gray
            edge_styles.append("dashed")
            edge_widths.append(args.edge_width * 0.5)
        elif relation == "performs":
            edge_colors.append("#e74c3c")  # Red
            edge_styles.append("solid")
            edge_widths.append(args.edge_width)
        elif relation == "speaks":
            edge_colors.append("#9b59b6")  # Purple
            edge_styles.append("solid")
            edge_widths.append(args.edge_width)
        elif relation == "feels":
            edge_colors.append("#f1c40f")  # Yellow
            edge_styles.append("solid")
            edge_widths.append(args.edge_width)
        elif relation == "spatial":
            edge_colors.append("#3498db")  # Blue
            edge_styles.append("dotted")
            edge_widths.append(args.edge_width * 0.5)
        elif relation == "causes":
            edge_colors.append("#e67e22")  # Orange
            edge_styles.append("solid")
            
            # Scale width by causation probability if available
            probability = attrs.get("probability", 0.5)
            edge_widths.append(args.edge_width * (0.5 + probability))
        elif relation == "relationship":
            edge_colors.append("#2ecc71")  # Green
            edge_styles.append("solid")
            edge_widths.append(args.edge_width * 1.5)
        elif relation == "has_goal":
            edge_colors.append("#1abc9c")  # Teal
            edge_styles.append("solid")
            edge_widths.append(args.edge_width)
        else:
            edge_colors.append("#bdc3c7")  # Light Gray
            edge_styles.append("solid")
            edge_widths.append(args.edge_width * 0.5)
    
    # Draw the graph
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5
    )
    
    # Draw edges with different styles
    for (u, v, attrs), color, style, width in zip(graph.edges(data=True), edge_colors, edge_styles, edge_widths):
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=[(u, v)],
            edge_color=color,
            style=style,
            width=width,
            alpha=0.7,
            arrowsize=10
        )
    
    # Create node labels
    node_labels = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get("type", "")
        
        if node_type == "character":
            label = graph.nodes[node].get("name", node)
        elif node_type == "emotion":
            label = f"{graph.nodes[node].get('emotion', 'unknown')}"
        elif node_type == "action":
            # Shorten action descriptions
            desc = graph.nodes[node].get("description", "")
            label = desc[:15] + "..." if len(desc) > 15 else desc
        elif node_type == "speech":
            # Shorten speech text
            text = graph.nodes[node].get("text", "")
            label = text[:15] + "..." if len(text) > 15 else text
        elif node_type == "object":
            label = graph.nodes[node].get("description", node)
        else:
            label = node_type
        
        node_labels[node] = label
    
    # Draw node labels
    nx.draw_networkx_labels(
        graph, pos,
        labels=node_labels,
        font_size=args.label_size,
        font_family="sans-serif",
        font_weight="bold"
    )
    
    # Add a legend for node types
    legend_elements = []
    used_node_types = set()
    
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        used_node_types.add(node_type)
    
    for node_type in sorted(used_node_types):
        if node_type in node_type_colors:
            color = node_type_colors[node_type]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=10, 
                                             label=node_type.title()))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Set up plot
    plt.title("Narrative Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    
    plt.close()

def create_timeline_layout(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Create a timeline-based layout for the graph.
    
    Args:
        graph: NetworkX DiGraph
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates
    """
    pos = {}
    
    # Get time info for each node
    node_times = {}
    
    for node, attrs in graph.nodes(data=True):
        # Get time based on node type
        if attrs.get("type") == "scene":
            time = attrs.get("timestamp", 0.0)
        elif attrs.get("type") == "speech":
            time = attrs.get("start_time", 0.0)
        elif attrs.get("type") in ["action", "emotion"]:
            time = attrs.get("timestamp", 0.0)
        elif attrs.get("type") == "character":
            # Use first appearance time
            time = attrs.get("first_seen", 0.0)
        else:
            # For other node types, use average of connected nodes
            times = []
            for neighbor in graph.neighbors(node):
                neighbor_attrs = graph.nodes[neighbor]
                if "timestamp" in neighbor_attrs:
                    times.append(neighbor_attrs["timestamp"])
                elif "start_time" in neighbor_attrs:
                    times.append(neighbor_attrs["start_time"])
            
            if times:
                time = sum(times) / len(times)
            else:
                time = 0.0
        
        node_times[node] = time
    
    # Normalize timeline to 0-10 range
    if node_times:
        min_time = min(node_times.values())
        max_time = max(node_times.values())
        time_range = max(max_time - min_time, 0.001)  # Avoid division by zero
        
        for node, time in node_times.items():
            normalized_time = (time - min_time) / time_range * 10.0
            node_times[node] = normalized_time
    
    # Group nodes by type for vertical placement
    node_types = set()
    for _, attrs in graph.nodes(data=True):
        node_types.add(attrs.get("type", "unknown"))
    
    # Assign vertical positions based on node type
    vertical_positions = {}
    for i, node_type in enumerate(sorted(node_types)):
        vertical_positions[node_type] = i
    
    # Calculate positions
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        time = node_times[node]
        
        # Add some vertical jitter within each type
        jitter = (hash(node) % 100) / 200.0  # Small random jitter
        y_pos = vertical_positions[node_type] + jitter
        
        pos[node] = (time, y_pos)
    
    return pos

def create_interactive_visualization(graph: nx.DiGraph, output_path: str, args):
    """
    Create an interactive visualization of the graph using Plotly.
    
    Args:
        graph: NetworkX DiGraph to visualize
        output_path: Path to save the visualization
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    
    if not HAVE_PLOTLY:
        logger.error("Plotly is required for interactive visualization")
        return
    
    logger.info("Creating interactive visualization")
    
    # Get node positions
    if args.layout == "graphviz" and HAVE_GRAPHVIZ:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    elif args.layout == "timeline":
        pos = create_timeline_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42)
    
    # Create edge traces
    edge_traces = []
    
    # Define edge colors based on edge type
    edge_colors = {
        "contains": "#95a5a6",  # Gray
        "performs": "#e74c3c",  # Red
        "speaks": "#9b59b6",    # Purple
        "feels": "#f1c40f",     # Yellow
        "spatial": "#3498db",   # Blue
        "causes": "#e67e22",    # Orange
        "relationship": "#2ecc71",  # Green
        "has_goal": "#1abc9c",  # Teal
        "unknown": "#bdc3c7"    # Light Gray
    }
    
    # Group edges by relation type
    edges_by_relation = {}
    
    for u, v, attrs in graph.edges(data=True):
        relation = attrs.get("relation", "unknown")
        
        if relation not in edges_by_relation:
            edges_by_relation[relation] = []
        
        edges_by_relation[relation].append((u, v, attrs))
    
    # Create edge traces for each relation type
    for relation, edges in edges_by_relation.items():
        x_edges = []
        y_edges = []
        edge_text = []
        
        for u, v, attrs in edges:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                x_edges.extend([x0, x1, None])
                y_edges.extend([y0, y1, None])
                
                # Create edge label
                label = f"Relation: {relation}"
                if relation == "causes" and "probability" in attrs:
                    label += f"\nProbability: {attrs['probability']:.2f}"
                elif relation == "spatial" and "spatial" in attrs:
                    label += f"\nSpatial: {attrs['spatial']}"
                
                edge_text.append(label)
        
        # Set color
        color = edge_colors.get(relation, edge_colors["unknown"])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line=dict(width=2, color=color),
            hoverinfo='none',
            showlegend=True,
            name=f"{relation.title()} Edge"
        )
        
        edge_traces.append(edge_trace)
    
    # Create node traces
    node_traces = []
    
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
    
    # Create node traces for each node type
    for node_type, nodes in nodes_by_type.items():
        x_nodes = []
        y_nodes = []
        node_text = []
        node_size = []
        
        for node, attrs in nodes:
            if node in pos:
                x, y = pos[node]
                x_nodes.append(x)
                y_nodes.append(y)
                
                # Create node label
                if node_type == "character":
                    label = f"Character: {attrs.get('name', node)}"
                    if "description" in attrs:
                        label += f"\nDescription: {attrs['description']}"
                elif node_type == "emotion":
                    label = f"Emotion: {attrs.get('emotion', 'unknown')}"
                    if "intensity" in attrs:
                        label += f"\nIntensity: {attrs['intensity']:.2f}"
                elif node_type == "action":
                    label = f"Action: {attrs.get('description', 'unknown')}"
                    if "timestamp" in attrs:
                        label += f"\nTime: {attrs['timestamp']:.2f}s"
                elif node_type == "speech":
                    label = f"Speech: {attrs.get('text', 'unknown')}"
                    if "speaker_id" in attrs:
                        label += f"\nSpeaker: {attrs['speaker_id']}"
                elif node_type == "object":
                    label = f"Object: {attrs.get('description', node)}"
                else:
                    label = f"{node_type.title()}: {node}"
                
                node_text.append(label)
                
                # Scale node size based on type
                base_size = 10
                if node_type == "character":
                    node_size.append(base_size * 1.5)
                elif node_type == "scene":
                    node_size.append(base_size * 2.0)
                elif node_type == "emotion":
                    # Scale by intensity
                    intensity = attrs.get("intensity", 0.5)
                    node_size.append(base_size * (0.5 + intensity))
                else:
                    node_size.append(base_size)
        
        # Get color
        color = node_type_colors.get(node_type, "#bdc3c7")
        
        # Create node trace
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers',
            marker=dict(
                size=node_size,
                color=color,
                line=dict(width=1, color='black')
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=True,
            name=f"{node_type.title()} Node"
        )
        
        node_traces.append(node_trace)
    
    # Create the figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title="Interactive Narrative Knowledge Graph",
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
    )
    
    # Save as HTML file
    fig.write_html(output_path)
    logger.info(f"Saved interactive visualization to {output_path}")

if __name__ == "__main__":
    main()