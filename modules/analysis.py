# modules/analysis.py
"""
Narrative Analysis Module for Narrative Scene Understanding.
This module analyzes the knowledge graph to extract narrative insights.
"""

import os
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional
import json

class NarrativeAnalyzer:
    """
    Analyzes narrative knowledge graphs to extract insights about character motivations,
    emotions, relationships, and narrative arcs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the narrative analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, narrative_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze a narrative knowledge graph.
        
        Args:
            narrative_graph: NetworkX DiGraph containing the narrative representation
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing narrative knowledge graph...")
        
        # Extract character information
        character_analysis = self._analyze_characters(narrative_graph)
        
        # Extract causal chains
        causal_chains = self._extract_causal_chains(narrative_graph)
        
        # Extract narrative arcs
        narrative_arcs = self._extract_narrative_arcs(narrative_graph)
        
        # Extract key events
        key_events = self._extract_key_events(narrative_graph)
        
        # Extract themes
        themes = self._extract_themes(narrative_graph)
        
        # Combine all analyses
        analysis_results = {
            "characters": character_analysis,
            "causal_chains": causal_chains,
            "narrative_arcs": narrative_arcs,
            "key_events": key_events,
            "themes": themes
        }
        
        self.logger.info("Narrative analysis complete")
        return analysis_results
    
    def _analyze_characters(self, graph: nx.DiGraph) -> Dict[str, Dict]:
        """
        Analyze character information from the knowledge graph.
        
        Args:
            graph: Narrative knowledge graph
            
        Returns:
            Dictionary mapping character IDs to character analysis
        """
        self.logger.debug("Analyzing characters...")
        
        character_analysis = {}
        
        # Find all character nodes
        character_nodes = [
            (node_id, attrs) for node_id, attrs in graph.nodes(data=True)
            if attrs.get("type") == "character" and not attrs.get("temporary", False)
        ]
        
        for char_id, attrs in character_nodes:
            # Extract basic information
            character_info = {
                "id": char_id,
                "name": attrs.get("name", char_id),
                "description": attrs.get("description", ""),
                "first_seen": attrs.get("first_seen", 0),
                "last_seen": attrs.get("last_seen", 0)
            }
            
            # Analyze character emotions
            emotions = self._analyze_character_emotions(graph, char_id)
            character_info["emotions"] = emotions
            
            # Analyze character goals
            goals = self._analyze_character_goals(graph, char_id)
            character_info["goals"] = goals
            
            # Analyze character relationships
            relationships = self._analyze_character_relationships(graph, char_id)
            character_info["relationships"] = relationships
            
            # Analyze character arc
            character_arc = self._analyze_character_arc(graph, char_id, emotions)
            character_info["arc"] = character_arc
            
            character_analysis[char_id] = character_info
        
        return character_analysis
    
    def _analyze_character_emotions(self, graph: nx.DiGraph, char_id: str) -> List[Dict]:
        """
        Analyze a character's emotions through the narrative.
        
        Args:
            graph: Narrative knowledge graph
            char_id: Character ID
            
        Returns:
            List of emotion states with timestamps
        """
        emotions = []
        
        # Find all emotion nodes connected to this character
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "emotion":
                continue
            
            # Check if this emotion belongs to the character
            for pred in graph.predecessors(node_id):
                if pred == char_id:
                    emotions.append({
                        "emotion": attrs.get("emotion", "unknown"),
                        "timestamp": attrs.get("timestamp", 0),
                        "intensity": attrs.get("intensity", 0.5)
                    })
        
        # Sort by timestamp
        emotions.sort(key=lambda x: x["timestamp"])
        
        return emotions
    
    def _analyze_character_goals(self, graph: nx.DiGraph, char_id: str) -> List[Dict]:
        """
        Analyze a character's goals through the narrative.
        
        Args:
            graph: Narrative knowledge graph
            char_id: Character ID
            
        Returns:
            List of goals with confidence scores
        """
        goals = []
        
        # Find all goal nodes connected to this character
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "goal":
                continue
            
            # Check if this goal belongs to the character
            for pred in graph.predecessors(node_id):
                if pred == char_id:
                    goals.append({
                        "description": attrs.get("description", "unknown goal"),
                        "target": attrs.get("target", None),
                        "confidence": attrs.get("confidence", 0.5)
                    })
        
        # Sort by confidence (descending)
        goals.sort(key=lambda x: x["confidence"], reverse=True)
        
        return goals
    
    def _analyze_character_relationships(self, graph: nx.DiGraph, char_id: str) -> List[Dict]:
        """
        Analyze a character's relationships with other characters.
        
        Args:
            graph: Narrative knowledge graph
            char_id: Character ID
            
        Returns:
            List of relationships with other characters
        """
        relationships = []
        
        # Check for explicit relationship edges
        for source, target, edge_attrs in graph.edges(data=True):
            if edge_attrs.get("relation") != "relationship":
                continue
            
            if source == char_id:
                # Relationship from this character to another
                relationships.append({
                    "character": target,
                    "type": edge_attrs.get("type", "unknown"),
                    "sentiment": edge_attrs.get("sentiment", 0),
                    "confidence": edge_attrs.get("confidence", 0.5)
                })
            elif target == char_id:
                # Relationship from another character to this one
                relationships.append({
                    "character": source,
                    "type": edge_attrs.get("type", "unknown"),
                    "sentiment": edge_attrs.get("sentiment", 0),
                    "confidence": edge_attrs.get("confidence", 0.5)
                })
        
        # Sort by confidence (descending)
        relationships.sort(key=lambda x: x["confidence"], reverse=True)
        
        return relationships
    
    def _analyze_character_arc(self, graph: nx.DiGraph, char_id: str, emotions: List[Dict]) -> Dict:
        """
        Analyze a character's arc through the narrative.
        
        Args:
            graph: Narrative knowledge graph
            char_id: Character ID
            emotions: Pre-computed emotions for this character
            
        Returns:
            Character arc information
        """
        # Default values
        arc = {
            "type": "flat",
            "confidence": 0.5,
            "start_state": None,
            "end_state": None,
            "key_moments": []
        }
        
        # Need at least two emotions to determine an arc
        if len(emotions) < 2:
            return arc
        
        # Get starting and ending emotional states
        start_state = emotions[0]["emotion"]
        end_state = emotions[-1]["emotion"]
        
        arc["start_state"] = start_state
        arc["end_state"] = end_state
        
        # Analyze emotional trajectory
        if start_state != end_state:
            # Map emotions to valence values
            valence_map = {
                "happy": 1.0,
                "content": 0.5,
                "neutral": 0.0,
                "sad": -0.5,
                "angry": -0.7,
                "afraid": -0.8,
                "disgusted": -0.6,
                "surprised": 0.2
            }
            
            # Get valence values if available
            start_valence = valence_map.get(start_state, 0)
            end_valence = valence_map.get(end_state, 0)
            
            # Determine arc type
            if end_valence > start_valence + 0.5:
                arc["type"] = "positive"
                arc["confidence"] = 0.7
            elif end_valence < start_valence - 0.5:
                arc["type"] = "negative"
                arc["confidence"] = 0.7
            else:
                arc["type"] = "complex"
                arc["confidence"] = 0.6
        
        # Identify key moments in the character's arc
        # These are points where the character's emotional state changes significantly
        key_moments = []
        for i in range(1, len(emotions)):
            if emotions[i]["emotion"] != emotions[i-1]["emotion"]:
                key_moments.append({
                    "timestamp": emotions[i]["timestamp"],
                    "from_emotion": emotions[i-1]["emotion"],
                    "to_emotion": emotions[i]["emotion"]
                })
        
        arc["key_moments"] = key_moments
        
        return arc
    
    def _extract_causal_chains(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Extract causal chains from the knowledge graph.
        
        Args:
            graph: Narrative knowledge graph
            
        Returns:
            List of causal chains
        """
        self.logger.debug("Extracting causal chains...")
        
        causal_chains = []
        
        # Find all causal edges
        causal_edges = [
            (source, target, attrs) for source, target, attrs in graph.edges(data=True)
            if attrs.get("relation") == "causes"
        ]
        
        # Group edges into chains
        chains = []
        processed_edges = set()
        
        for source, target, attrs in causal_edges:
            if (source, target) in processed_edges:
                continue
            
            # Start a new chain
            chain = [(source, target, attrs)]
            processed_edges.add((source, target))
            
            # Extend chain forward
            current_target = target
            while True:
                # Find next causal edge
                next_edge = None
                for s, t, a in causal_edges:
                    if s == current_target and (s, t) not in processed_edges:
                        next_edge = (s, t, a)
                        break
                
                if next_edge:
                    source, target, attrs = next_edge
                    chain.append((source, target, attrs))
                    processed_edges.add((source, target))
                    current_target = target
                else:
                    break
            
            chains.append(chain)
        
        # Convert chains to structured format
        for chain in chains:
            if not chain:
                continue
            
            events = []
            for i, (source, target, attrs) in enumerate(chain):
                # Get source event details
                if i == 0:
                    source_attrs = graph.nodes[source]
                    source_type = source_attrs.get("type", "unknown")
                    
                    source_event = {
                        "id": source,
                        "type": source_type
                    }
                    
                    if source_type == "action":
                        source_event["description"] = source_attrs.get("description", "")
                    elif source_type == "speech":
                        source_event["text"] = source_attrs.get("text", "")
                    
                    events.append(source_event)
                
                # Get target event details
                target_attrs = graph.nodes[target]
                target_type = target_attrs.get("type", "unknown")
                
                target_event = {
                    "id": target,
                    "type": target_type
                }
                
                if target_type == "action":
                    target_event["description"] = target_attrs.get("description", "")
                elif target_type == "speech":
                    target_event["text"] = target_attrs.get("text", "")
                elif target_type == "emotion":
                    target_event["emotion"] = target_attrs.get("emotion", "")
                    
                    # Find the character experiencing this emotion
                    for pred in graph.predecessors(target):
                        if graph.nodes[pred].get("type") == "character":
                            target_event["character"] = pred
                            break
                
                events.append(target_event)
            
            # Add causal chain if it has at least two events
            if len(events) >= 2:
                causal_chains.append({
                    "events": events,
                    "confidence": sum(attrs.get("probability", 0.5) for _, _, attrs in chain) / len(chain)
                })
        
        # Sort by confidence (descending)
        causal_chains.sort(key=lambda x: x["confidence"], reverse=True)
        
        return causal_chains
    
    def _extract_narrative_arcs(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Extract narrative arcs from the knowledge graph.
        
        Args:
            graph: Narrative knowledge graph
            
        Returns:
            List of narrative arcs
        """
        self.logger.debug("Extracting narrative arcs...")
        
        # Find scenes
        scenes = [
            (node_id, attrs) for node_id, attrs in graph.nodes(data=True)
            if attrs.get("type") == "scene"
        ]
        
        # Sort scenes by timestamp
        scenes.sort(key=lambda x: x[1].get("timestamp", 0))
        
        # Group scenes into acts based on emotional intensity and causal connections
        acts = []
        current_act = []
        
        for scene_id, scene_attrs in scenes:
            # Add scene to current act
            current_act.append((scene_id, scene_attrs))
            
            # Check if this scene is a potential act boundary
            is_boundary = False
            
            # Look for significant emotional changes
            emotions_in_scene = []
            for node_id, attrs in graph.nodes(data=True):
                if (attrs.get("type") == "emotion" and 
                    graph.has_edge(scene_id, node_id)):
                    emotions_in_scene.append(attrs.get("intensity", 0.5))
            
            # High emotional intensity suggests an act boundary
            if emotions_in_scene and max(emotions_in_scene, default=0) > 0.8:
                is_boundary = True
            
            if is_boundary and current_act:
                acts.append(current_act)
                current_act = []
        
        # Add any remaining scenes to the final act
        if current_act:
            acts.append(current_act)
        
        # Convert acts to narrative arcs
        narrative_arcs = []
        
        for i, act in enumerate(acts):
            arc_type = "exposition" if i == 0 else "climax" if i == len(acts) - 1 else "development"
            
            # Get time range
            start_time = act[0][1].get("timestamp", 0)
            end_time = act[-1][1].get("timestamp", 0)
            
            # Get key characters in this act
            characters_in_act = set()
            for scene_id, _ in act:
                for node_id, attrs in graph.nodes(data=True):
                    if (attrs.get("type") == "character" and 
                        graph.has_edge(scene_id, node_id)):
                        characters_in_act.add(node_id)
            
            narrative_arcs.append({
                "type": arc_type,
                "start_time": start_time,
                "end_time": end_time,
                "scenes": [scene_id for scene_id, _ in act],
                "key_characters": list(characters_in_act)
            })
        
        return narrative_arcs
    
    def _extract_key_events(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Extract key events from the knowledge graph.
        
        Args:
            graph: Narrative knowledge graph
            
        Returns:
            List of key events
        """
        self.logger.debug("Extracting key events...")
        
        # Collect all events (actions, speeches, emotions)
        events = []
        
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") in ["action", "speech", "emotion"]:
                event_type = attrs.get("type")
                
                # Basic event information
                event = {
                    "id": node_id,
                    "type": event_type,
                    "timestamp": attrs.get("timestamp", attrs.get("start_time", 0))
                }
                
                # Add type-specific information
                if event_type == "action":
                    event["description"] = attrs.get("description", "")
                    event["action_type"] = attrs.get("action_type", "unknown")
                    
                    # Find subject and object
                    for pred in graph.predecessors(node_id):
                        if graph.nodes[pred].get("type") == "character":
                            event["subject"] = pred
                            break
                    
                    for succ in graph.successors(node_id):
                        if graph.nodes[succ].get("type") in ["character", "object"]:
                            event["object"] = succ
                            break
                
                elif event_type == "speech":
                    event["text"] = attrs.get("text", "")
                    event["speaker"] = attrs.get("speaker", "unknown")
                    
                    # Find speaker
                    for pred in graph.predecessors(node_id):
                        if graph.nodes[pred].get("type") == "character":
                            event["speaker"] = pred
                            break
                
                elif event_type == "emotion":
                    event["emotion"] = attrs.get("emotion", "unknown")
                    event["intensity"] = attrs.get("intensity", 0.5)
                    
                    # Find character
                    for pred in graph.predecessors(node_id):
                        if graph.nodes[pred].get("type") == "character":
                            event["character"] = pred
                            break
                
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        
        # Determine key events based on:
        # 1. Participation in causal chains
        # 2. Emotional significance
        # 3. Relationship to character goals
        
        # Count event references in causal chains
        event_importance = {event["id"]: 0 for event in events}
        
        for source, target, attrs in graph.edges(data=True):
            if attrs.get("relation") == "causes":
                event_importance[source] = event_importance.get(source, 0) + 1
                event_importance[target] = event_importance.get(target, 0) + 1
        
        # Add importance for high emotional intensity
        for event in events:
            if event["type"] == "emotion" and event.get("intensity", 0) > 0.7:
                event_importance[event["id"]] = event_importance.get(event["id"], 0) + 2
        
        # Add importance for events related to character goals
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") == "goal":
                target = attrs.get("target")
                if target in event_importance:
                    event_importance[target] = event_importance.get(target, 0) + 2
        
        # Select top events based on importance
        key_events = []
        for event in events:
            importance = event_importance.get(event["id"], 0)
            
            # Add key event if importance is above threshold or it's emotionally significant
            if importance >= 2 or (event["type"] == "emotion" and event.get("intensity", 0) > 0.8):
                event["importance"] = importance
                key_events.append(event)
        
        # Limit to a reasonable number
        max_events = 10
        if len(key_events) > max_events:
            key_events.sort(key=lambda x: x.get("importance", 0), reverse=True)
            key_events = key_events[:max_events]
        
        # Sort by timestamp for final result
        key_events.sort(key=lambda x: x["timestamp"])
        
        return key_events
    
    def _extract_themes(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Extract themes from the knowledge graph.
        
        Args:
            graph: Narrative knowledge graph
            
        Returns:
            List of themes with confidence scores
        """
        self.logger.debug("Extracting themes...")
        
        # In a real implementation, this would use more sophisticated theme extraction
        # Here we'll use a simple keyword-based approach
        
        # Define potential themes and their associated keywords
        theme_keywords = {
            "conflict": ["angry", "fight", "argument", "disagreement", "tension", "confrontation"],
            "romance": ["love", "kiss", "embrace", "affection", "romantic", "relationship"],
            "suspense": ["afraid", "nervous", "tense", "suspicious", "mystery", "unknown"],
            "discovery": ["find", "discover", "reveal", "uncover", "learn", "realize"],
            "transformation": ["change", "transform", "grow", "evolve", "different", "new"],
            "loss": ["sad", "grief", "lose", "missing", "gone", "departed"],
            "triumph": ["happy", "victory", "succeed", "overcome", "achieve", "win"],
            "betrayal": ["trust", "betray", "deceive", "lie", "false", "trick"]
        }
        
        # Count theme occurrences
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        # Analyze text from speech, descriptions, and emotions
        text_sources = []
        
        # Add speech text
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") == "speech":
                text_sources.append(attrs.get("text", "").lower())
        
        # Add action descriptions
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") == "action":
                text_sources.append(attrs.get("description", "").lower())
        
        # Add emotions
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") == "emotion":
                text_sources.append(attrs.get("emotion", "").lower())
        
        # Count keyword occurrences
        for text in text_sources:
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        theme_counts[theme] += 1
        
        # Calculate theme confidence based on occurrence count
        themes = []
        total_occurrences = sum(theme_counts.values())
        
        if total_occurrences > 0:
            for theme, count in theme_counts.items():
                if count > 0:
                    confidence = min(0.9, 0.3 + 0.6 * count / total_occurrences)
                    themes.append({
                        "name": theme,
                        "confidence": confidence,
                        "keyword_matches": count
                    })
        
        # Sort by confidence (descending)
        themes.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Limit to top themes
        max_themes = 5
        themes = themes[:max_themes]
        
        return themes