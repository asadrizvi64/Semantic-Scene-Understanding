
class NarrativeGraphBuilder:
    """
    Builds and maintains a narrative knowledge graph from visual, audio, and OCR data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # existing initialization code...
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.graph = nx.DiGraph()


import os
import logging
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Set
import time
from collections import defaultdict
import json
from datetime import datetime

class EnhancedNarrativeGraphBuilder:
    """
    Builds and maintains a comprehensive narrative knowledge graph from visual, audio, and OCR data.
    Combines functionality from multiple implementations for improved narrative understanding.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the narrative graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize the graph
        self.graph = nx.DiGraph()
        
        # Entity tracking
        self.character_identities = {}  # Map track_id to character_id
        self.object_identities = {}     # Map track_id to object_id
        self.text_identities = {}       # Map track_id to text_id
        
        # Feature embeddings for entity resolution
        self.character_embeddings = {}  # Map character_id to facial features
        
        # Space-time tracking
        self.current_scene_id = None
        self.current_time = 0.0
        
        # Configuration parameters
        self.spatial_distance_threshold = config.get("spatial_distance_threshold", 50)
        self.character_face_similarity_threshold = config.get("face_similarity_threshold", 0.6)
        self.character_description_similarity_threshold = config.get("description_similarity_threshold", 0.8)
        self.causal_window = config.get("causal_window", 2.0)  # seconds
    
    def _query_event_sequence(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Gets a chronological sequence of events within a specified time period.
        Returns a dict with start_time, end_time, and a list of events sorted by timestamp.
        Each event includes its id, type, timestamp, and full attribute dict.
        """
        events = []
        for node_id, attrs in self.graph.nodes(data=True):
            event_type = attrs.get("type")
            if event_type in ["action", "speech", "sound", "emotion"]:
                # Determine timestamp field
                ts = attrs.get("timestamp") or attrs.get("start_time") or attrs.get("end_time")
                if ts is None:
                    continue
                if start_time <= ts <= end_time:
                    events.append({
                        "event_id": node_id,
                        "type": event_type,
                        "timestamp": ts,
                        "attributes": attrs
                    })
        # Sort events chronologically
        events.sort(key=lambda e: e["timestamp"])
        return {"start_time": start_time, "end_time": end_time, "events": events}

    # def _query_causal_chain(self, event_id: str) -> Dict[str, Any]:
    #     """
    #     Gets a causal chain starting from a specific event, showing both causes and effects.
    #     Returns a dict with the event_id, list of predecessor events (causes), and successor events (effects).
    #     """
    #     causes = []
    #     for pred in self.graph.predecessors(event_id):
    #         edge = self.graph.get_edge_data(pred, event_id)
    #         if edge.get("relation") == "causes":
    #             causes.append({"event_id": pred, "edge_data": edge})

    #     effects = []
    #     for succ in self.graph.successors(event_id):
    #         edge = self.graph.get_edge_data(event_id, succ)
    #         if edge.get("relation") == "causes":
    #             effects.append({"event_id": succ, "edge_data": edge})

    #     return {"event_id": event_id, "causes": causes, "effects": effects}
    
    def _query_causal_chain(self, event_id: str) -> Dict[str, Any]:
        """
        Get a causal chain starting from a specific event.
        
        Args:
            event_id: ID of the starting event
            
        Returns:
            Dictionary with causal chain
        """
        if event_id not in self.graph.nodes:
            raise ValueError(f"Event not found: {event_id}")
        
        # Build chain forward and backward
        forward_chain = self._build_causal_chain_forward(event_id, max_depth=5)
        backward_chain = self._build_causal_chain_backward(event_id, max_depth=5)
        
        # Combine chains
        chain = {
            "root_event": self._format_event(event_id),
            "causes": backward_chain,
            "effects": forward_chain
        }
        
        return chain
    
    def _build_causal_chain_forward(self, event_id: str, max_depth: int = 5, depth: int = 0) -> List[Dict[str, Any]]:
        """
        Recursively build a forward causal chain (effects).
        
        Args:
            event_id: Starting event ID
            max_depth: Maximum recursion depth
            depth: Current depth
            
        Returns:
            List of effect events
        """
        if depth >= max_depth:
            return []
        
        effects = []
        
        # Get direct effects
        for effect_id in self.graph.successors(event_id):
            edge_data = self.graph.get_edge_data(event_id, effect_id)
            if edge_data.get("relation") != "causes":
                continue
            
            # Format event
            effect = self._format_event(effect_id)
            effect["probability"] = edge_data.get("probability", 0.5)
            
            # Recursively get this effect's effects
            next_effects = self._build_causal_chain_forward(effect_id, max_depth, depth + 1)
            
            if next_effects:
                effect["effects"] = next_effects
            
            effects.append(effect)
        
        return effects
    
    def _build_causal_chain_backward(self, event_id: str, max_depth: int = 5, depth: int = 0) -> List[Dict[str, Any]]:
        """
        Recursively build a backward causal chain (causes).
        
        Args:
            event_id: Starting event ID
            max_depth: Maximum recursion depth
            depth: Current depth
            
        Returns:
            List of causal events
        """
        if depth >= max_depth:
            return []
        
        causes = []
        
        # Get direct causes
        for cause_id in self.graph.predecessors(event_id):
            edge_data = self.graph.get_edge_data(cause_id, event_id)
            if edge_data.get("relation") != "causes":
                continue
            
            # Format event
            cause = self._format_event(cause_id)
            cause["probability"] = edge_data.get("probability", 0.5)
            
            # Recursively get this cause's causes
            prior_causes = self._build_causal_chain_backward(cause_id, max_depth, depth + 1)
            
            if prior_causes:
                cause["causes"] = prior_causes
            
            causes.append(cause)
        
        return causes
    
    def _format_event(self, event_id: str) -> Dict[str, Any]:
        """
        Format an event for inclusion in query results.
        
        Args:
            event_id: Event ID
            
        Returns:
            Formatted event dictionary
        """
        attrs = self.graph.nodes[event_id]
        node_type = attrs.get("type")
        
        event = {
            "id": event_id,
            "type": node_type
        }
        
        if node_type == "action":
            event["description"] = attrs.get("description", "")
            event["timestamp"] = attrs.get("timestamp")
            
            # Get actor
            actor = None
            for pred in self.graph.predecessors(event_id):
                if self.graph[pred][event_id].get("relation") == "performs":
                    actor_type = self.graph.nodes[pred].get("type")
                    actor = {
                        "id": pred,
                        "type": actor_type,
                        "name": self.graph.nodes[pred].get("name", ""),
                        "description": self.graph.nodes[pred].get("description", "")
                    }
                    break
            
            event["actor"] = actor
            
            # Get target
            target = None
            for succ in self.graph.successors(event_id):
                if self.graph[event_id][succ].get("relation") == "affects":
                    target_type = self.graph.nodes[succ].get("type")
                    target = {
                        "id": succ,
                        "type": target_type,
                        "name": self.graph.nodes[succ].get("name", ""),
                        "description": self.graph.nodes[succ].get("description", "")
                    }
                    break
            
            event["target"] = target
        
        elif node_type == "speech":
            event["text"] = attrs.get("text", "")
            event["start_time"] = attrs.get("start_time")
            event["end_time"] = attrs.get("end_time")
            event["sentiment"] = attrs.get("sentiment", 0.0)
            
            # Get speaker
            speaker = None
            for pred in self.graph.predecessors(event_id):
                if self.graph[pred][event_id].get("relation") == "speaks":
                    speaker = {
                        "id": pred,
                        "name": self.graph.nodes[pred].get("name", ""),
                        "description": self.graph.nodes[pred].get("description", "")
                    }
                    break
            
            event["speaker"] = speaker
        
        elif node_type == "emotion":
            event["emotion"] = attrs.get("emotion", "")
            event["intensity"] = attrs.get("intensity", 0.5)
            event["timestamp"] = attrs.get("timestamp")
            
            # Get character
            character = None
            for pred in self.graph.predecessors(event_id):
                if self.graph.nodes[pred].get("type") == "character":
                    character = {
                        "id": pred,
                        "name": self.graph.nodes[pred].get("name", ""),
                        "description": self.graph.nodes[pred].get("description", "")
                    }
                    break
            
            event["character"] = character
        
        return event
        
    def _query_spatial_relationships(self, entity_id: str, timestamp: float = None) -> Dict[str, Any]:
        """
        Get spatial relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            timestamp: Optional timestamp to get relationships at a specific time
            
        Returns:
            Dictionary with spatial relationship information
        """
        if entity_id not in self.graph.nodes:
            raise ValueError(f"Entity not found: {entity_id}")
        
        node = self.graph.nodes[entity_id]
        entity_type = node.get("type")
        
        if entity_type not in ["character", "object"]:
            raise ValueError(f"Entity is not a character or object: {entity_id}")
        
        # Basic entity info
        info = {
            "id": entity_id,
            "type": entity_type,
            "name": node.get("name", ""),
            "description": node.get("description", ""),
            "relationships": []
        }
        
        # Get explicit spatial relationships
        for related_id in self.graph.successors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, related_id)
            if edge_data.get("relation") != "spatial":
                continue
            
            related_node = self.graph.nodes[related_id]
            related_type = related_node.get("type")
            
            if timestamp is not None:
                # Check if relationship is valid at the specified time
                first_seen = edge_data.get("first_seen", 0.0)
                last_updated = edge_data.get("last_updated", 0.0)
                
                if timestamp < first_seen or timestamp > last_updated:
                    continue
            
            relationship = {
                "id": related_id,
                "type": related_type,
                "name": related_node.get("name", ""),
                "description": related_node.get("description", ""),
                "spatial": edge_data.get("spatial", ""),
                "first_seen": edge_data.get("first_seen"),
                "last_updated": edge_data.get("last_updated")
            }
            
            info["relationships"].append(relationship)
        
        for related_id in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(related_id, entity_id)
            if edge_data.get("relation") != "spatial":
                continue
            
            related_node = self.graph.nodes[related_id]
            related_type = related_node.get("type")
            
            if timestamp is not None:
                # Check if relationship is valid at the specified time
                first_seen = edge_data.get("first_seen", 0.0)
                last_updated = edge_data.get("last_updated", 0.0)
                
                if timestamp < first_seen or timestamp > last_updated:
                    continue
            
            # Invert the spatial relationship
            spatial_rel = edge_data.get("spatial", "")
            inverted_rel = ""
            
            if spatial_rel == "left_of":
                inverted_rel = "right_of"
            elif spatial_rel == "right_of":
                inverted_rel = "left_of"
            elif spatial_rel == "above":
                inverted_rel = "below"
            elif spatial_rel == "below":
                inverted_rel = "above"
            else:
                inverted_rel = spatial_rel  # "near" stays the same
            
            relationship = {
                "id": related_id,
                "type": related_type,
                "name": related_node.get("name", ""),
                "description": related_node.get("description", ""),
                "spatial": inverted_rel,
                "first_seen": edge_data.get("first_seen"),
                "last_updated": edge_data.get("last_updated")
            }
            
            info["relationships"].append(relationship)
        
        # If timestamp is specified and no explicit relationships,
        # infer from positions at that time
        if timestamp is not None and not info["relationships"]:
            # Get entity position at the specified time
            entity_positions = node.get("positions", [])
            
            # Find the closest position in time
            closest_position = None
            closest_time_diff = float('inf')
            
            for pos in entity_positions:
                pos_time = pos.get("timestamp", 0.0)
                time_diff = abs(pos_time - timestamp)
                
                if time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_position = pos
            
            # If found a position within 1 second of the requested time
            if closest_position and closest_time_diff < 1.0:
                entity_box = closest_position.get("box")
                
                if entity_box:
                    # Look for other entities with positions at similar time
                    for other_id, other_attrs in self.graph.nodes(data=True):
                        if other_id == entity_id:
                            continue
                        
                        if other_attrs.get("type") not in ["character", "object"]:
                            continue
                        
                        other_positions = other_attrs.get("positions", [])
                        
                        for other_pos in other_positions:
                            other_time = other_pos.get("timestamp", 0.0)
                            other_box = other_pos.get("box")
                            
                            if other_box and abs(other_time - timestamp) < 1.0:
                                # Calculate spatial relationship
                                spatial_rel = self._determine_spatial_relationship(entity_box, other_box)
                                
                                if spatial_rel:
                                    relationship = {
                                        "id": other_id,
                                        "type": other_attrs.get("type"),
                                        "name": other_attrs.get("name", ""),
                                        "description": other_attrs.get("description", ""),
                                        "spatial": spatial_rel,
                                        "inferred": True,
                                        "timestamp": other_time
                                    }
                                    
                                    info["relationships"].append(relationship)
        
        return info
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'EnhancedNarrativeGraphBuilder':
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            EnhancedNarrativeGraphBuilder instance with loaded graph
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a new instance
        builder = cls()
        
        # Create a new graph
        builder.graph = nx.DiGraph()
        
        # Add nodes
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            builder.graph.add_node(node_id, **node)
        
        # Add edges
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            builder.graph.add_edge(source, target, **edge)
        
        # Rebuild identity maps
        builder._rebuild_identity_maps()
        
        return builder
    
    def _rebuild_identity_maps(self):
        """Rebuild identity maps from graph nodes."""
        self.character_identities = {}
        self.object_identities = {}
        self.text_identities = {}
        
        # Find tracking information from node IDs and attributes
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type")
            
            if node_type == "character":
                # Extract track_id from positions if available
                for pos in attrs.get("positions", []):
                    if "track_id" in pos:
                        self.character_identities[pos["track_id"]] = node_id
            
            elif node_type == "object":
                # Extract track_id from positions if available
                for pos in attrs.get("positions", []):
                    if "track_id" in pos:
                        self.object_identities[pos["track_id"]] = node_id
            
            elif node_type == "text":
                # Extract track_id from positions if available
                for pos in attrs.get("positions", []):
                    if "track_id" in pos:
                        self.text_identities[pos["track_id"]] = node_id
    
    def merge_with(self, other_graph: nx.DiGraph) -> None:
            """
            Merge another graph into this one, avoiding ID conflicts and resolving entities.
            Args:
                other_graph: NetworkX DiGraph to merge
            """
            node_mapping: Dict[str, str] = {}
            # Merge nodes
            for node_id, attrs in other_graph.nodes(data=True):
                if node_id in self.graph.nodes:
                    new_id = f"{node_id}_merged_{len(node_mapping)}"
                    node_mapping[node_id] = new_id
                    self.graph.add_node(new_id, **attrs)
                else:
                    node_mapping[node_id] = node_id
                    self.graph.add_node(node_id, **attrs)
            # Merge edges
            for u, v, attrs in other_graph.edges(data=True):
                mapped_u = node_mapping[u]
                mapped_v = node_mapping[v]
                self.graph.add_edge(mapped_u, mapped_v, **attrs)
            # Resolve any duplicate entities across merged data
            self._resolve_entities()
        
    def _query_character_info(self, character_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a character.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Dictionary with character information
        """
        if character_id not in self.graph.nodes:
            raise ValueError(f"Character not found: {character_id}")
        
        """
Enhanced Knowledge Graph Module for Narrative Scene Understanding.
This module builds a comprehensive narrative knowledge graph from multi-modal processing results,
combining features from multiple implementations for improved narrative understanding.
"""
    def build_graph(self, 
                   visual_data: List[Dict] = None, 
                   audio_data: List[Dict] = None, 
                   ocr_data: List[Dict] = None,
                   scene_boundaries: List[float] = None,
                   scene_data: Dict[str, Any] = None) -> nx.DiGraph:
        """
        Build a narrative knowledge graph from processed data.
        
        Args:
            visual_data: List of visual processing results by frame
            audio_data: List of audio processing results
            ocr_data: List of OCR processing results by frame
            scene_boundaries: List of scene boundary timestamps
            scene_data: Consolidated scene data dictionary (alternative input method)
            
        Returns:
            NetworkX DiGraph containing the narrative knowledge graph
        """
        self.logger.info("Building enhanced narrative knowledge graph...")
        
        # Start with a fresh graph
        self.graph = nx.DiGraph()
        
        # Handle different input formats
        if scene_data is not None:
            # Extract data from scene_data format
            scene_boundaries = scene_data.get("scene_boundaries", [0.0])
            visual_data = scene_data.get("frames", [])
            audio_data = self._extract_audio_data(scene_data)
            ocr_data = self._extract_ocr_data(scene_data)
        
        # Step 1: Add scene boundaries and time nodes
        self._add_scene_boundaries(scene_boundaries or [0.0])
        
        # Step 2: Process visual data (characters, objects, actions)
        if visual_data:
            self._process_visual_data(visual_data)
        
        # Step 3: Process audio data (speech, non-speech sounds)
        if audio_data:
            self._process_audio_data(audio_data)
        
        # Step 4: Process OCR data (text in frames)
        if ocr_data:
            self._process_ocr_data(ocr_data)
        
        # Step 5: Perform entity resolution (merge identities)
        self._resolve_entities()
        
        # Step 6: Infer relationships
        self._infer_relationships()
        
        # Step 7: Infer causal relationships
        self._infer_causal_relationships()
        
        # Step 8: Infer character goals
        self._infer_character_goals()
        
        # Step 9: Infer emotional responses
        self._infer_emotional_responses()
        
        self.logger.info(f"Completed graph construction: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        
        return self.graph
    
    def _extract_audio_data(self, scene_data: Dict[str, Any]) -> List[Dict]:
        """
        Extract audio data from scene_data format.
        
        Args:
            scene_data: Consolidated scene data
            
        Returns:
            List of audio data segments
        """
        audio_data = []
        
        # Add speech segments
        for i, segment in enumerate(scene_data.get("speech", [])):
            audio_segment = dict(segment)
            audio_segment["type"] = "speech"
            audio_data.append(audio_segment)
        
        # Add non-speech segments
        for i, segment in enumerate(scene_data.get("non_speech", [])):
            audio_segment = dict(segment)
            audio_segment["type"] = "non-speech"
            audio_data.append(audio_segment)
        
        return audio_data
    
    def _extract_ocr_data(self, scene_data: Dict[str, Any]) -> List[Dict]:
        """
        Extract OCR data from scene_data format.
        
        Args:
            scene_data: Consolidated scene data
            
        Returns:
            List of OCR data by frame
        """
        ocr_data = []
        
        for ocr_result in scene_data.get("ocr", []):
            # Get frame index and find corresponding frame
            frame_idx = ocr_result.get("frame_idx", 0)
            timestamp = None
            
            # Find timestamp for this frame
            for frame in scene_data.get("frames", []):
                if frame.get("frame_idx") == frame_idx:
                    timestamp = frame.get("timestamp", 0.0)
                    break
            
            if timestamp is None:
                continue
            
            # Find or create frame entry in OCR data
            frame_entry = None
            for entry in ocr_data:
                if entry.get("timestamp") == timestamp:
                    frame_entry = entry
                    break
            
            if frame_entry is None:
                frame_entry = {"timestamp": timestamp, "text_detections": []}
                ocr_data.append(frame_entry)
            
            # Add text detection
            text_det = {
                "text": ocr_result.get("text", ""),
                "confidence": ocr_result.get("confidence", 0.5),
                "box": ocr_result.get("box")
            }
            
            frame_entry["text_detections"].append(text_det)
        
        return ocr_data
    
    def _add_scene_boundaries(self, scene_boundaries: List[float]):
        """
        Add scene boundary nodes to the graph.
        
        Args:
            scene_boundaries: List of scene boundary timestamps
        """
        self.logger.debug(f"Adding {len(scene_boundaries)} scene boundaries")
        
        # Create scene nodes and connect them temporally
        prev_scene_id = None
        
        for i, timestamp in enumerate(scene_boundaries):
            scene_id = f"scene_{i}"
            
            # Add scene node
            self.graph.add_node(
                scene_id,
                type="scene",
                timestamp=timestamp,
                index=i
            )
            
            # Add time node
            time_id = f"time_{i}"
            self.graph.add_node(
                time_id,
                type="time",
                timestamp=timestamp
            )
            
            # Connect scene to time
            self.graph.add_edge(
                scene_id,
                time_id,
                relation="has_time"
            )
            
            # Connect to previous scene (if any)
            if prev_scene_id is not None:
                self.graph.add_edge(
                    prev_scene_id,
                    scene_id,
                    relation="precedes"
                )
            
            prev_scene_id = scene_id
    
    def _process_visual_data(self, visual_data: List[Dict]):
        """
        Process visual data frames to extract characters, objects, and actions.
        
        Args:
            visual_data: List of visual processing results by frame
        """
        self.logger.debug(f"Processing {len(visual_data)} visual frames")
        
        for frame_data in visual_data:
            timestamp = frame_data.get("timestamp", 0.0)
            
            # Find the current scene
            scene_id = self._find_current_scene(timestamp)
            if scene_id is None:
                continue
            
            # Process overall frame caption
            caption = frame_data.get("overall_caption", "")
            if caption:
                caption_id = f"caption_{scene_id}_{timestamp}"
                self.graph.add_node(
                    caption_id,
                    type="caption",
                    text=caption,
                    timestamp=timestamp
                )
                self.graph.add_edge(scene_id, caption_id, relation="contains")
            
            # Process objects (characters and physical objects)
            self._process_objects(frame_data.get("objects", []), timestamp, scene_id)
            
            # Process actions
            self._process_actions(frame_data.get("actions", []), timestamp, scene_id)
            
            # Process faces
            self._process_faces(frame_data.get("faces", []), frame_data.get("objects", []), timestamp, scene_id)
            
            # Process spatial relationships between entities
            self._process_spatial_relationships(frame_data, timestamp, scene_id)
    
    def _process_objects(self, objects: List[Dict], timestamp: float, scene_id: str):
        """
        Process detected objects (characters and physical objects).
        
        Args:
            objects: List of detected objects
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        for obj in objects:
            obj_id = obj.get("id")
            obj_type = obj.get("type")
            
            if obj_type == "person":
                self._process_person(obj, timestamp, scene_id)
            else:
                self._process_physical_object(obj, timestamp, scene_id)
    
    def _process_person(self, person: Dict, timestamp: float, scene_id: str):
        """
        Process a detected person as a character.
        
        Args:
            person: Person object data
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        track_id = person.get("track_id")
        
        if track_id is None:
            # Not tracking this person yet
            return
        
        # Check if this track is already mapped to a character
        if track_id in self.character_identities:
            character_id = self.character_identities[track_id]
        else:
            # Create a new character ID
            character_id = f"character_{len(self.character_identities) + 1}"
            self.character_identities[track_id] = character_id
            
            # Create a new character node
            self.graph.add_node(
                character_id,
                type="character",
                first_seen=timestamp,
                last_seen=timestamp,
                description=person.get("caption", ""),
                name=person.get("name", ""),
                positions=[],
                temporary=False
            )
            
            # Connect to the scene
            self.graph.add_edge(scene_id, character_id, relation="contains")
        
        # Update character properties
        if character_id in self.graph.nodes:
            # Update last seen
            self.graph.nodes[character_id]["last_seen"] = timestamp
            
            # Update positions
            position = {
                "timestamp": timestamp,
                "box": person.get("box"),
                "scene_id": scene_id
            }
            self.graph.nodes[character_id]["positions"].append(position)
            
            # Store face features if available
            if "face" in person and "features" in person:
                self.character_embeddings[character_id] = person["features"]
    
    def _process_physical_object(self, obj: Dict, timestamp: float, scene_id: str):
        """
        Process a detected physical object.
        
        Args:
            obj: Object data
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        track_id = obj.get("track_id")
        
        # If not tracked, create a temporary object
        if track_id is None:
            obj_id = f"object_temp_{obj.get('id', timestamp)}"
            temporary = True
        elif track_id in self.object_identities:
            obj_id = self.object_identities[track_id]
            temporary = False
        else:
            obj_id = f"object_{len(self.object_identities) + 1}"
            self.object_identities[track_id] = obj_id
            temporary = False
        
        # Create or update object node
        if obj_id not in self.graph.nodes:
            self.graph.add_node(
                obj_id,
                type="object",
                first_seen=timestamp,
                last_seen=timestamp,
                description=obj.get("caption", ""),
                positions=[],
                temporary=temporary
            )
            
            # Add attributes like color if available
            if "color" in obj:
                self.graph.nodes[obj_id]["color"] = obj["color"]
            
            if "size" in obj:
                self.graph.nodes[obj_id]["size"] = obj["size"]
                
            if "material" in obj:
                self.graph.nodes[obj_id]["material"] = obj["material"]
            
            # Connect to the scene
            self.graph.add_edge(scene_id, obj_id, relation="contains")
        
        # Update object properties
        if obj_id in self.graph.nodes:
            # Update last seen
            self.graph.nodes[obj_id]["last_seen"] = timestamp
            
            # Update positions
            position = {
                "timestamp": timestamp,
                "box": obj.get("box"),
                "scene_id": scene_id
            }
            self.graph.nodes[obj_id]["positions"].append(position)
            
            # Update attributes if available
            if "color" in obj and "color" not in self.graph.nodes[obj_id]:
                self.graph.nodes[obj_id]["color"] = obj["color"]
            
            if "size" in obj and "size" not in self.graph.nodes[obj_id]:
                self.graph.nodes[obj_id]["size"] = obj["size"]
                
            if "material" in obj and "material" not in self.graph.nodes[obj_id]:
                self.graph.nodes[obj_id]["material"] = obj["material"]
    
    def _process_actions(self, actions: List[Dict], timestamp: float, scene_id: str):
        """
        Process detected actions.
        
        Args:
            actions: List of actions
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        for i, action in enumerate(actions):
            action_id = f"action_{timestamp}_{i}"
            action_type = action.get("type", "unknown")
            subject_id = action.get("subject_id")
            subject_type = action.get("subject_type", "unknown")
            object_id = action.get("object_id")
            confidence = action.get("confidence", 0.5)
            description = action.get("description", "")
            
            # Create action node
            self.graph.add_node(
                action_id,
                type="action",
                action_type=action_type,
                description=description,
                timestamp=timestamp,
                confidence=confidence
            )
            
            # Connect action to the scene
            self.graph.add_edge(scene_id, action_id, relation="contains")
            
            # Connect subject to action
            if subject_id:
                # Find the corresponding character or object
                for track_id, char_id in self.character_identities.items():
                    if subject_type == "person" and subject_id in str(track_id):
                        self.graph.add_edge(char_id, action_id, relation="performs")
                        break
                
                for track_id, obj_id in self.object_identities.items():
                    if subject_type == "object" and subject_id in str(track_id):
                        self.graph.add_edge(obj_id, action_id, relation="performs")
                        break
            
            # Connect action to object
            if object_id:
                # Find the corresponding character or object
                for track_id, char_id in self.character_identities.items():
                    if object_id in str(track_id):
                        self.graph.add_edge(action_id, char_id, relation="affects")
                        break
                
                for track_id, obj_id in self.object_identities.items():
                    if object_id in str(track_id):
                        self.graph.add_edge(action_id, obj_id, relation="affects")
                        break
    
    def _process_faces(self, faces: List[Dict], objects: List[Dict], timestamp: float, scene_id: str):
        """
        Process detected faces and link to characters.
        
        Args:
            faces: List of detected faces
            objects: List of detected objects (to match faces to persons)
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        # Process is handled in _process_person with face data
        pass
    
    def _process_spatial_relationships(self, frame_data: Dict, timestamp: float, scene_id: str):
        """
        Process spatial relationships between entities.
        
        Args:
            frame_data: Frame data with objects
            timestamp: Frame timestamp
            scene_id: Current scene ID
        """
        objects = frame_data.get("objects", [])
        
        # Process object-object spatial relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue  # Skip self
                
                # Get entity IDs
                id1 = None
                id2 = None
                
                track_id1 = obj1.get("track_id")
                track_id2 = obj2.get("track_id")
                
                if track_id1 in self.character_identities:
                    id1 = self.character_identities[track_id1]
                elif track_id1 in self.object_identities:
                    id1 = self.object_identities[track_id1]
                
                if track_id2 in self.character_identities:
                    id2 = self.character_identities[track_id2]
                elif track_id2 in self.object_identities:
                    id2 = self.object_identities[track_id2]
                
                if id1 is None or id2 is None:
                    continue
                
                # Determine spatial relationship
                box1 = obj1.get("box")
                box2 = obj2.get("box")
                
                if box1 and box2:
                    spatial_rel = self._determine_spatial_relationship(box1, box2)
                    
                    if spatial_rel:
                        # Check if relationship already exists
                        if self.graph.has_edge(id1, id2) and self.graph.get_edge_data(id1, id2).get("relation") == "spatial":
                            # Update existing relationship
                            self.graph[id1][id2]["spatial"] = spatial_rel
                            self.graph[id1][id2]["last_updated"] = timestamp
                        else:
                            # Create new relationship
                            self.graph.add_edge(
                                id1,
                                id2,
                                relation="spatial",
                                spatial=spatial_rel,
                                first_seen=timestamp,
                                last_updated=timestamp
                            )
    
    def _find_current_scene(self, timestamp: float) -> Optional[str]:
        """
        Find the scene ID that contains the given timestamp.
        
        Args:
            timestamp: Timestamp to find scene for
            
        Returns:
            Scene ID or None if not found
        """
        scene_timestamps = [
            (node_id, attrs.get("timestamp", 0.0))
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "scene"
        ]
        
        # Sort by timestamp
        scene_timestamps.sort(key=lambda x: x[1])
        
        # Find the scene
        current_scene = None
        for scene_id, scene_time in scene_timestamps:
            if scene_time <= timestamp:
                current_scene = scene_id
            else:
                break
        
        return current_scene
    
    def _find_scenes_in_timespan(self, start_time: float, end_time: float) -> List[str]:
        """
        Find all scenes that overlap with the given timespan.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of scene IDs
        """
        scene_timestamps = [
            (node_id, attrs.get("timestamp", 0.0))
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "scene"
        ]
        
        # Sort by timestamp
        scene_timestamps.sort(key=lambda x: x[1])
        
        # Find scenes in timespan
        in_timespan = []
        next_scene_start = float('inf')
        
        for i, (scene_id, scene_time) in enumerate(scene_timestamps):
            # Get end time of this scene (start of next scene)
            if i < len(scene_timestamps) - 1:
                next_scene_start = scene_timestamps[i+1][1]
            
            # Check if scene overlaps with timespan
            if scene_time <= end_time and next_scene_start >= start_time:
                in_timespan.append(scene_id)
        
        return in_timespan
    
    def visualize_graph(self, output_dir: str = "output") -> str:
        """
        Visualize the knowledge graph and save as an image.
        
        Args:
            output_dir: Directory to save the image
            
        Returns:
            Path to the saved image
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narrative_graph_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Set up figure
        plt.figure(figsize=(14, 12))
        
        # Create positions for nodes
        try:
            # Try using spring layout with optimized parameters for narrative graphs
            pos = nx.spring_layout(self.graph, k=0.3, iterations=50, scale=2.0)
        except:
            # Fallback to a simpler layout
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Get node types for coloring
        node_types = nx.get_node_attributes(self.graph, 'type')
        
        # Define colors for different node types
        color_map = {
            'scene': 'skyblue',
            'time': 'lightgray',
            'caption': 'lightgreen',
            'object': 'orange',
            'character': 'red',
            'action': 'purple',
            'speech': 'gold',
            'sound': 'brown',
            'text': 'pink',
            'emotion': 'magenta',
            'goal': 'darkgreen',
            'attribute': 'cyan',
            'entity': 'teal'
        }
        
        # Group nodes by type
        nodes_by_type = {}
        for node, node_type in node_types.items():
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # Draw nodes for each type
        for node_type, nodes in nodes_by_type.items():
            nx.draw_networkx_nodes(
                self.graph, 
                pos, 
                nodelist=nodes,
                node_color=color_map.get(node_type, 'gray'),
                node_size=300,
                alpha=0.8,
                label=node_type
            )
        
        # Group edges by relation type
        edges_by_relation = {}
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'default')
            if relation not in edges_by_relation:
                edges_by_relation[relation] = []
            edges_by_relation[relation].append((u, v))
        
        # Edge colors and styles
        edge_colors = {
            'contains': 'gray',
            'performs': 'blue',
            'affects': 'red',
            'speaks': 'gold',
            'relationship': 'green',
            'spatial': 'orange',
            'causes': 'purple',
            'has_goal': 'darkgreen',
            'default': 'black'
        }
        
        edge_styles = {
            'contains': ':',
            'performs': '-',
            'affects': '-',
            'speaks': '-',
            'relationship': '--',
            'spatial': ':',
            'causes': '->',
            'has_goal': '-.',
            'default': '-'
        }
        
        # Draw edges by relation type
        for relation, edge_list in edges_by_relation.items():
            nx.draw_networkx_edges(
                self.graph, 
                pos, 
                edgelist=edge_list,
                width=1.0, 
                alpha=0.6,
                edge_color=edge_colors.get(relation, 'gray'),
                style=edge_styles.get(relation, '-')
            )
        
        # Create a label dictionary for selected nodes
        labels = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type')
            
            if node_type == 'character':
                # For characters, use name or description
                if 'name' in attrs and attrs['name']:
                    labels[node] = attrs['name']
                else:
                    desc = attrs.get('description', '')
                    labels[node] = desc[:15] + '...' if len(desc) > 15 else desc
            
            elif node_type == 'action':
                # For actions, use description
                desc = attrs.get('description', '')
                labels[node] = desc[:15] + '...' if len(desc) > 15 else desc
            
            elif node_type == 'object':
                # For objects, use description
                desc = attrs.get('description', '')
                labels[node] = desc[:15] + '...' if len(desc) > 15 else desc
            
            elif node_type == 'speech':
                # For speech, use part of text
                text = attrs.get('text', '')
                labels[node] = text[:15] + '...' if len(text) > 15 else text
            
            else:
                # Use node ID for other types
                labels[node] = node
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        # Create legend for nodes
        node_legend_items = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=node_type)
                           for node_type, color in color_map.items() 
                           if node_type in node_types.values()]
        
        # Create legend for edges
        edge_legend_items = [plt.Line2D([0], [0], color=color, linestyle=edge_styles.get(relation, '-'),
                                       label=relation)
                            for relation, color in edge_colors.items()
                            if relation in edges_by_relation]
        
        # Add legends
        plt.legend(handles=node_legend_items, loc='upper right', title="Node Types")
        plt.legend(handles=edge_legend_items, loc='lower right', title="Edge Types")
        
        # Remove axis
        plt.axis('off')
        
        # Set title
        plt.title("Narrative Knowledge Graph")
        
        # Save figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Graph visualization saved to {filepath}")
        
        return filepath
    
    def generate_narrative_summary(self) -> str:
        """
        Generate a narrative summary of the graph.
        
        Returns:
            String with narrative summary
        """
        summary = []
        
        # Get scenes in order
        scenes = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "scene":
                scenes.append((node_id, attrs))
        
        # Sort scenes by timestamp
        scenes.sort(key=lambda x: x[1].get("timestamp", 0.0))
        
        if not scenes:
            return "No scenes found in the narrative."
        
        summary.append("# Narrative Summary\n")
        
        # Characters overview
        characters = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "character" and not attrs.get("temporary", False):
                characters.append((node_id, attrs))
        
        if characters:
            summary.append("## Characters\n")
            
            for char_id, char_attrs in characters:
                name = char_attrs.get("name", "")
                desc = char_attrs.get("description", "")
                
                char_name = name if name else f"Character {char_id}"
                char_desc = f": {desc}" if desc else ""
                
                summary.append(f"- **{char_name}**{char_desc}")
                
                # Check for goals
                has_goals = False
                for goal_id in self.graph.successors(char_id):
                    if self.graph.nodes[goal_id].get("type") != "goal":
                        continue
                        
                    if self.graph[char_id][goal_id].get("relation") == "has_goal":
                        if not has_goals:
                            summary.append(f"  - **Goals**:")
                            has_goals = True
                        
                        goal_desc = self.graph.nodes[goal_id].get("description", "")
                        summary.append(f"    - {goal_desc}")
            
            summary.append("")
        
        # Relationships overview
        relationships = []
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get("relation") == "relationship":
                
                u_node = self.graph.nodes[u]
                v_node = self.graph.nodes[v]
                
                if u_node.get("type") == "character" and v_node.get("type") == "character":
                    u_name = u_node.get("name", f"Character {u}")
                    v_name = v_node.get("name", f"Character {v}")
                    rel_type = attrs.get("type", "unknown")
                    sentiment = attrs.get("sentiment", 0.0)
                    
                    relationships.append((u_name, v_name, rel_type, sentiment))
        
        if relationships:
            summary.append("## Relationships\n")
            
            for u_name, v_name, rel_type, sentiment in relationships:
                rel_desc = ""
                if rel_type == "friendly":
                    rel_desc = "friendly"
                elif rel_type == "antagonistic":
                    rel_desc = "antagonistic"
                elif rel_type == "neutral":
                    rel_desc = "neutral"
                else:
                    # Determine from sentiment
                    if sentiment > 0.3:
                        rel_desc = "positive"
                    elif sentiment < -0.3:
                        rel_desc = "negative"
                    else:
                        rel_desc = "neutral"
                
                summary.append(f"- **{u_name}** and **{v_name}** have a {rel_desc} relationship.")
            
            summary.append("")
        
        # Scene-by-scene summary
        summary.append("## Scene-by-Scene Summary\n")
        
        for i, (scene_id, scene_attrs) in enumerate(scenes):
            timestamp = scene_attrs.get("timestamp", 0.0)
            scene_time_str = self._format_timestamp(timestamp)
            
            summary.append(f"### Scene {i+1} ({scene_time_str})\n")
            
            # Get captions for this scene
            captions = []
            for succ in self.graph.successors(scene_id):
                if self.graph.nodes[succ].get("type") == "caption":
                    caption_text = self.graph.nodes[succ].get("text", "")
                    if caption_text:
                        captions.append(caption_text)
            
            if captions:
                summary.append("**Visual Description:**")
                for caption in captions:
                    summary.append(f"- {caption}")
                summary.append("")
            
            # Get characters in this scene
            scene_characters = []
            for succ in self.graph.successors(scene_id):
                if self.graph.nodes[succ].get("type") == "character":
                    char_name = self.graph.nodes[succ].get("name", f"Character {succ}")
                    char_desc = self.graph.nodes[succ].get("description", "")
                    scene_characters.append((succ, char_name, char_desc))
            
            if scene_characters:
                summary.append("**Characters Present:**")
                for _, char_name, char_desc in scene_characters:
                    char_entry = char_name
                    if char_desc:
                        char_entry += f" ({char_desc})"
                    summary.append(f"- {char_entry}")
                summary.append("")
            
            # Get key events in this scene
            events = []
            
            # Actions
            for succ in self.graph.successors(scene_id):
                if self.graph.nodes[succ].get("type") != "action":
                    continue
                
                action = self.graph.nodes[succ]
                action_desc = action.get("description", "")
                action_time = action.get("timestamp", 0.0)
                
                # Get actor
                actor_name = "Someone"
                for pred in self.graph.predecessors(succ):
                    if self.graph[pred][succ].get("relation") == "performs":
                        if self.graph.nodes[pred].get("type") == "character":
                            actor_name = self.graph.nodes[pred].get("name", f"Character {pred}")
                        break
                
                # Get target
                target_desc = ""
                for target in self.graph.successors(succ):
                    if self.graph[succ][target].get("relation") == "affects":
                        target_type = self.graph.nodes[target].get("type")
                        if target_type == "character":
                            target_name = self.graph.nodes[target].get("name", f"Character {target}")
                            target_desc = f" towards {target_name}"
                        elif target_type == "object":
                            obj_desc = self.graph.nodes[target].get("description", "an object")
                            target_desc = f" involving {obj_desc}"
                        break
                
                events.append((action_time, f"{actor_name} {action_desc}{target_desc}"))
            
            # Speeches
            for succ in self.graph.successors(scene_id):
                if self.graph.nodes[succ].get("type") != "speech":
                    continue
                
                speech = self.graph.nodes[succ]
                speech_text = speech.get("text", "")
                speech_time = speech.get("start_time", 0.0)
                
                # Get speaker
                speaker_name = "Someone"
                for pred in self.graph.predecessors(succ):
                    if self.graph[pred][succ].get("relation") == "speaks":
                        if self.graph.nodes[pred].get("type") == "character":
                            speaker_name = self.graph.nodes[pred].get("name", f"Character {pred}")
                        break
                
                events.append((speech_time, f"{speaker_name} says: \"{speech_text}\""))
            
            # Sounds
            for succ in self.graph.successors(scene_id):
                if self.graph.nodes[succ].get("type") != "sound":
                    continue
                
                sound = self.graph.nodes[succ]
                sound_class = sound.get("sound_class", "")
                sound_time = sound.get("start_time", 0.0)
                
                if sound_class:
                    events.append((sound_time, f"Sound: {sound_class}"))
            
            # Emotions
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("type") != "emotion":
                    continue
                
                emotion_time = attrs.get("timestamp", 0.0)
                
                # Check if this emotion is in the current scene
                next_scene_time = float('inf')
                if i < len(scenes) - 1:
                    next_scene_time = scenes[i+1][1].get("timestamp", 0.0)
                
                if emotion_time < timestamp or emotion_time >= next_scene_time:
                    continue
                
                emotion = attrs.get("emotion", "")
                intensity = attrs.get("intensity", 0.5)
                intensity_desc = "strongly " if intensity > 0.7 else ""
                
                # Get character
                char_name = "Someone"
                for pred in self.graph.predecessors(node_id):
                    if self.graph.nodes[pred].get("type") == "character":
                        char_name = self.graph.nodes[pred].get("name", f"Character {pred}")
                        break
                
                events.append((emotion_time, f"{char_name} feels {intensity_desc}{emotion}"))
            
            # Sort events by time
            events.sort(key=lambda x: x[0])
            
            if events:
                summary.append("**Key Events:**")
                for _, event_desc in events:
                    summary.append(f"- {event_desc}")
                summary.append("")
        
        return "\n".join(summary)
    
    def _format_timestamp(self, timestamp: float) -> str:
        """
        Format a timestamp in a human-readable way.
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def query_graph(self, query_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the knowledge graph for insights.
        
        Args:
            query_type: Type of query to perform
            params: Additional parameters for the query
            
        Returns:
            Dictionary with query results
        """
        params = params or {}
        result = {"query_type": query_type, "success": True, "data": None}
        
        try:
            if query_type == "character_info":
                result["data"] = self._query_character_info(params.get("character_id"))
            
            elif query_type == "scene_summary":
                result["data"] = self._query_scene_summary(params.get("scene_id"))
            
            elif query_type == "character_relationships":
                result["data"] = self._query_character_relationships(params.get("character_id"))
            
            elif query_type == "character_goals":
                result["data"] = self._query_character_goals(params.get("character_id"))
            
            elif query_type == "event_sequence":
                result["data"] = self._query_event_sequence(
                    params.get("start_time", 0.0),
                    params.get("end_time", float('inf'))
                )
            
            elif query_type == "causal_chain":
                result["data"] = self._query_causal_chain(params.get("event_id"))
            
            elif query_type == "spatial_relationships":
                result["data"] = self._query_spatial_relationships(
                    params.get("entity_id"),
                    params.get("timestamp")
                )
            
            else:
                result["success"] = False
                result["error"] = f"Unknown query type: {query_type}"
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def save_graph(self, output_dir: str = "output") -> str:
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narrative_graph_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert graph to dictionary
        graph_dict = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            # Convert non-serializable objects
            serializable_data = {}
            for key, value in node_data.items():
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    serializable_data[key] = value
                else:
                    serializable_data[key] = str(value)
                    
            node_dict = {"id": node_id}
            node_dict.update(serializable_data)
            graph_dict["nodes"].append(node_dict)
        
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            # Convert non-serializable objects
            serializable_edge_data = {}
            for key, value in edge_data.items():
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    serializable_edge_data[key] = value
                else:
                    serializable_edge_data[key] = str(value)
                    
            edge_dict = {
                "source": source,
                "target": target
            }
            edge_dict.update(serializable_edge_data)
            graph_dict["edges"].append(edge_dict)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, indent=2)
        
        self.logger.info(f"Graph saved to {filepath}")
        
        return filepath
    
    def _assess_causal_probability(self, cause_id: str, cause_attrs: Dict,
                                 effect_id: str, effect_attrs: Dict) -> float:
        """
        Assess the probability that one event caused another.
        
        Args:
            cause_id: ID of potential cause
            cause_attrs: Attributes of potential cause
            effect_id: ID of potential effect
            effect_attrs: Attributes of potential effect
            
        Returns:
            Probability value between 0 and 1
        """
        # Default probability
        probability = 0.5
        
        cause_type = cause_attrs.get("type")
        effect_type = effect_attrs.get("type")
        
        # Special case: actions often cause emotions
        if cause_type == "action" and effect_type == "emotion":
            probability = 0.7
            
            # Check if action is notable
            if "throw" in cause_attrs.get("description", "").lower() or \
               "hit" in cause_attrs.get("description", "").lower() or \
               "attack" in cause_attrs.get("description", "").lower():
                probability = 0.8
            
            # Check if emotion is strong
            if effect_attrs.get("intensity", 0.5) > 0.7:
                probability += 0.1
            
            # Check if emotion matches action
            action_desc = cause_attrs.get("description", "").lower()
            emotion = effect_attrs.get("emotion", "").lower()
            
            if ("threat" in action_desc or "scary" in action_desc) and emotion == "afraid":
                probability = 0.9
            elif "hit" in action_desc and emotion == "angry":
                probability = 0.9
            elif "gift" in action_desc and emotion == "happy":
                probability = 0.9
        
        # Special case: speech can cause emotions
        elif cause_type == "speech" and effect_type == "emotion":
            probability = 0.6
            
            # Check speech sentiment vs emotion
            speech_sentiment = cause_attrs.get("sentiment", 0.0)
            emotion = effect_attrs.get("emotion", "").lower()
            
            if speech_sentiment > 0.5 and emotion == "happy":
                probability = 0.8
            elif speech_sentiment < -0.5 and emotion in ["angry", "sad"]:
                probability = 0.8
        
        # Cap probability
        return min(max(probability, 0.0), 1.0)
    
    def _infer_character_goals(self):
        """
        Infer character goals based on actions.
        """
        self.logger.debug("Inferring character goals")
        
        # Find all character nodes
        character_nodes = [
            node_id for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "character" and not attrs.get("temporary", False)
        ]
        
        # For each character, analyze actions to infer goals
        for char_id in character_nodes:
            # Get all actions performed by this character
            actions = []
            
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("type") != "action":
                    continue
                
                # Check if this character performed the action
                is_actor = False
                for pred in self.graph.predecessors(node_id):
                    if pred == char_id:
                        is_actor = True
                        break
                
                if is_actor:
                    actions.append((node_id, attrs))
            
            # If there are actions, analyze patterns
            if actions:
                # Count action targets
                target_counts = defaultdict(int)
                
                for action_id, action_attrs in actions:
                    # Find action target
                    for succ in self.graph.successors(action_id):
                        if self.graph.nodes[succ].get("type") in ["object", "character"]:
                            target_counts[succ] += 1
                
                # Find the most common targets
                if target_counts:
                    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
                    top_target, count = sorted_targets[0]
                    
                    # If a target appears multiple times, it's likely a goal
                    if count >= 2:
                        # Create a goal node
                        goal_id = f"goal_{char_id}_{top_target}"
                        
                        # Determine goal type based on target
                        target_type = self.graph.nodes[top_target].get("type")
                        
                        if target_type == "object":
                            description = f"obtain the {self.graph.nodes[top_target].get('description', 'object')}"
                        else:  # character
                            # Infer relationship goal
                            relationship_type = "interact with"
                            
                            # Check for existing relationship
                            if self.graph.has_edge(char_id, top_target) and \
                               self.graph.get_edge_data(char_id, top_target).get("relation") == "relationship":
                                rel_data = self.graph.get_edge_data(char_id, top_target)
                                rel_type = rel_data.get("type")
                                sentiment = rel_data.get("sentiment", 0.0)
                                
                                if rel_type == "friendly" or sentiment > 0.3:
                                    relationship_type = "befriend"
                                elif rel_type == "antagonistic" or sentiment < -0.3:
                                    relationship_type = "confront"
                            
                            description = f"{relationship_type} {self.graph.nodes[top_target].get('name', 'the person')}"
                        
                        # Add goal node
                        self.graph.add_node(
                            goal_id,
                            type="goal",
                            description=description,
                            target=top_target,
                            confidence=min(1.0, count / 3.0),  # Normalize confidence
                            evidence=count
                        )
                        
                        # Link character to goal
                        self.graph.add_edge(char_id, goal_id, relation="has_goal")
    
    def _infer_emotional_responses(self):
        """
        Infer emotional responses to events.
        """
        self.logger.debug("Inferring emotional responses")
        
        # Get all emotion nodes
        emotion_nodes = [
            (node_id, attrs) for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "emotion"
        ]
        
        # For each emotion, look for a cause if none exists
        for emotion_id, emotion_attrs in emotion_nodes:
            # Check if emotion already has a cause
            has_cause = False
            
            for pred in self.graph.predecessors(emotion_id):
                edge_data = self.graph.get_edge_data(pred, emotion_id)
                if edge_data.get("relation") == "causes":
                    has_cause = True
                    break
            
            if has_cause:
                continue  # Already has a cause
            
            # Find character experiencing the emotion
            character_id = None
            for pred in self.graph.predecessors(emotion_id):
                if self.graph.nodes[pred].get("type") == "character":
                    character_id = pred
                    break
            
            if not character_id:
                continue  # No character found
            
            # Find possible causes
            emotion_time = emotion_attrs.get("timestamp", 0.0)
            
            # Look for events shortly before this emotion
            possible_causes = []
            
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("type") not in ["action", "speech"]:
                    continue
                
                # Get event time
                if attrs.get("type") == "speech":
                    event_time = attrs.get("end_time", 0.0)
                else:
                    event_time = attrs.get("timestamp", 0.0)
                
                # Check if event happened before emotion but within causal window
                time_diff = emotion_time - event_time
                if 0 < time_diff < self.causal_window:
                    # Get characters involved in the event
                    event_characters = set()
                    
                    if attrs.get("type") == "action":
                        # Get actor
                        for pred in self.graph.predecessors(node_id):
                            if self.graph.nodes[pred].get("type") == "character":
                                event_characters.add(pred)
                        
                        # Get target
                        for succ in self.graph.successors(node_id):
                            if self.graph.nodes[succ].get("type") == "character":
                                event_characters.add(succ)
                    
                    elif attrs.get("type") == "speech":
                        # Get speaker
                        for pred in self.graph.predecessors(node_id):
                            if self.graph.nodes[pred].get("type") == "character":
                                event_characters.add(pred)
                    
                    # Check if character is involved
                    if character_id in event_characters or not event_characters:
                        possible_causes.append((node_id, attrs, time_diff))
            
            # Find the most likely cause
            if possible_causes:
                # Sort by time (closest first)
                possible_causes.sort(key=lambda x: x[2])
                
                best_cause_id, best_cause_attrs, _ = possible_causes[0]
                
                # Calculate probability
                probability = self._assess_causal_probability(
                    best_cause_id, best_cause_attrs,
                    emotion_id, emotion_attrs
                )
                
                # Add causal edge if probability is high enough
                if probability > 0.5:
                    self.graph.add_edge(
                        best_cause_id,
                        emotion_id,
                        relation="causes",
                        probability=probability
                    )
    
    def _analyze_character_interactions(self, char1_id: str, char2_id: str) -> Dict[str, Any]:
        """
        Analyze interactions between two characters.
        
        Args:
            char1_id: ID of first character
            char2_id: ID of second character
            
        Returns:
            Dictionary with interaction analysis
        """
        # Initialize scores
        interaction = {
            "score": 0.0,
            "type": "unknown",
            "sentiment": 0.0,
            "evidence": []
        }
        
        # Count different types of interactions
        direct_speech = 0  # Character speaking directly to the other
        about_speech = 0   # Character speaking about the other
        proximity = 0      # Characters being near each other
        actions = 0        # Actions involving both characters
        
        # Track sentiment
        sentiment_sum = 0.0
        sentiment_count = 0
        
        # Check speeches
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "speech":
                continue
            
            # Check if one character is speaking
            is_speaker1 = False
            for pred in self.graph.predecessors(node_id):
                if pred == char1_id:
                    is_speaker1 = True
                    break
            
            is_speaker2 = False
            for pred in self.graph.predecessors(node_id):
                if pred == char2_id:
                    is_speaker2 = True
                    break
            
            text = attrs.get("text", "").lower()
            sentiment = attrs.get("sentiment", 0.0)
            
            # Check for direct speech
            if is_speaker1:
                # Character 1 is speaking
                char2_name = self.graph.nodes[char2_id].get("name", "").lower()
                if char2_name and any(name_part in text for name_part in char2_name.split()):
                    # Speaking directly to/about character 2
                    direct_speech += 1
                    sentiment_sum += sentiment
                    sentiment_count += 1
                    
                    interaction["evidence"].append({
                        "type": "speech",
                        "node_id": node_id,
                        "text": text
                    })
            
            if is_speaker2:
                # Character 2 is speaking
                char1_name = self.graph.nodes[char1_id].get("name", "").lower()
                if char1_name and any(name_part in text for name_part in char1_name.split()):
                    # Speaking directly to/about character 1
                    direct_speech += 1
                    sentiment_sum += sentiment
                    sentiment_count += 1
                    
                    interaction["evidence"].append({
                        "type": "speech",
                        "node_id": node_id,
                        "text": text
                    })
        
        # Check spatial proximity
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get("relation") != "spatial":
                continue
            
            if (u == char1_id and v == char2_id) or (u == char2_id and v == char1_id):
                proximity += 1
                
                interaction["evidence"].append({
                    "type": "proximity",
                    "spatial": attrs.get("spatial"),
                    "first_seen": attrs.get("first_seen"),
                    "last_updated": attrs.get("last_updated")
                })
        
        # Check actions involving both characters
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "action":
                continue
            
            # Check if action involves both characters
            involves_char1 = False
            involves_char2 = False
            
            for pred in self.graph.predecessors(node_id):
                if pred == char1_id:
                    involves_char1 = True
                elif pred == char2_id:
                    involves_char2 = True
            
            for succ in self.graph.successors(node_id):
                if succ == char1_id:
                    involves_char1 = True
                elif succ == char2_id:
                    involves_char2 = True
            
            if involves_char1 and involves_char2:
                actions += 1
                
                interaction["evidence"].append({
                    "type": "action",
                    "node_id": node_id,
                    "description": attrs.get("description"),
                    "timestamp": attrs.get("timestamp")
                })
        
        # Calculate overall interaction score
        total_interactions = direct_speech + proximity + actions
        
        if total_interactions > 0:
            interaction["score"] = min(1.0, total_interactions / 3.0)
            
            # Calculate average sentiment
            if sentiment_count > 0:
                interaction["sentiment"] = sentiment_sum / sentiment_count
            
            # Determine relationship type
            if interaction["sentiment"] > 0.3:
                interaction["type"] = "friendly"
            elif interaction["sentiment"] < -0.3:
                interaction["type"] = "antagonistic"
            else:
                interaction["type"] = "neutral"
        
        return interaction
    
    def _infer_causal_relationships(self):
        """
        Infer causal relationships between events.
        """
        self.logger.debug("Inferring causal relationships")
        
        # Find all event-like nodes (actions, speeches, emotions)
        event_nodes = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") in ["action", "speech", "emotion"]:
                # Get timestamp
                if attrs.get("type") == "speech":
                    timestamp = attrs.get("start_time", 0.0)
                else:
                    timestamp = attrs.get("timestamp", 0.0)
                
                event_nodes.append((node_id, attrs, timestamp))
        
        # Sort by timestamp
        event_nodes.sort(key=lambda x: x[2])
        
        # For each event, look for potential effects in subsequent events
        for i, (cause_id, cause_attrs, cause_time) in enumerate(event_nodes):
            cause_type = cause_attrs.get("type")
            
            # Get characters involved in the cause
            cause_characters = set()
            
            if cause_type == "action":
                # Get actor
                for pred in self.graph.predecessors(cause_id):
                    if self.graph.nodes[pred].get("type") == "character":
                        cause_characters.add(pred)
                
                # Get target
                for succ in self.graph.successors(cause_id):
                    if self.graph.nodes[succ].get("type") == "character":
                        cause_characters.add(succ)
            
            elif cause_type == "speech":
                # Get speaker
                for pred in self.graph.predecessors(cause_id):
                    if self.graph.nodes[pred].get("type") == "character":
                        cause_characters.add(pred)
            
            elif cause_type == "emotion":
                # Get character feeling the emotion
                for pred in self.graph.predecessors(cause_id):
                    if self.graph.nodes[pred].get("type") == "character":
                        cause_characters.add(pred)
            
            # Look for potential effects within the causal window
            for j in range(i + 1, len(event_nodes)):
                effect_id, effect_attrs, effect_time = event_nodes[j]
                
                # Check if within causal window
                if effect_time - cause_time > self.causal_window:
                    break  # Too far in the future
                
                effect_type = effect_attrs.get("type")
                
                # Get characters involved in the effect
                effect_characters = set()
                
                if effect_type == "action":
                    # Get actor
                    for pred in self.graph.predecessors(effect_id):
                        if self.graph.nodes[pred].get("type") == "character":
                            effect_characters.add(pred)
                    
                    # Get target
                    for succ in self.graph.successors(effect_id):
                        if self.graph.nodes[succ].get("type") == "character":
                            effect_characters.add(succ)
                
                elif effect_type == "speech":
                    # Get speaker
                    for pred in self.graph.predecessors(effect_id):
                        if self.graph.nodes[pred].get("type") == "character":
                            effect_characters.add(pred)
                
                elif effect_type == "emotion":
                    # Get character feeling the emotion
                    for pred in self.graph.predecessors(effect_id):
                        if self.graph.nodes[pred].get("type") == "character":
                            effect_characters.add(pred)
                
                # Check if there's character overlap
                if cause_characters and effect_characters and cause_characters.intersection(effect_characters):
                    # Characters overlap, so potential causal relationship
                    
                    # Check relationship plausibility
                    probability = self._assess_causal_probability(cause_id, cause_attrs, effect_id, effect_attrs)
                    
                    if probability > 0.5:
                        # Add causal edge
                        self.graph.add_edge(
                            cause_id,
                            effect_id,
                            relation="causes",
                            probability=probability
                        )
    
    def _determine_spatial_relationship(self, box1: List[int], box2: List[int]) -> Optional[str]:
        """
        Determine the spatial relationship between two bounding boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            Spatial relationship string or None
        """
        # Calculate centers
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        dx = center2_x - center1_x
        dy = center2_y - center1_y
        
        # Check if boxes overlap or are very close
        if self._boxes_overlap_or_close(box1, box2):
            return "near"
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal relationship is primary
            return "right_of" if dx > 0 else "left_of"
        else:
            # Vertical relationship is primary
            return "below" if dy > 0 else "above"
    
    def _boxes_overlap_or_close(self, box1: List[int], box2: List[int], threshold: int = None) -> bool:
        """
        Check if two boxes overlap or are close to each other.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            threshold: Distance threshold (overrides config)
            
        Returns:
            True if boxes overlap or are close
        """
        if threshold is None:
            threshold = self.spatial_distance_threshold
        
        # Check for overlap
        if box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]:
            return True
        
        # Calculate centers
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        # Calculate distance
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        return distance < threshold
    
    def _process_audio_data(self, audio_data: List[Dict]):
        """
        Process audio data to extract speech and non-speech sounds.
        
        Args:
            audio_data: List of audio processing results
        """
        self.logger.debug(f"Processing {len(audio_data)} audio segments")
        
        for i, segment in enumerate(audio_data):
            segment_type = segment.get("type")
            start_time = segment.get("start")
            end_time = segment.get("end")
            
            # Find scenes that overlap with this segment
            scenes = self._find_scenes_in_timespan(start_time, end_time)
            
            if not scenes:
                continue
            
            if segment_type == "speech":
                self._process_speech_segment(segment, i, scenes)
            elif segment_type == "non-speech":
                self._process_non_speech_segment(segment, i, scenes)
    
    def _process_speech_segment(self, segment: Dict, index: int, scenes: List[str]):
        """
        Process a speech audio segment.
        
        Args:
            segment: Speech segment data
            index: Segment index
            scenes: List of scene IDs that overlap with this segment
        """
        speech_id = f"speech_{index}"
        text = segment.get("text", "")
        speaker = segment.get("speaker", "unknown")
        start_time = segment.get("start")
        end_time = segment.get("end")
        confidence = segment.get("confidence", 0.5)
        sentiment = segment.get("sentiment", 0.0)
        
        # Create speech node
        self.graph.add_node(
            speech_id,
            type="speech",
            text=text,
            speaker_id=speaker,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            sentiment=sentiment
        )
        
        # Link to scenes
        for scene_id in scenes:
            self.graph.add_edge(scene_id, speech_id, relation="contains")
        
        # Try to link to character
        speaker_char_id = None
        
        # Check if any character's name matches the speaker ID
        for char_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "character":
                continue
            
            # Check if character name contains speaker ID
            name = attrs.get("name", "")
            if speaker.lower() in name.lower() or name.lower() in speaker.lower():
                speaker_char_id = char_id
                break
        
        # If no match by name, use heuristics:
        # - Link to character that appears in all scenes this speech appears in
        # - With time overlap between character's presence and speech
        if speaker_char_id is None:
            for char_id, attrs in self.graph.nodes(data=True):
                if attrs.get("type") != "character":
                    continue
                
                # Check time overlap
                char_first_seen = attrs.get("first_seen")
                char_last_seen = attrs.get("last_seen")
                
                if char_first_seen <= end_time and char_last_seen >= start_time:
                    # Character was present during the speech
                    speaker_char_id = char_id
                    break
        
        # Link speech to character
        if speaker_char_id:
            self.graph.add_edge(speaker_char_id, speech_id, relation="speaks")
    
    def _process_non_speech_segment(self, segment: Dict, index: int, scenes: List[str]):
        """
        Process a non-speech audio segment.
        
        Args:
            segment: Non-speech segment data
            index: Segment index
            scenes: List of scene IDs that overlap with this segment
        """
        sound_id = f"sound_{index}"
        sound_class = segment.get("class", "unknown")
        start_time = segment.get("start")
        end_time = segment.get("end")
        confidence = segment.get("confidence", 0.5)
        
        # Create sound node
        self.graph.add_node(
            sound_id,
            type="sound",
            sound_class=sound_class,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence
        )
        
        # Link to scenes
        for scene_id in scenes:
            self.graph.add_edge(scene_id, sound_id, relation="contains")
    
    def _process_ocr_data(self, ocr_data: List[Dict]):
        """
        Process OCR data to extract text in frames.
        
        Args:
            ocr_data: List of OCR processing results by frame
        """
        self.logger.debug(f"Processing {len(ocr_data)} OCR frames")
        
        for frame_data in ocr_data:
            timestamp = frame_data.get("timestamp", 0.0)
            
            # Find the current scene
            scene_id = self._find_current_scene(timestamp)
            if scene_id is None:
                continue
            
            # Process text detections
            for text_det in frame_data.get("text_detections", []):
                text = text_det.get("text")
                confidence = text_det.get("confidence", 0.5)
                box = text_det.get("box")
                track_id = text_det.get("track_id")
                
                if not text or not box:
                    continue
                
                # Check if this text is already tracked
                if track_id in self.text_identities:
                    text_id = self.text_identities[track_id]
                    
                    # Update existing text node
                    if text_id in self.graph.nodes:
                        # Update confidence to maximum observed
                        current_conf = self.graph.nodes[text_id].get("confidence", 0.0)
                        if confidence > current_conf:
                            self.graph.nodes[text_id]["confidence"] = confidence
                        
                        # Update last seen timestamp
                        self.graph.nodes[text_id]["last_seen"] = timestamp
                        
                        # Add position
                        position = {
                            "timestamp": timestamp,
                            "box": box,
                            "scene_id": scene_id
                        }
                        if "positions" in self.graph.nodes[text_id]:
                            self.graph.nodes[text_id]["positions"].append(position)
                else:
                    # Create new text node
                    text_id = f"text_{len(self.text_identities) + 1}"
                    self.text_identities[track_id if track_id else f"temp_{timestamp}_{text[:10]}"] = text_id
                    
                    self.graph.add_node(
                        text_id,
                        type="text",
                        content=text,
                        confidence=confidence,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        positions=[
                            {
                                "timestamp": timestamp,
                                "box": box,
                                "scene_id": scene_id
                            }
                        ]
                    )
                    
                    # Link to scene
                    self.graph.add_edge(scene_id, text_id, relation="contains")
    
    def _resolve_entities(self):
        """
        Resolve entity identities to merge duplicates.
        """
        self.logger.debug("Resolving entity identities")
        
        # Resolve character identities
        self._resolve_character_identities()
        
        # Resolve object identities
        self._resolve_object_identities()
        
        # Resolve text identities
        self._resolve_text_identities()
    
    def _resolve_character_identities(self):
        """
        Resolve character identities using facial features and descriptions.
        """
        # Find all character nodes
        character_nodes = [
            (node_id, attrs) for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "character" and not attrs.get("temporary", False)
        ]
        
        # Check each pair of characters for potential merges
        merged_characters = set()
        
        for i, (char1_id, char1_attrs) in enumerate(character_nodes):
            if char1_id in merged_characters:
                continue  # Already merged
            
            for j, (char2_id, char2_attrs) in enumerate(character_nodes[i+1:], i+1):
                if char2_id in merged_characters:
                    continue  # Already merged
                
                # Check if these characters should be merged
                should_merge = False
                
                # Check facial feature similarity
                if char1_id in self.character_embeddings and char2_id in self.character_embeddings:
                    feat1 = self.character_embeddings[char1_id]
                    feat2 = self.character_embeddings[char2_id]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                    
                    if similarity > self.character_face_similarity_threshold:
                        should_merge = True
                
                # Check description similarity
                if not should_merge:
                    desc1 = char1_attrs.get("description", "").lower()
                    desc2 = char2_attrs.get("description", "").lower()
                    
                    if desc1 and desc2:
                        # Simple token overlap
                        tokens1 = set(desc1.split())
                        tokens2 = set(desc2.split())
                        
                        overlap = len(tokens1.intersection(tokens2))
                        union = len(tokens1.union(tokens2))
                        
                        if union > 0 and overlap / union > self.character_description_similarity_threshold:
                            should_merge = True
                
                # Merge if needed
                if should_merge:
                    self._merge_characters(char1_id, char2_id)
                    merged_characters.add(char2_id)
    
    def _merge_characters(self, primary_id: str, secondary_id: str):
        """
        Merge secondary character into primary character.
        
        Args:
            primary_id: ID of the primary character to keep
            secondary_id: ID of the secondary character to merge in
        """
        self.logger.debug(f"Merging characters: {secondary_id} into {primary_id}")
        
        if primary_id not in self.graph.nodes or secondary_id not in self.graph.nodes:
            return
        
        # Get attributes of both characters
        primary_attrs = self.graph.nodes[primary_id]
        secondary_attrs = self.graph.nodes[secondary_id]
        
        # Merge positions
        positions = primary_attrs.get("positions", []) + secondary_attrs.get("positions", [])
        primary_attrs["positions"] = sorted(positions, key=lambda p: p.get("timestamp", 0))
        
        # Update first_seen and last_seen
        primary_attrs["first_seen"] = min(
            primary_attrs.get("first_seen", float('inf')),
            secondary_attrs.get("first_seen", float('inf'))
        )
        primary_attrs["last_seen"] = max(
            primary_attrs.get("last_seen", 0),
            secondary_attrs.get("last_seen", 0)
        )
        
        # Redirect all edges from secondary to primary
        for pred in list(self.graph.predecessors(secondary_id)):
            edge_data = self.graph.get_edge_data(pred, secondary_id)
            if not self.graph.has_edge(pred, primary_id):
                self.graph.add_edge(pred, primary_id, **edge_data)
        
        for succ in list(self.graph.successors(secondary_id)):
            edge_data = self.graph.get_edge_data(secondary_id, succ)
            if not self.graph.has_edge(primary_id, succ):
                self.graph.add_edge(primary_id, succ, **edge_data)
        
        # Remove the secondary node
        self.graph.remove_node(secondary_id)
        
        # Update character_identities map
        for track_id, char_id in list(self.character_identities.items()):
            if char_id == secondary_id:
                self.character_identities[track_id] = primary_id
        
        # Update character_embeddings
        if secondary_id in self.character_embeddings:
            self.character_embeddings[primary_id] = self.character_embeddings.pop(secondary_id)
    
    def _resolve_object_identities(self):
        """
        Resolve object identities using descriptions and positions.
        """
        # Find all object nodes
        object_nodes = [
            (node_id, attrs) for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "object" and not attrs.get("temporary", False)
        ]
        
        # Check each pair of objects for potential merges
        merged_objects = set()
        
        for i, (obj1_id, obj1_attrs) in enumerate(object_nodes):
            if obj1_id in merged_objects:
                continue  # Already merged
            
            for j, (obj2_id, obj2_attrs) in enumerate(object_nodes[i+1:], i+1):
                if obj2_id in merged_objects:
                    continue  # Already merged
                
                # Check if these objects should be merged
                should_merge = False
                
                # Check description similarity
                desc1 = obj1_attrs.get("description", "").lower()
                desc2 = obj2_attrs.get("description", "").lower()
                
                if desc1 and desc2:
                    # Simple token overlap
                    tokens1 = set(desc1.split())
                    tokens2 = set(desc2.split())
                    
                    overlap = len(tokens1.intersection(tokens2))
                    union = len(tokens1.union(tokens2))
                    
                    if union > 0 and overlap / union > 0.7:  # Lower threshold for objects
                        should_merge = True
                
                # Check position proximity
                if not should_merge:
                    obj1_positions = obj1_attrs.get("positions", [])
                    obj2_positions = obj2_attrs.get("positions", [])
                    
                    # Check if positions are close in time but not in space
                    for pos1 in obj1_positions:
                        for pos2 in obj2_positions:
                            time_diff = abs(pos1.get("timestamp", 0) - pos2.get("timestamp", 0))
                            
                            if time_diff < 0.5:  # Close in time
                                # Check if positions are far apart in space
                                if pos1.get("box") and pos2.get("box"):
                                    if not self._boxes_overlap_or_close(pos1["box"], pos2["box"], threshold=100):
                                        # If objects are distant at same timestamp, they're different objects
                                        should_merge = False
                                        break
                        if not should_merge:
                            break
                
                # Merge if needed
                if should_merge:
                    self._merge_objects(obj1_id, obj2_id)
                    merged_objects.add(obj2_id)
    
    def _merge_objects(self, primary_id: str, secondary_id: str):
        """
        Merge secondary object into primary object.
        
        Args:
            primary_id: ID of the primary object to keep
            secondary_id: ID of the secondary object to merge in
        """
        self.logger.debug(f"Merging objects: {secondary_id} into {primary_id}")
        
        if primary_id not in self.graph.nodes or secondary_id not in self.graph.nodes:
            return
        
        # Get attributes of both objects
        primary_attrs = self.graph.nodes[primary_id]
        secondary_attrs = self.graph.nodes[secondary_id]
        
        # Merge positions
        positions = primary_attrs.get("positions", []) + secondary_attrs.get("positions", [])
        primary_attrs["positions"] = sorted(positions, key=lambda p: p.get("timestamp", 0))
        
        # Update first_seen and last_seen
        primary_attrs["first_seen"] = min(
            primary_attrs.get("first_seen", float('inf')),
            secondary_attrs.get("first_seen", float('inf'))
        )
        primary_attrs["last_seen"] = max(
            primary_attrs.get("last_seen", 0),
            secondary_attrs.get("last_seen", 0)
        )
        
        # Merge attributes like color, size, material
        for attr in ["color", "size", "material"]:
            if attr in secondary_attrs and attr not in primary_attrs:
                primary_attrs[attr] = secondary_attrs[attr]
        
        # Redirect all edges from secondary to primary
        for pred in list(self.graph.predecessors(secondary_id)):
            edge_data = self.graph.get_edge_data(pred, secondary_id)
            if not self.graph.has_edge(pred, primary_id):
                self.graph.add_edge(pred, primary_id, **edge_data)
        
        for succ in list(self.graph.successors(secondary_id)):
            edge_data = self.graph.get_edge_data(secondary_id, succ)
            if not self.graph.has_edge(primary_id, succ):
                self.graph.add_edge(primary_id, succ, **edge_data)
        
        # Remove the secondary node
        self.graph.remove_node(secondary_id)
        
        # Update object_identities map
        for track_id, obj_id in list(self.object_identities.items()):
            if obj_id == secondary_id:
                self.object_identities[track_id] = primary_id
    
    def _resolve_text_identities(self):
        """
        Resolve text identities to merge duplicate text occurrences.
        """
        # Group nodes by content
        text_content_groups = defaultdict(list)
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") != "text":
                continue
            
            # Skip temporary nodes
            if attrs.get("temporary", False):
                continue
            
            content = attrs.get("content", "").strip().lower()
            if content:
                text_content_groups[content].append((node_id, attrs))
        
        # Merge text with same content
        for content, nodes in text_content_groups.items():
            if len(nodes) <= 1:
                continue
                
            # Use the first occurrence as primary
            primary_id, _ = nodes[0]
            
            for secondary_id, _ in nodes[1:]:
                self._merge_text_nodes(primary_id, secondary_id)
    
    def _merge_text_nodes(self, primary_id: str, secondary_id: str):
        """
        Merge secondary text node into primary text node.
        
        Args:
            primary_id: ID of the primary text node to keep
            secondary_id: ID of the secondary text node to merge in
        """
        if primary_id not in self.graph.nodes or secondary_id not in self.graph.nodes:
            return
            
        # Get attributes of both nodes
        primary_attrs = self.graph.nodes[primary_id]
        secondary_attrs = self.graph.nodes[secondary_id]
        
        # Merge positions
        positions = primary_attrs.get("positions", []) + secondary_attrs.get("positions", [])
        primary_attrs["positions"] = sorted(positions, key=lambda p: p.get("timestamp", 0))
        
        # Update first_seen and last_seen
        primary_attrs["first_seen"] = min(
            primary_attrs.get("first_seen", float('inf')),
            secondary_attrs.get("first_seen", float('inf'))
        )
        primary_attrs["last_seen"] = max(
            primary_attrs.get("last_seen", 0),
            secondary_attrs.get("last_seen", 0)
        )
        
        # Update confidence to max
        primary_attrs["confidence"] = max(
            primary_attrs.get("confidence", 0.0),
            secondary_attrs.get("confidence", 0.0)
        )
        
        # Redirect all edges
        for pred in list(self.graph.predecessors(secondary_id)):
            edge_data = self.graph.get_edge_data(pred, secondary_id)
            if not self.graph.has_edge(pred, primary_id):
                self.graph.add_edge(pred, primary_id, **edge_data)
        
        for succ in list(self.graph.successors(secondary_id)):
            edge_data = self.graph.get_edge_data(secondary_id, succ)
            if not self.graph.has_edge(primary_id, succ):
                self.graph.add_edge(primary_id, succ, **edge_data)
        
        # Remove the secondary node
        self.graph.remove_node(secondary_id)
        
        # Update text_identities map
        for track_id, txt_id in list(self.text_identities.items()):
            if txt_id == secondary_id:
                self.text_identities[track_id] = primary_id
    
    def _infer_relationships(self):
        """
        Infer relationships between entities based on interactions.
        """
        self.logger.debug("Inferring entity relationships")
        
        # Find all character nodes
        character_nodes = [
            node_id for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "character" and not attrs.get("temporary", False)
        ]
        
        # Analyze interactions between each pair of characters
        for i, char1_id in enumerate(character_nodes):
            for j, char2_id in enumerate(character_nodes[i+1:], i+1):
                # Analyze direct interactions
                interaction_score = self._analyze_character_interactions(char1_id, char2_id)
                
                # If there are significant interactions, add a relationship edge
                if interaction_score["score"] > 0.5:
                    rel_type = interaction_score["type"]
                    sentiment = interaction_score["sentiment"]
                    
                    self.graph.add_edge(
                        char1_id,
                        char2_id,
                        relation="relationship",
                        type=rel_type,
                        sentiment=sentiment,
                        confidence=interaction_score["score"],
                        evidence=interaction_score["evidence"]
                    )