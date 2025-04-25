# modules/knowledge_graph.py
"""
Knowledge Graph Construction Module for Narrative Scene Understanding.
This module builds a narrative knowledge graph from multi-modal processing results.
"""

import os
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import time
from collections import defaultdict
import json

class NarrativeGraphBuilder:
    """
    Builds and maintains a narrative knowledge graph from visual, audio, and OCR data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the narrative graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
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
    
    def build_graph(self, 
                   visual_data: List[Dict], 
                   audio_data: List[Dict], 
                   ocr_data: List[Dict],
                   scene_boundaries: List[float]) -> nx.DiGraph:
        """
        Build a narrative knowledge graph from processed data.
        
        Args:
            visual_data: List of visual processing results by frame
            audio_data: List of audio processing results
            ocr_data: List of OCR processing results by frame
            scene_boundaries: List of scene boundary timestamps
            
        Returns:
            NetworkX DiGraph containing the narrative knowledge graph
        """
        self.logger.info("Building narrative knowledge graph...")
        
        # Start with a fresh graph
        self.graph = nx.DiGraph()
        
        # Step 1: Add scene boundaries and time nodes
        self._add_scene_boundaries(scene_boundaries)
        
        # Step 2: Process visual data (characters, objects, actions)
        self._process_visual_data(visual_data)
        
        # Step 3: Process audio data (speech, non-speech sounds)
        self._process_audio_data(audio_data)
        
        # Step 4: Process OCR data (text in frames)
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
            obj_id = f"object_temp_{obj.get('id')}"
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
                    self.text_identities[track_id] = text_id
                    
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
                
                # Check facial feature similarity if available
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
                
                # Merge# Check position proximity
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
                        confidence=interaction_score["score"]
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