# modules/query.py
"""
Scene Query Engine for Narrative Scene Understanding.
This module enables natural language querying of scene content using LLM reasoning.
"""

import os
import networkx as nx
import logging
import re
import json
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass

@dataclass
class QueryContext:
    """Container for assembled query context information."""
    characters: Dict[str, Dict[str, Any]]
    objects: Dict[str, Dict[str, Any]]
    actions: List[Dict[str, Any]]
    dialogue: List[Dict[str, Any]]
    spatial_info: List[Dict[str, Any]]
    emotional_states: Dict[str, List[Dict[str, Any]]]
    events: List[Dict[str, Any]]

class SceneQueryEngine:
    """
    Engine for querying narrative scenes using LLM reasoning.
    """
    
    def __init__(self, model_name: str = "llama-7b-chat", temperature: float = 0.2):
        """
        Initialize the scene query engine.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature setting for LLM generation
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Query type patterns for better context building
        self.query_patterns = {
            "motivation": r"motivation|why did|reason for|intent|purpose|trying to",
            "relationship": r"relationship|feel about|connection|dynamic between|how does \w+ see",
            "emotional": r"emotion|feeling|mood|tone|affect|attitude",
            "subtext": r"subtext|implication|suggest|hint|underlying|implied|between the lines",
            "technique": r"technique|symbolism|cinematography|director|filmmaker|shot|framing",
            "prediction": r"what will happen|what's going to|predict|next|future|expect",
            "backstory": r"backstory|history|background|past|before|previously"
        }
    
    def _init_llm_client(self):
        """Initialize the LLM client."""
        self.logger.info(f"Initializing LLM client with model: {self.model_name}")
        
        # Check if model name contains specific providers
        if "llama" in self.model_name.lower():
            self._init_llama_client()
        else:
            self._init_openai_client()
    
    def _init_llama_client(self):
        """Initialize a Llama model client."""
        try:
            from llama_cpp import Llama
            
            # Try to find the model file
            model_path = os.environ.get("LLAMA_MODEL_PATH", "models/llama-7b.gguf")
            if not os.path.exists(model_path):
                self.logger.warning(f"Llama model not found at {model_path}, using OpenAI fallback")
                self._init_openai_client()
                return
            
            # Initialize the model
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window size
                n_gpu_layers=-1  # Use all available GPU layers
            )
            
            self.llm_type = "llama"
            self.logger.info(f"Llama model loaded from {model_path}")
        
        except ImportError:
            self.logger.warning("llama-cpp package not available, using OpenAI fallback")
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize an OpenAI client."""
        try:
            import openai
            
            self.llm = openai.OpenAI()
            self.llm_type = "openai"
            self.logger.info("OpenAI client initialized")
        
        except ImportError:
            self.logger.error("Neither llama-cpp nor openai package available")
            self.llm = None
            self.llm_type = None
    
    def query_scene(self, question: str, narrative_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Process a natural language query about a scene.
        
        Args:
            question: Natural language question about the scene
            narrative_graph: NetworkX DiGraph containing scene knowledge
            
        Returns:
            Dict with answer, reasoning process, and supporting evidence
        """
        self.logger.info(f"Processing query: {question}")
        
        if self.llm is None:
            return {
                "answer": "Cannot process query: LLM not available",
                "reasoning": "",
                "evidence": [],
                "query_type": "unknown"
            }
        
        # 1. Parse query to understand intent and targets
        query_type, query_targets = self._parse_query(question)
        self.logger.debug(f"Query type: {query_type}, targets: {query_targets}")
        
        # 2. Retrieve relevant subgraph based on targets
        relevant_nodes = self._retrieve_relevant_nodes(narrative_graph, query_targets)
        
        # 3. Assemble context for LLM
        context = self._build_query_context(relevant_nodes, narrative_graph, query_type)
        
        # 4. Generate reasoning using LLM
        reasoning, answer = self._generate_reasoning_and_answer(question, context, query_type)
        
        # 5. Extract supporting evidence
        evidence = self._extract_supporting_evidence(relevant_nodes, query_targets)
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "evidence": evidence,
            "query_type": query_type
        }
    
    def _parse_query(self, question: str) -> Tuple[str, Set[str]]:
        """
        Parse the question to identify query type and target entities.
        
        Args:
            question: The natural language question
            
        Returns:
            Tuple of (query_type, set of target entities)
        """
        # Determine query type based on patterns
        query_type = "general"
        for qtype, pattern in self.query_patterns.items():
            if re.search(pattern, question.lower()):
                query_type = qtype
                break
        
        # Extract potential targets (entities mentioned in the question)
        targets = set()
        
        # Look for character names, objects, and other entities in the question
        # This is a simplified approach; a more robust NER would be better
        words = re.findall(r'\b[A-Z][a-z]+\b', question)  # Simplistic named entity extraction
        targets.update(words)
        
        # Add common objects that might be mentioned
        common_objects = ["door", "window", "car", "phone", "gun", "table", "chair", "book"]
        for obj in common_objects:
            if obj in question.lower():
                targets.add(obj)
        
        return query_type, targets
    
    def _retrieve_relevant_nodes(self, graph: nx.DiGraph, targets: Set[str]) -> Dict[str, List]:
        """
        Retrieve nodes from the graph that are relevant to the query targets.
        
        Args:
            graph: The narrative knowledge graph
            targets: Set of target entities from the query
            
        Returns:
            Dictionary of node types to lists of relevant nodes
        """
        relevant = {
            "characters": [],
            "objects": [],
            "locations": [],
            "events": [],
            "speeches": [],
            "emotions": [],
            "actions": []
        }
        
        # Get nodes that match target names or have target names in their attributes
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', '')
            
            # Check if this node matches any of our targets
            is_relevant = False
            
            # Direct match on node ID
            node_name = str(node_id).lower()
            for target in targets:
                if target.lower() in node_name:
                    is_relevant = True
                    break
            
            # Match in node attributes
            if not is_relevant:
                for attr_key, attr_value in attrs.items():
                    if isinstance(attr_value, str):
                        for target in targets:
                            if target.lower() in attr_value.lower():
                                is_relevant = True
                                break
                    elif isinstance(attr_value, list):
                        for item in attr_value:
                            if isinstance(item, str) and any(t.lower() in item.lower() for t in targets):
                                is_relevant = True
                                break
            
            # If nothing matched but targets is empty, include core nodes anyway
            if not is_relevant and not targets and node_type in ['character', 'event', 'location']:
                is_relevant = True
            
            # Add to the appropriate category if relevant
            if is_relevant:
                if node_type == 'character':
                    relevant["characters"].append((node_id, attrs))
                elif node_type in ['object', 'prop']:
                    relevant["objects"].append((node_id, attrs))
                elif node_type == 'location':
                    relevant["locations"].append((node_id, attrs))
                elif node_type == 'event':
                    relevant["events"].append((node_id, attrs))
                elif node_type == 'speech':
                    relevant["speeches"].append((node_id, attrs))
                elif node_type == 'emotion':
                    relevant["emotions"].append((node_id, attrs))
                elif node_type == 'action':
                    relevant["actions"].append((node_id, attrs))
        
        # Expand to include connected nodes (1-hop neighbors)
        character_ids = [node_id for node_id, _ in relevant["characters"]]
        for char_id in character_ids:
            # Get all nodes connected to this character
            for neighbor_id in graph.neighbors(char_id):
                neighbor_attrs = graph.nodes[neighbor_id]
                neighbor_type = neighbor_attrs.get('type', '')
                
                # Add to appropriate category if not already included
                if neighbor_type == 'action':
                    if not any(node_id == neighbor_id for node_id, _ in relevant["actions"]):
                        relevant["actions"].append((neighbor_id, neighbor_attrs))
                elif neighbor_type == 'emotion':
                    if not any(node_id == neighbor_id for node_id, _ in relevant["emotions"]):
                        relevant["emotions"].append((neighbor_id, neighbor_attrs))
                elif neighbor_type == 'speech':
                    if not any(node_id == neighbor_id for node_id, _ in relevant["speeches"]):
                        relevant["speeches"].append((neighbor_id, neighbor_attrs))
        
        return relevant
    
    def _build_query_context(self, relevant_nodes: Dict[str, List], graph: nx.DiGraph, query_type: str) -> QueryContext:
        """
        Build a comprehensive context for the LLM based on relevant nodes.
        
        Args:
            relevant_nodes: Dictionary of relevant nodes by type
            graph: The full narrative graph
            query_type: Type of query being processed
            
        Returns:
            QueryContext object with organized context information
        """
        # Process characters with their attributes
        characters = {}
        for node_id, attrs in relevant_nodes["characters"]:
            characters[node_id] = {
                "description": attrs.get('description', ''),
                "traits": attrs.get('traits', []),
                "first_seen": attrs.get('first_seen', None),
                "last_seen": attrs.get('last_seen', None)
            }
        
        # Process objects
        objects = {}
        for node_id, attrs in relevant_nodes["objects"]:
            objects[node_id] = {
                "description": attrs.get('description', ''),
                "location": attrs.get('location', None),
                "state": attrs.get('state', None)
            }
        
        # Process actions
        actions = []
        for node_id, attrs in relevant_nodes["actions"]:
            # Find the subject and object of this action through graph connections
            subject = None
            action_object = None
            
            for pred in graph.predecessors(node_id):
                pred_attrs = graph.nodes[pred]
                if pred_attrs.get('type') == 'character':
                    subject = pred
                
            for succ in graph.successors(node_id):
                succ_attrs = graph.nodes[succ]
                if succ_attrs.get('type') in ['object', 'character']:
                    action_object = succ
            
            actions.append({
                "id": node_id,
                "action": attrs.get('description', ''),
                "subject": subject,
                "object": action_object,
                "time": attrs.get('timestamp', None)
            })
        
        # Process dialogue/speech
        dialogue = []
        for node_id, attrs in relevant_nodes["speeches"]:
            # Find the speaker through graph connections
            speaker = None
            for pred in graph.predecessors(node_id):
                pred_attrs = graph.nodes[pred]
                if pred_attrs.get('type') == 'character':
                    speaker = pred
            
            dialogue.append({
                "id": node_id,
                "text": attrs.get('text', ''),
                "speaker": speaker,
                "time": attrs.get('start_time', None),
                "sentiment": attrs.get('sentiment', None)
            })
        
        # Process spatial information
        spatial_info = []
        # Extract spatial relationships between characters and objects
        for char_id in characters:
            for obj_id in objects:
                # Look for direct edges between character and object
                if graph.has_edge(char_id, obj_id):
                    edge_data = graph.get_edge_data(char_id, obj_id)
                    spatial_info.append({
                        "subject": char_id,
                        "object": obj_id,
                        "relation": edge_data.get('relation', 'near'),
                        "time": edge_data.get('timestamp', None)
                    })
        
        # Process emotional states over time
        emotional_states = {}
        for char_id in characters:
            emotional_states[char_id] = []
            # Find all emotion nodes connected to this character
            for node_id, attrs in relevant_nodes["emotions"]:
                # Check if this emotion belongs to the character
                for pred in graph.predecessors(node_id):
                    if pred == char_id:
                        emotional_states[char_id].append({
                            "emotion": attrs.get('emotion', ''),
                            "intensity": attrs.get('intensity', None),
                            "time": attrs.get('timestamp', None),
                            "cause": attrs.get('cause', None)
                        })
        
        # Process events
        events = []
        for node_id, attrs in relevant_nodes["events"]:
            # Find characters involved in this event
            involved_characters = []
            for edge in graph.in_edges(node_id):
                src = edge[0]
                if graph.nodes[src].get('type') == 'character':
                    involved_characters.append(src)
            
            events.append({
                "id": node_id,
                "description": attrs.get('description', ''),
                "time": attrs.get('timestamp', None),
                "location": attrs.get('location', None),
                "involved_characters": involved_characters
            })
        
        # Return the compiled context
        return QueryContext(
            characters=characters,
            objects=objects,
            actions=actions,
            dialogue=dialogue,
            spatial_info=spatial_info,
            emotional_states=emotional_states,
            events=events
        )
    
    def _generate_reasoning_and_answer(self, question: str, context: QueryContext, query_type: str) -> Tuple[str, str]:
        """
        Generate reasoning and answer using LLM based on the assembled context.
        
        Args:
            question: Original question
            context: Assembled context information
            query_type: Type of query being processed
            
        Returns:
            Tuple of (reasoning process, final answer)
        """
        # Convert context to a format suitable for the prompt
        context_str = self._format_context_for_prompt(context, query_type)
        
        # Build the prompt based on query type
        prompt_prefix = "You are an expert at analyzing video scenes and understanding narrative elements. "
        
        if query_type == "motivation":
            prompt_prefix += "Focus on character motivations, intentions, and the reasons behind their actions. "
        elif query_type == "relationship":
            prompt_prefix += "Focus on the relationships, dynamics, and connections between characters. "
        elif query_type == "emotional":
            prompt_prefix += "Focus on the emotional states, changes, and expressions of characters. "
        elif query_type == "subtext":
            prompt_prefix += "Focus on the subtext, implied meanings, and what's not explicitly stated. "
        elif query_type == "technique":
            prompt_prefix += "Focus on narrative techniques, visual storytelling, and directorial choices. "
        
        system_prompt = prompt_prefix + "Analyze the following scene information carefully."
        
        user_prompt = f"""
Question: {question}

Scene Context:
{context_str}

First, carefully analyze the relevant information and think through your reasoning step by step.
Then provide a concise, insightful answer that addresses the question directly.
"""
        
        # Generate response using the appropriate LLM client
        try:
            if self.llm_type == "llama":
                return self._generate_with_llama(system_prompt, user_prompt)
            else:
                return self._generate_with_openai(system_prompt, user_prompt)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return (f"Error generating reasoning: {str(e)}", 
                    f"I encountered an error analyzing this scene: {str(e)}")
    
    def _generate_with_llama(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        """
        Generate response with Llama model.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query with context
            
        Returns:
            Tuple of (reasoning, answer)
        """
        # Format the prompt for Llama
        formatted_prompt = f"""
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]
"""
        # Generate response
        response = self.llm(
            formatted_prompt,
            max_tokens=1024,
            temperature=self.temperature,
            stop=["</s>"]
        )
        
        # Extract generated text
        full_response = response["choices"][0]["text"].strip()
        
        # Split reasoning and answer - assuming the answer is the last paragraph
        paragraphs = full_response.split('\n\n')
        
        if len(paragraphs) > 1:
            reasoning = '\n\n'.join(paragraphs[:-1])
            answer = paragraphs[-1]
        else:
            # If there's just one paragraph, use it as both reasoning and answer
            reasoning = full_response
            answer = full_response
            
        return reasoning, answer
    
    def _generate_with_openai(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        """
        Generate response with OpenAI API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query with context
            
        Returns:
            Tuple of (reasoning, answer)
        """
        # Call OpenAI API
        response = self.llm.chat.completions.create(
            model=self.model_name if "gpt" in self.model_name else "gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
        )
        
        # Extract reasoning and answer
        full_response = response.choices[0].message.content
        
        # Split reasoning and answer - assuming the answer is the last paragraph
        paragraphs = full_response.split('\n\n')
        
        if len(paragraphs) > 1:
            reasoning = '\n\n'.join(paragraphs[:-1])
            answer = paragraphs[-1]
        else:
            # If there's just one paragraph, use it as both reasoning and answer
            reasoning = full_response
            answer = full_response
            
        return reasoning, answer
    
    def _extract_supporting_evidence(self, relevant_nodes: Dict[str, List], query_targets: Set[str]) -> List[Dict[str, Any]]:
        """
        Extract supporting evidence for the answer.
        
        Args:
            relevant_nodes: Dictionary of relevant nodes
            query_targets: Set of target entities from the query
            
        Returns:
            List of evidence items with source and relevance
        """
        evidence = []
        
        # Extract evidence from characters
        for node_id, attrs in relevant_nodes["characters"]:
            if any(target.lower() in str(node_id).lower() for target in query_targets) or not query_targets:
                evidence.append({
                    "type": "character",
                    "id": node_id,
                    "description": attrs.get('description', ''),
                    "relevance": "direct" if any(target.lower() in str(node_id).lower() for target in query_targets) else "contextual"
                })
        
        # Extract evidence from actions
        for node_id, attrs in relevant_nodes["actions"]:
            description = attrs.get('description', '')
            if any(target.lower() in description.lower() for target in query_targets) or not query_targets:
                evidence.append({
                    "type": "action",
                    "id": node_id,
                    "description": description,
                    "time": attrs.get('timestamp', None),
                    "relevance": "direct" if any(target.lower() in description.lower() for target in query_targets) else "contextual"
                })
        
        # Extract evidence from dialogue
        for node_id, attrs in relevant_nodes["speeches"]:
            text = attrs.get('text', '')
            if any(target.lower() in text.lower() for target in query_targets) or not query_targets:
                evidence.append({
                    "type": "dialogue",
                    "id": node_id,
                    "text": text,
                    "speaker": attrs.get('speaker', None),
                    "relevance": "direct" if any(target.lower() in text.lower() for target in query_targets) else "contextual"
                })
        
        return evidence
    
    def _format_context_for_prompt(self, context: QueryContext, query_type: str) -> str:
        """
        Format the context into a string suitable for the LLM prompt.
        
        Args:
            context: The assembled context
            query_type: Type of query being processed
            
        Returns:
            Formatted context string
        """
        sections = []
        
        # Format characters section
        if context.characters:
            char_section = "CHARACTERS:\n"
            for char_id, char_info in context.characters.items():
                char_section += f"- {char_id}: {char_info['description']}\n"
                if char_info['traits']:
                    char_section += f"  Traits: {', '.join(char_info['traits'])}\n"
            sections.append(char_section)
        
        # Format actions section - prioritize based on query type
        if context.actions:
            action_section = "ACTIONS:\n"
            for action in sorted(context.actions, key=lambda x: x.get('time', 0)):
                subject = action['subject'] if action['subject'] else "Unknown"
                obj = f" to/with {action['object']}" if action['object'] else ""
                time = f" at {action['time']}" if action['time'] else ""
                action_section += f"- {subject} {action['action']}{obj}{time}\n"
            sections.append(action_section)
        
        # Format dialogue section
        if context.dialogue:
            dialogue_section = "DIALOGUE:\n"
            for speech in sorted(context.dialogue, key=lambda x: x.get('time', 0)):
                speaker = speech['speaker'] if speech['speaker'] else "Unknown"
                time = f" [{speech['time']}]" if speech['time'] else ""
                dialogue_section += f"- {speaker}{time}: \"{speech['text']}\"\n"
            sections.append(dialogue_section)
        
        # Format spatial information
        if context.spatial_info:
            spatial_section = "SPATIAL RELATIONSHIPS:\n"
            for rel in context.spatial_info:
                time = f" at {rel['time']}" if rel['time'] else ""
                spatial_section += f"- {rel['subject']} is {rel['relation']} {rel['object']}{time}\n"
            sections.append(spatial_section)
        
        # Format emotional states - especially important for emotional queries
        if context.emotional_states and (query_type in ["emotional", "motivation", "relationship"] or query_type == "general"):
            emotion_section = "EMOTIONAL STATES:\n"
            for char_id, emotions in context.emotional_states.items():
                if emotions:
                    emotion_section += f"- {char_id}:\n"
                    for emotion in sorted(emotions, key=lambda x: x.get('time', 0)):
                        intensity = f" ({emotion['intensity']})" if emotion['intensity'] else ""
                        time = f" at {emotion['time']}" if emotion['time'] else ""
                        cause = f" due to {emotion['cause']}" if emotion['cause'] else ""
                        emotion_section += f"  * {emotion['emotion']}{intensity}{time}{cause}\n"
            sections.append(emotion_section)
        
        # Format events
        if context.events:
            event_section = "EVENTS:\n"
            for event in sorted(context.events, key=lambda x: x.get('time', 0)):
                time = f" at {event['time']}" if event['time'] else ""
                location = f" in {event['location']}" if event['location'] else ""
                chars = f" involving {', '.join(event['involved_characters'])}" if event['involved_characters'] else ""
                event_section += f"- {event['description']}{time}{location}{chars}\n"
            sections.append(event_section)
        
        # Join all sections
        return "\n".join(sections)