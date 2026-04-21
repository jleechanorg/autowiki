"""
Persistent Knowledge Graph for the Living Wiki.
Uses NetworkX + simple JSON persistence.
"""

import networkx as nx
import json
from typing import Dict, List, Any
from datetime import datetime
import os


class KnowledgeGraph:
    """Stores entities, relations, and full research outputs for the Living Wiki."""

    def __init__(self, storage_path: str = "knowledge_graph.json"):
        self.G = nx.MultiDiGraph()
        self.storage_path = storage_path
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.G = nx.node_link_graph(data)

    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(nx.node_link_data(self.G), f, indent=2)

    def add_entry(self, query: str, result: Dict[str, Any]):
        """Add a full research output as a node with relations."""
        entry_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.G.add_node(entry_id,
                        type="research_output",
                        query=query,
                        timestamp=datetime.now().isoformat(),
                        quality_score=result.get("quality_score", 0),
                        mode=result.get("mode", "unknown"))

        # Link to key concepts (simple entity extraction)
        concepts = self._extract_concepts(query)
        for concept in concepts:
            if not self.G.has_node(concept):
                self.G.add_node(concept, type="concept")
            self.G.add_edge(entry_id, concept, relation="about")

        self._save()
        return entry_id

    def _extract_concepts(self, text: str) -> List[str]:
        # Very naive concept extraction (replace with LLM or NER in production)
        keywords = ["solid-state", "battery", "electrolyte", "energy density", "dendrite", "commercialization"]
        return [k for k in keywords if k.lower() in text.lower()]

    def query(self, concept: str) -> List[Dict]:
        """Return all research outputs related to a concept."""
        results = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") == "research_output":
                if concept.lower() in str(data.get("query", "")).lower():
                    results.append(data)
        return results

    def get_stats(self) -> Dict:
        return {
            "total_entries": self.G.number_of_nodes(),
            "research_outputs": len([n for n, d in self.G.nodes(data=True) if d.get("type") == "research_output"]),
            "concepts": len([n for n, d in self.G.nodes(data=True) if d.get("type") == "concept"])
        }