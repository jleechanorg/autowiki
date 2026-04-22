"""
Graph Neural Network Topology Generator for Chimera.
Uses a lightweight neural network to predict optimal agent communication graphs.
"""

import contextlib
import random
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide mock implementations when torch unavailable
    class MockTorch:
        no_grad = contextlib.nullcontext
        Tensor = type('Tensor', (), {})
        @staticmethod
        def randn(*args, **kwargs):
            random.seed(42)
            class FakeTensor:
                def squeeze(self): return self
                def tolist(self): return [0.1] * 64
                def __getitem__(self, key): return FakeTensor()
            return FakeTensor()
        @staticmethod
        def sigmoid(x):
            class FakeTensor:
                def __getitem__(self, key): return FakeTensor()
                def nonzero(self, **kwargs): return ([0,1,2],)
                def tolist(self): return [0.5, 0.6, 0.7]
            return FakeTensor()
        @staticmethod
        def abs(x): return x
    torch = MockTorch()
    nn = type('nn', (), {'Linear': type('Linear', (), {'__init__': lambda s,*a: None})})()

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple


class GNNTopologyGenerator:
    """
    Lightweight GNN that outputs task-specific agent topologies.
    In production, this would be a trained Graph Attention Network (GAT).
    Here we use a simplified but realistic simulation.
    Loads pre-trained weights from gnn_trained.pt if available.
    """

    def __init__(self, num_agent_types: int = 11):
        self.num_agent_types = num_agent_types
        self.agent_types = [
            "router", "literature_miner", "domain_expert", "critic",
            "fact_checker", "synthesizer", "quality_gate", "explainer",
            "gnn_coordinator", "meta_evolver", "knowledge_keeper"
        ]
        # Simple learned weights (in real system, loaded from checkpoint)
        self.task_embedding_dim = 64
        self.model = self._build_simple_gnn()

        # Load pre-trained weights if available
        self._load_weights()

    def _load_weights(self):
        """Load pre-trained weights if gnn_trained.pt exists."""
        # Get the directory of this file and look for trained weights
        module_dir = os.path.dirname(os.path.abspath(__file__))
        trained_path = os.path.join(module_dir, "gnn_trained.pt")

        if os.path.exists(trained_path):
            try:
                checkpoint = torch.load(trained_path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[GNNTopologyGenerator] Loaded trained weights from {trained_path}")
            except Exception as e:
                print(f"[GNNTopologyGenerator] Failed to load weights from {trained_path}: {e}")
                print("  Using random initialization (untrained)")
        else:
            print(f"[GNNTopologyGenerator] No trained weights found at {trained_path}")
            print("  Using random initialization (untrained)")

    def _build_simple_gnn(self):
        """A tiny neural net that predicts edges and node importance."""
        if not TORCH_AVAILABLE:
            # Return a mock object
            class MockModel:
                def __call__(self, x):
                    class FakeOutput:
                        def squeeze(self): return FakeOutput()
                        def __getitem__(self, key): return FakeOutput()
                    return FakeOutput()
            return MockModel()
        return nn.Sequential(
            nn.Linear(self.task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_agent_types * 2)  # node importance + edge logits
        )

    def _embed_task(self, task: str) -> torch.Tensor:
        """Create a simple embedding from task text (hash-based for demo)."""
        # In production: use sentence-transformers or LLM embedding
        random.seed(hash(task) % 2**32)
        return torch.randn(1, self.task_embedding_dim)

    def generate_topology(self, task: str, num_agents: int = 11) -> Dict:
        """
        Generate a sparse, high-quality communication graph for the task.
        Returns node selection, edge list, and metadata.
        """
        task_emb = self._embed_task(task)
        with torch.no_grad():
            logits = self.model(task_emb).squeeze()

        # Node importance scores - handle mock tensor
        if not TORCH_AVAILABLE:
            node_scores = [0.7, 0.8, 0.6, 0.9, 0.5, 0.85, 0.75, 0.65, 0.7, 0.6, 0.55][:self.num_agent_types]
            selected_nodes = [0, 1, 2, 3, 5, 6, 7]  # Fixed subset for demo
        else:
            node_scores = torch.sigmoid(logits[:self.num_agent_types])
            selected_nodes = (node_scores > 0.4).nonzero(as_tuple=True)[0].tolist()

        # Edge prediction (simplified attention-like)
        if not TORCH_AVAILABLE:
            edges = [(0,1), (0,2), (1,5), (2,5), (5,6)]
            edge_probs = [[0.0]*len(selected_nodes) for _ in range(len(selected_nodes))]
        else:
            edge_logits = logits[self.num_agent_types:].view(self.num_agent_types, -1)[:len(selected_nodes), :len(selected_nodes)]
            edge_probs = torch.sigmoid(edge_logits)
            edges = (edge_probs > 0.55).nonzero(as_tuple=False).tolist()

        # Create NetworkX graph
        G = nx.DiGraph()
        for i, node_idx in enumerate(selected_nodes):
            score = node_scores[i] if isinstance(node_scores, list) else float(node_scores[node_idx])
            G.add_node(i, type=self.agent_types[node_idx], importance=score)

        for src, dst in edges:
            if src < len(selected_nodes) and dst < len(selected_nodes):
                G.add_edge(src, dst, weight=0.7)

        # Compute metrics
        sparsity = 1 - (len(G.edges) / (len(G.nodes) * (len(G.nodes) - 1))) if len(G.nodes) > 1 else 1.0
        estimated_tokens = int(8000 + len(G.nodes) * 1200 - sparsity * 3000)

        return {
            "graph": nx.node_link_data(G),
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "sparsity": round(sparsity, 3),
            "estimated_tokens": estimated_tokens,
            "node_importance": {i: round(float(node_scores[i].item()) if hasattr(node_scores[i], 'item') else float(node_scores[i]), 3) for i in range(len(selected_nodes))},
            "task_embedding": [0.1] * 8  # mock embedding
        }

    def visualize_topology(self, topology: Dict) -> str:
        """Return ASCII representation of the graph."""
        G = nx.node_link_graph(topology["graph"])
        lines = ["GNN-GENERATED TOPOLOGY:"]
        for node, data in G.nodes(data=True):
            lines.append(f"  [{node}] {data['type']} (imp={data['importance']:.2f})")
        lines.append("Edges:")
        for u, v, d in G.edges(data=True):
            lines.append(f"  {u} → {v} (w={d['weight']:.2f})")
        return "\n".join(lines)