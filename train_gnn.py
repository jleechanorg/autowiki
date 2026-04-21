"""
GNN Training Script for Chimera Topology Generator.

This script:
1. Runs 5 research queries through the GNN pipeline
2. Tracks agent selection, graph structure, quality_score, estimated_tokens
3. Updates GNN weights to maximize (quality_score * sparsity) / log(tokens+1)
4. Saves trained weights to gnn_trained.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from typing import Dict, List, Tuple
import os
import sys

# Add chimera to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chimera.gnn import GNNTopologyGenerator


# Research queries for training
TRAINING_QUERIES = [
    "What are the latest advances in solid-state batteries for electric vehicles?",
    "How does transformer attention mechanism work in modern LLMs?",
    "What are the key challenges in fusion energy commercialization?",
    "Explain the current state of quantum computing supremacy experiments.",
    "What progress has been made in AI alignment research in 2025-2026?"
]


def compute_reward(topology: Dict, quality_score: float) -> float:
    """
    Compute reward for the GNN optimization.
    Reward = (quality_score * sparsity) / log(tokens+1)
    Higher sparsity and quality with lower tokens = higher reward.
    """
    tokens = topology.get("estimated_tokens", 10000)
    sparsity = topology.get("sparsity", 0.5)

    # Avoid log(0)
    token_penalty = np.log(max(tokens, 1) + 1)
    reward = (quality_score * sparsity) / token_penalty

    return reward


def generate_mock_quality_score(query: str, topology: Dict) -> float:
    """
    Generate a hardcoded quality score for training.
    In production, this would come from actual task execution results.
    """
    # Use query hash for determinism but variation
    base_score = 7.5 + (hash(query) % 30) / 10.0  # 7.5 - 10.5

    # Boost for good topologies (sparse but connected)
    num_nodes = topology.get("num_nodes", 5)
    num_edges = topology.get("num_edges", 0)

    # Ideal: 5-8 nodes with 3-6 edges
    if 5 <= num_nodes <= 8 and 3 <= num_edges <= 6:
        base_score += 0.5
    elif num_nodes > 10:
        base_score -= 0.3  # Too many agents = overhead
    elif num_edges > num_nodes * 2:
        base_score -= 0.3  # Too dense

    return min(max(base_score, 5.0), 10.0)


def collect_training_samples(gnn: GNNTopologyGenerator, queries: List[str]) -> List[Dict]:
    """Run queries through GNN and collect training samples."""
    samples = []

    for query in queries:
        # Generate topology
        topology = gnn.generate_topology(query, num_agents=11)

        # Get quality score (mock for now)
        quality_score = generate_mock_quality_score(query, topology)

        # Compute reward
        reward = compute_reward(topology, quality_score)

        sample = {
            "query": query,
            "topology": topology,
            "quality_score": quality_score,
            "estimated_tokens": topology["estimated_tokens"],
            "sparsity": topology["sparsity"],
            "reward": reward,
            "num_nodes": topology["num_nodes"],
            "num_edges": topology["num_edges"],
            "selected_agents": [d["type"] for _, d in
                               zip(range(topology["num_nodes"]),
                                   topology["graph"].get("nodes", []))]
        }
        samples.append(sample)

        print(f"  Query: {query[:50]}...")
        print(f"    Nodes: {topology['num_nodes']}, Edges: {topology['num_edges']}, "
              f"Sparsity: {topology['sparsity']:.3f}")
        print(f"    Tokens: {topology['estimated_tokens']}, Quality: {quality_score:.2f}, "
              f"Reward: {reward:.4f}")

    return samples


class GNNTrainer:
    """Training loop for the GNN topology generator."""

    def __init__(self, gnn: GNNTopologyGenerator, learning_rate: float = 0.001):
        self.gnn = gnn
        self.model = gnn.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)
        self.training_history = []

    def compute_loss(self, topology: Dict, quality_score: float) -> torch.Tensor:
        """
        Compute loss for a single sample.
        Loss = -reward (we want to maximize reward)
        Plus regularization for model complexity.
        """
        reward = compute_reward(topology, quality_score)
        # Negative because we minimize loss but want to maximize reward
        base_loss = -reward

        # L2 regularization on model weights
        l2_reg = 0.0001 * sum(p.norm(2).item() for p in self.model.parameters())

        total_loss = base_loss + l2_reg
        return torch.tensor(total_loss, requires_grad=True)

    def train_epoch(self, samples: List[Dict]) -> Tuple[float, float]:
        """Train for one epoch over all samples."""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0

        for sample in samples:
            self.optimizer.zero_grad()

            # Get fresh topology prediction from current model
            topology = self.gnn.generate_topology(sample["query"], num_agents=11)
            quality_score = sample["quality_score"]

            # Compute loss
            loss = self.compute_loss(topology, quality_score)

            # Backprop
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_reward += compute_reward(topology, quality_score)

        avg_loss = total_loss / len(samples)
        avg_reward = total_reward / len(samples)

        return avg_loss, avg_reward

    def train(self, queries: List[str], num_epochs: int = 10) -> Dict:
        """Main training loop."""
        print("\n" + "="*60)
        print("GNN TRAINING LOOP")
        print("="*60)
        print(f"Queries: {len(queries)}, Epochs: {num_epochs}")
        print(f"Optimizer: Adam, LR: {self.optimizer.param_groups[0]['lr']}")
        print("="*60 + "\n")

        best_reward = float('-inf')
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Collect samples
            samples = collect_training_samples(self.gnn, queries)

            # Train one epoch
            avg_loss, avg_reward = self.train_epoch(samples)

            # Update scheduler
            self.scheduler.step()

            # Log
            print(f"  Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.training_history.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "lr": self.optimizer.param_groups[0]['lr']
            })

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                print(f"  [NEW BEST] Reward: {best_reward:.4f}")

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model with reward: {best_reward:.4f}")

        return {
            "best_reward": best_reward,
            "training_history": self.training_history,
            "final_model_state": self.model.state_dict()
        }


def save_trained_weights(model: nn.Sequential, filepath: str):
    """Save trained model weights."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_architecture": {
            "layers": [layer.__class__.__name__ for layer in model],
            "task_embedding_dim": 64,
            "num_agent_types": 11
        }
    }, filepath)
    print(f"\nSaved trained weights to: {filepath}")


def main():
    """Main training entry point."""
    print("\n" + "#"*60)
    print("# GNN TOPOLOGY GENERATOR - TRAINING SCRIPT")
    print("#"*60)

    # Initialize GNN
    gnn = GNNTopologyGenerator(num_agent_types=11)
    print(f"Initialized GNN with {gnn.num_agent_types} agent types")

    # Create trainer
    trainer = GNNTrainer(gnn, learning_rate=0.001)

    # Training queries
    queries = TRAINING_QUERIES

    # Run training
    results = trainer.train(queries, num_epochs=10)

    # Save weights
    module_dir = os.path.join(os.path.dirname(__file__), "chimera")
    output_path = os.path.join(module_dir, "gnn_trained.pt")
    save_trained_weights(gnn.model, output_path)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Reward: {results['best_reward']:.4f}")
    print(f"Training History:")
    for h in results['training_history']:
        print(f"  Epoch {h['epoch']}: Loss={h['avg_loss']:.4f}, "
              f"Reward={h['avg_reward']:.4f}, LR={h['lr']:.6f}")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
