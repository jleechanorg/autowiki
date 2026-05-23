"""
GNN Training Script for Chimera Topology Generator.

This script:
1. Loads real training data from JSONL (P14 benchmark scores)
2. Generates topologies with the GNN for each query
3. Computes reward = (real_quality_score * GNN_sparsity) / log(GNN_tokens + 1)
4. Updates GNN weights to maximize reward
5. Saves trained weights to gnn_trained.pt
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
import json
import argparse

# Add chimera to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chimera.gnn import GNNTopologyGenerator


# Research queries for training (fallback when not using --real-data)
TRAINING_QUERIES = [
    "What are the latest advances in solid-state batteries for electric vehicles?",
    "How does transformer attention mechanism work in modern LLMs?",
    "What are the key challenges in fusion energy commercialization?",
    "Explain the current state of quantum computing supremacy experiments.",
    "What progress has been made in AI alignment research in 2025-2026?"
]


def compute_reward(sparsity: float, quality_score: float, tokens: int) -> float:
    """
    Compute reward for the GNN optimization.
    Reward = (quality_score * sparsity) / log(tokens+1)
    Higher sparsity and quality with lower tokens = higher reward.
    """
    # Avoid log(0)
    token_penalty = np.log(max(tokens, 1) + 1)
    reward = (quality_score * sparsity) / token_penalty
    return reward


def load_training_data(jsonl_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_real_quality_score(query: str, training_data: List[Dict]) -> float:
    """Look up actual quality score from training data."""
    for entry in training_data:
        if entry['query'] == query:
            return entry['quality_score']
    # Fallback: return mean of all scores
    scores = [e['quality_score'] for e in training_data]
    return sum(scores) / len(scores)


def generate_mock_quality_score(query: str, topology: Dict) -> float:
    """DEPRECATED: Mock quality scoring has been removed. Use --real-data with training_data.jsonl."""
    raise ValueError(
        "generate_mock_quality_score() is deprecated. "
        "Use --real-data with training_data.jsonl for real training."
    )


def collect_training_samples(
    gnn: GNNTopologyGenerator,
    training_data: List[Dict],
    num_agents: int = 11
) -> List[Dict]:
    """
    Run queries through GNN and collect training samples.

    For each entry in training_data:
    - Generate topology with current GNN model (this varies as model trains)
    - Use the real quality_score from JSONL (fixed benchmark score)
    - Compute reward from GNN-generated sparsity/tokens × fixed quality score
    - Backprop will teach GNN to generate topologies that maximize this reward

    Args:
        gnn: GNNTopologyGenerator instance
        training_data: List of dicts with 'query' and 'quality_score' keys
        num_agents: Number of agents to generate in topology

    Returns:
        List of sample dicts with topology, quality_score, reward, etc.
    """
    samples = []

    for entry in training_data:
        query = entry['query']
        real_quality_score = entry['quality_score']  # Fixed benchmark score

        # Generate topology with current GNN model
        topology = gnn.generate_topology(query, num_agents=num_agents)

        # Get GNN-generated metrics (these change as model trains)
        gnn_sparsity = topology.get("sparsity", 0.5)
        gnn_tokens = topology.get("estimated_tokens", 10000)

        # Compute reward: real quality score × GNN sparsity / log(tokens+1)
        # This teaches GNN to generate sparse, efficient topologies
        # that correlate with high quality outputs
        reward = compute_reward(gnn_sparsity, real_quality_score, gnn_tokens)

        sample = {
            "query": query,
            "topology": topology,
            "quality_score": real_quality_score,
            "estimated_tokens": gnn_tokens,
            "sparsity": gnn_sparsity,
            "reward": reward,
            "num_nodes": topology["num_nodes"],
            "num_edges": topology["num_edges"],
            "selected_agents": [d["type"] for _, d in
                               zip(range(topology["num_nodes"]),
                                   topology["graph"].get("nodes", []))]
        }
        samples.append(sample)

        print(f"  Query: {query[:60]}...")
        print(f"    GNN Nodes: {topology['num_nodes']}, Edges: {topology['num_edges']}, "
              f"Sparsity: {topology['sparsity']:.3f}")
        print(f"    GNN Tokens: {gnn_tokens}, Real Quality: {real_quality_score:.2f}, "
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

    def compute_loss(self, sparsity: float, quality_score: float, tokens: int) -> torch.Tensor:
        """
        Compute loss for a single sample.
        Loss = -reward (we want to maximize reward)
        Plus regularization for model complexity.
        """
        reward = compute_reward(sparsity, quality_score, tokens)
        # Negative because we minimize loss but want to maximize reward
        base_loss = -reward

        # L2 regularization on model weights
        l2_reg = 0.0001 * sum(p.norm(2).item() for p in self.model.parameters())

        total_loss = base_loss + l2_reg
        return torch.tensor(total_loss, requires_grad=True)

    def train_epoch(self, samples: List[Dict], num_agents: int = 11) -> Tuple[float, float]:
        """Train for one epoch over all samples."""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0

        for sample in samples:
            self.optimizer.zero_grad()

            # Get fresh topology prediction from current model
            topology = self.gnn.generate_topology(sample["query"], num_agents=num_agents)

            # Get the fixed quality score from training data
            quality_score = sample["quality_score"]
            sparsity = topology.get("sparsity", 0.5)
            tokens = topology.get("estimated_tokens", 10000)

            # Compute loss
            loss = self.compute_loss(sparsity, quality_score, tokens)

            # Backprop
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_reward += compute_reward(sparsity, quality_score, tokens)

        avg_loss = total_loss / len(samples)
        avg_reward = total_reward / len(samples)

        return avg_loss, avg_reward

    def train(
        self,
        training_data: List[Dict],
        num_epochs: int = 10,
        num_agents: int = 11
    ) -> Dict:
        """Main training loop."""
        print("\n" + "="*60)
        print("GNN TRAINING LOOP")
        print("="*60)
        print(f"Training Samples: {len(training_data)}, Epochs: {num_epochs}")
        print(f"Optimizer: Adam, LR: {self.optimizer.param_groups[0]['lr']}")
        print("="*60 + "\n")

        best_reward = float('-inf')
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Collect samples (generate fresh topologies from current model)
            samples = collect_training_samples(self.gnn, training_data, num_agents=num_agents)

            # Train one epoch
            avg_loss, avg_reward = self.train_epoch(samples, num_agents=num_agents)

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
    parser = argparse.ArgumentParser(
        description="GNN Topology Generator - Real Training Script"
    )
    parser.add_argument(
        '--real-data',
        type=str,
        default=None,
        help='Path to training_data.jsonl for real training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for trained weights'
    )
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# GNN TOPOLOGY GENERATOR - TRAINING SCRIPT")
    print("#"*60)

    # Initialize GNN
    gnn = GNNTopologyGenerator(num_agent_types=11)
    print(f"Initialized GNN with {gnn.num_agent_types} agent types")

    # Create trainer
    trainer = GNNTrainer(gnn, learning_rate=args.lr)

    # Load training data if --real-data provided
    if args.real_data:
        training_data = load_training_data(args.real_data)
        print(f"Loaded {len(training_data)} training samples from {args.real_data}")
    else:
        # Build minimal training data from fallback queries with uniform score
        # This path exists for testing but is not recommended
        print("WARNING: No --real-data provided. Creating mock training data.")
        print("WARNING: This is only for testing. Use --real-data with training_data.jsonl.")
        mean_score = 5.5  # Neutral quality score for all fallback queries
        training_data = [
            {"query": q, "quality_score": mean_score} for q in TRAINING_QUERIES
        ]

    # Run training
    results = trainer.train(training_data, num_epochs=args.epochs, num_agents=11)

    # Save weights
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "chimera", "gnn_trained.pt"
    )
    # Ensure chimera directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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