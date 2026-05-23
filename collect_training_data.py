#!/usr/bin/env python3
"""
Training Dataset Collector for Project Chimera.
Reads benchmark checkpoint, generates GNN topologies, and outputs training data as JSONL.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add chimera module to path
sys.path.insert(0, str(Path(__file__).parent))

from chimera.gnn import GNNTopologyGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Collect training data from benchmark checkpoints")
    parser.add_argument(
        "--checkpoint",
        default="benchmark_logs/checkpoint.json",
        help="Path to benchmark checkpoint.json"
    )
    parser.add_argument(
        "--output",
        default="training_data.jsonl",
        help="Output path for training data JSONL"
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load and parse the benchmark checkpoint JSON."""
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    return data


def extract_gnn_results(checkpoint_data: dict) -> list[dict]:
    """Extract GNN mode results from checkpoint data."""
    results = []
    for entry in checkpoint_data.get("results", []):
        query = entry.get("query", "")
        query_num = entry.get("query_num", "")
        gnn_mode = entry.get("modes", {}).get("gnn", {})

        if not gnn_mode:
            continue

        error_flag = gnn_mode.get("error_flag")
        if error_flag:
            # Skip queries with errors in GNN mode
            continue

        scores = gnn_mode.get("scores", {})
        quality_score = scores.get("overall")

        if quality_score is None:
            continue

        tokens = gnn_mode.get("tokens", 0)

        results.append({
            "query": query,
            "query_num": query_num,
            "quality_score": quality_score,
            "tokens": tokens,
            "dimensions": {
                "factual": scores.get("factual"),
                "comprehensive": scores.get("comprehensive"),
                "clarity": scores.get("clarity"),
                "useful": scores.get("useful"),
                "specific": scores.get("specific"),
            }
        })
    return results


def compute_reward(quality_score: float, sparsity: float, tokens: int) -> float:
    """Compute the reward metric: (quality_score * sparsity) / log(tokens+1)."""
    if tokens <= 0:
        tokens = 1
    return (quality_score * sparsity) / math.log(tokens + 1)


def collect_training_data(
    checkpoint_path: Path,
    output_path: Path,
    verbose: bool = True
) -> int:
    """
    Main collection pipeline.
    Returns the number of records written.
    """
    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = load_checkpoint(checkpoint_path)

    # Extract GNN results
    gnn_results = extract_gnn_results(checkpoint_data)
    if verbose:
        print(f"Found {len(gnn_results)} queries with valid GNN results")

    if not gnn_results:
        print("ERROR: No valid GNN results found in checkpoint", file=sys.stderr)
        return 0

    # Initialize GNN topology generator
    if verbose:
        print("Initializing GNNTopologyGenerator")
    gnn = GNNTopologyGenerator()

    # Open output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w") as f:
        for i, result in enumerate(gnn_results):
            query = result["query"]
            query_num = result["query_num"]
            quality_score = result["quality_score"]
            tokens = result["tokens"]
            dimensions = result["dimensions"]

            # Generate topology for this query
            topology = gnn.generate_topology(query)

            # Extract topology metrics
            sparsity = topology.get("sparsity", 0.0)
            estimated_tokens = topology.get("estimated_tokens", tokens)

            # Compute reward
            reward = compute_reward(quality_score, sparsity, tokens)

            # Build training record
            record = {
                "query": query,
                "query_id": query_num,
                "topology": {
                    "num_nodes": topology.get("num_nodes", 0),
                    "num_edges": topology.get("num_edges", 0),
                    "sparsity": sparsity,
                    "estimated_tokens": estimated_tokens,
                    "graph": topology.get("graph", {}),
                    "node_importance": topology.get("node_importance", {}),
                },
                "quality_score": quality_score,
                "reward": round(reward, 6),
                "tokens": tokens,
                "dimensions": dimensions,
            }

            # Write as JSON line
            f.write(json.dumps(record) + "\n")
            written += 1

            # Progress every 3 queries
            if verbose and (i + 1) % 3 == 0:
                print(f"  Processed {i + 1}/{len(gnn_results)} queries")

    if verbose:
        print(f"Done. Wrote {written} records to {output_path}")

    return written


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    written = collect_training_data(checkpoint_path, output_path)

    if written == 0:
        sys.exit(1)

    print(f"\nOutput: {written} training records -> {output_path.resolve()}")


if __name__ == "__main__":
    main()