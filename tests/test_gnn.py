"""
Tests for GNN Topology Generator.
"""

from chimera.gnn import GNNTopologyGenerator


def test_gnn_topology_generation():
    gnn = GNNTopologyGenerator()
    topo = gnn.generate_topology("solid-state batteries commercialization", num_agents=11)

    assert "graph" in topo
    assert topo["num_nodes"] >= 5
    assert 0.5 < topo["sparsity"] <= 1.0
    assert topo["estimated_tokens"] > 5000


def test_topology_visualization():
    gnn = GNNTopologyGenerator()
    topo = gnn.generate_topology("test task")
    viz = gnn.visualize_topology(topo)
    assert "GNN-GENERATED TOPOLOGY" in viz
    assert "→" in viz or "Edges" in viz