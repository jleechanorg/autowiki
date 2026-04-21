"""
Integration tests for SwarmOrchestrator.
"""

import pytest
from chimera.orchestrator import SwarmOrchestrator


@pytest.fixture
def orch():
    return SwarmOrchestrator(mock_mode=True)


def test_single_mode(orch):
    result = orch.run_research("Test query", mode="single")
    assert result["mode"] == "single"
    assert result["quality_score"] < 8.0  # Baseline should be lower


def test_fixed_vs_gnn(orch):
    result_fixed = orch.run_research("Battery tech 2026", mode="fixed")
    result_gnn = orch.run_research("Battery tech 2026", mode="gnn")

    assert result_gnn["quality_score"] > result_fixed["quality_score"]
    assert result_gnn["total_tokens_estimate"] < result_fixed["total_tokens_estimate"] * 1.1


def test_knowledge_graph_persistence(orch):
    result = orch.run_research("Test persistence", mode="gnn")
    stats = orch.kg.get_stats()
    assert stats["research_outputs"] >= 1