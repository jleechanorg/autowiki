"""
Project Chimera: Neural Network of LLM Agents
A dynamic multi-agent system with Graph Neural Network topology learning.

Usage:
    from chimera.orchestrator import SwarmOrchestrator
    orch = SwarmOrchestrator(mock_mode=True)
    result = orch.run_research("Your research question here")
"""

__version__ = "0.1.0"
__author__ = "Jeffrey"

from .orchestrator import SwarmOrchestrator
from .gnn import GNNTopologyGenerator
from .knowledge_graph import KnowledgeGraph
from .judge import AIJudge

__all__ = [
    "SwarmOrchestrator",
    "GNNTopologyGenerator",
    "KnowledgeGraph",
    "AIJudge"
]