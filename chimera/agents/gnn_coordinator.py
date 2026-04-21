"""
GNN Coordinator Agent - Manages dynamic topology and graph decisions.
"""

from .base import BaseAgent, AgentConfig
from typing import List, Dict


class GNNCoordinatorAgent(BaseAgent):
    """Uses GNN output to orchestrate agent interactions."""

    def __init__(self, name: str = "GNNCoordinator"):
        config = AgentConfig(
            name=name,
            role="gnn_coordinator",
            system_prompt="""You are the Graph Neural Network coordinator.
You receive GNN-generated topologies and translate them into execution plans.
You decide:
- Which agents participate
- Communication patterns (debate rounds, sequential handoff, parallel)
- Token budget allocation
- When to trigger critique loops or fact-checking
Optimize for quality vs. cost."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def create_execution_plan(self, gnn_output: Dict, task: str) -> Dict:
        prompt = f"""GNN Topology Output: {gnn_output}
Task: {task}

Create a detailed execution plan including:
- Participating agents and order
- Communication protocol (e.g., 2 debate rounds, then synthesize)
- Estimated token usage
- Quality checkpoints"""
        plan = self._call_llm(prompt)
        return {"plan": plan, "gnn_input": gnn_output}