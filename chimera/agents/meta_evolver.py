"""
Meta-Evolver Agent - Proposes new roles, patterns, and architectural improvements.
"""

from .base import BaseAgent, AgentConfig
from typing import Dict, List


class MetaEvolverAgent(BaseAgent):
    """Analyzes swarm performance and proposes new agent roles or patterns."""

    def __init__(self, name: str = "MetaEvolver"):
        config = AgentConfig(
            name=name,
            role="meta_evolver",
            system_prompt="""You are a meta-level researcher who analyzes the performance of multi-agent systems.
Your job is to:
1. Identify recurring failure modes or missing capabilities in the agent swarm
2. Propose new specialist roles that would improve quality or efficiency
3. Suggest pattern improvements for agent communication and orchestration
4. Detect when existing agents are underperforming or misaligned
Be constructive and specific. Propose roles with clear responsibilities and interfaces."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def analyze_and_evolve(self, swarm_history: str, quality_scores: List[float]) -> Dict:
        prompt = f"""Analyze the multi-agent swarm's performance and propose improvements.

Swarm History:
{swarm_history[:3000]}

Quality Scores: {quality_scores}

Provide:
1. Identified failure modes (2-3 specific issues)
2. Proposed new agent roles (with responsibilities)
3. Suggested pattern changes
4. Expected quality improvement"""
        result = self._call_llm(prompt)
        return {"evolution_proposal": result, "quality_scores": quality_scores}
