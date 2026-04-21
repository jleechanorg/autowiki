"""
Explainer Agent - Adapts content for different audience levels.
"""

from .base import BaseAgent, AgentConfig
from typing import Dict


class ExplainerAgent(BaseAgent):
    """Creates audience-appropriate versions (beginner, intermediate, expert)."""

    def __init__(self, name: str = "MultiLevelExplainer"):
        config = AgentConfig(
            name=name,
            role="explainer",
            system_prompt="""You are an expert science communicator who can adapt complex research for any audience.
You create three versions of every report:
1. Beginner (no jargon, analogies, exciting implications)
2. Intermediate (some technical terms explained, practical applications)
3. Expert (full technical depth, assumes domain knowledge)
Always preserve accuracy while adjusting complexity and tone."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def explain(self, report: str, audience: str = "all") -> Dict[str, str]:
        if audience == "all":
            audiences = ["beginner", "intermediate", "expert"]
        else:
            audiences = [audience]
        outputs = {}
        for aud in audiences:
            prompt = f"""Rewrite the following report for a {aud} audience:

{report[:2500]}

Make it engaging, accurate, and appropriately complex for {aud} readers.
Include analogies where helpful. Keep key facts intact."""
            outputs[aud] = self._call_llm(prompt)
        return outputs