"""
Quality Gate Agent - Final approval and scoring before release.
"""

from .base import BaseAgent, AgentConfig
from typing import Dict


class QualityGateAgent(BaseAgent):
    """Final quality control and scoring."""

    def __init__(self, name: str = "QualityGate"):
        config = AgentConfig(
            name=name,
            role="quality_gate",
            system_prompt="""You are a senior quality assurance lead for research outputs.
You score outputs on a 1-10 scale across multiple dimensions and only approve those meeting high standards.
You are strict but fair. Provide specific, actionable feedback for improvement.
Dimensions: Accuracy, Comprehensiveness, Clarity, Usefulness, Novelty, Evidence Strength."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def evaluate(self, report: str, criteria: Dict = None) -> Dict:
        prompt = f"""Evaluate this research output against the following criteria:

Report: {report[:3000]}

Criteria: {criteria or 'Default: Accuracy, Depth, Clarity, Usefulness, Evidence'}

Provide:
- Scores (1-10) for each dimension
- Overall Score (1-10)
- Pass/Fail with justification
- Top 3 specific improvements needed"""
        result = self._call_llm(prompt)
        return {"evaluation": result, "approved": "Pass" in result or "Approved" in result}