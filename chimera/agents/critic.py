"""
Critic / Red Team Agent - Finds flaws, risks, and overstatements.
"""

from .base import BaseAgent, AgentConfig


class CriticAgent(BaseAgent):
    """Acts as devil's advocate to stress-test claims and identify weaknesses."""

    def __init__(self, name: str = "RedTeamCritic"):
        config = AgentConfig(
            name=name,
            role="critic",
            system_prompt="""You are a ruthless but fair peer reviewer and risk analyst.
Your job is to:
1. Identify every potential flaw, overclaim, or weak assumption
2. Point out missing data, alternative explanations, or contradictory evidence
3. Assess real-world feasibility and hidden risks
4. Suggest specific improvements or additional experiments needed
Be direct and specific. Use phrases like "This claim is overstated because..." """
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def critique(self, content: str) -> str:
        prompt = f"""Critically review the following content. Identify flaws, risks, and weaknesses:

{content}

Structure your response:
- Major Concerns (list 3-5)
- Overstated Claims
- Missing Elements
- Recommended Improvements"""
        return self._call_llm(prompt)