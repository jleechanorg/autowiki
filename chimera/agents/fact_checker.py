"""
Fact Checker Agent - Verifies claims against sources and detects hallucinations.
"""

from .base import BaseAgent, AgentConfig


class FactCheckerAgent(BaseAgent):
    """Verifies factual accuracy and detects unsupported claims."""

    def __init__(self, name: str = "FactChecker"):
        config = AgentConfig(
            name=name,
            role="fact_checker",
            system_prompt="""You are a meticulous fact-checker with access to scientific databases.
For every claim, you:
1. Verify against known literature or logical consistency
2. Assign a confidence score (0-100%)
3. Flag any unsupported assertions or potential hallucinations
4. Provide corrected versions where possible
Be extremely precise. If something cannot be verified, say so explicitly."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def verify(self, claims: str, sources: str = "") -> str:
        prompt = f"""Verify the following claims for factual accuracy:

Claims: {claims}
Available Sources: {sources[:1500] if sources else 'None provided'}

For each major claim provide:
- Claim
- Verdict (True / Partially True / False / Unverifiable)
- Confidence (0-100)
- Explanation + Source"""
        return self._call_llm(prompt)