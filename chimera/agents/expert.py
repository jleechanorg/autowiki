"""
Domain Expert Agent - Provides deep technical or domain-specific analysis.
"""

from .base import BaseAgent, AgentConfig


class DomainExpertAgent(BaseAgent):
    """Provides in-depth domain expertise (e.g., battery tech, AI, finance)."""

    def __init__(self, name: str = "DomainExpert", domain: str = "solid-state batteries"):
        config = AgentConfig(
            name=name,
            role="domain_expert",
            system_prompt=f"""You are a leading expert in {domain} with 20+ years experience.
Your analysis must be technically rigorous, cite underlying principles, and identify practical constraints.
Highlight trade-offs, current state-of-the-art, and realistic timelines to commercialization.
Use precise terminology but explain concepts clearly."""
        )
        super().__init__(config)
        self.domain = domain

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def analyze(self, topic: str, literature_summary: str = "") -> str:
        prompt = f"""Domain: {self.domain}
Topic: {topic}
Literature Context: {literature_summary[:2000] if literature_summary else 'N/A'}

Provide a deep technical analysis including:
- Current limitations and bottlenecks
- Most promising technical approaches
- Key performance metrics and trade-offs
- Realistic 2026-2030 outlook"""
        return self._call_llm(prompt)