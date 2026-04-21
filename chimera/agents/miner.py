"""
Literature Miner Agent - Searches and summarizes academic literature.
"""

from .base import BaseAgent, AgentConfig


class LiteratureMinerAgent(BaseAgent):
    """Specializes in finding and summarizing research papers."""

    def __init__(self, name: str = "LitMiner"):
        config = AgentConfig(
            name=name,
            role="literature_miner",
            system_prompt="""You are a world-class academic researcher and literature miner.
Given a research topic, you:
1. Identify the most relevant, high-impact papers (real or plausible recent ones)
2. Extract key findings, methodologies, and limitations
3. Provide proper citations in APA format
4. Highlight contradictions or gaps in the literature
Always be precise and evidence-based."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def mine_literature(self, topic: str, num_papers: int = 8) -> str:
        prompt = f"""Research Topic: {topic}
Find and summarize the top {num_papers} most relevant papers from 2023-2026.
For each paper include: Title, Authors, Year, Key Finding, Methodology, Limitation.
Format as a structured report."""
        return self._call_llm(prompt)