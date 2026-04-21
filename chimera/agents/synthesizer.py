"""
Synthesizer Agent - Integrates all inputs into a coherent final output.
"""

from .base import BaseAgent, AgentConfig
from typing import Dict


class SynthesizerAgent(BaseAgent):
    """Combines research, analysis, critiques, and facts into polished output."""

    def __init__(self, name: str = "ChiefSynthesizer"):
        config = AgentConfig(
            name=name,
            role="synthesizer",
            system_prompt="""You are a master research synthesizer and science writer.
You excel at integrating diverse inputs (literature, expert analysis, critiques, fact-checks) into:
- Clear, well-structured reports
- Balanced perspectives that acknowledge trade-offs
- Actionable conclusions with realistic timelines
- Professional tone suitable for executives or academic publication
Always resolve conflicts between sources and highlight the most credible evidence."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def synthesize(self, inputs: Dict[str, str]) -> str:
        prompt = f"""Synthesize the following research components into a final report:

Literature Review: {inputs.get('literature', 'N/A')[:1200]}
Expert Analysis: {inputs.get('expert', 'N/A')[:1200]}
Critiques: {inputs.get('critiques', 'N/A')[:1000]}
Fact Check: {inputs.get('facts', 'N/A')[:800]}

Produce a comprehensive, balanced report with:
1. Executive Summary
2. Key Findings
3. Technical Assessment
4. Risks & Limitations
5. Recommendations & Timeline
6. References"""
        return self._call_llm(prompt)