"""
Router Agent - Decomposes tasks and routes to appropriate agents.
"""

from .base import BaseAgent, AgentConfig
from typing import List, Dict


class RouterAgent(BaseAgent):
    """Routes and decomposes complex tasks into sub-tasks."""

    def __init__(self, name: str = "TaskRouter"):
        config = AgentConfig(
            name=name,
            role="router",
            system_prompt="""You are an expert task router and project manager for a research team.
Your job is to break down complex research queries into clear, actionable sub-tasks.
Assign each sub-task to the most suitable specialist: Literature Miner, Domain Expert, Critic, Fact Checker, etc.
Output in structured JSON format with tasks, assigned roles, and dependencies."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def decompose_task(self, query: str) -> Dict:
        """Break down the research query."""
        prompt = f"""Decompose this research task into 5-8 sub-tasks with assigned roles and dependencies:

Query: {query}

Output JSON:
{{
  "subtasks": [
    {{"id": 1, "task": "...", "role": "literature_miner", "depends_on": []}},
    ...
  ],
  "execution_order": [1, 2, 3, ...]
}}"""
        result = self._call_llm(prompt)
        # In production, parse JSON properly
        return {"decomposition": result, "query": query}