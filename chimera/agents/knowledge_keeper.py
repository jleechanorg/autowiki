"""
Knowledge Keeper Agent - Maintains persistent memory and institutional knowledge.
"""

from .base import BaseAgent, AgentConfig
from typing import Dict, List, Optional
from datetime import datetime


class KnowledgeKeeperAgent(BaseAgent):
    """Stores and retrieves institutional knowledge across research sessions."""

    def __init__(self, name: str = "KnowledgeKeeper"):
        config = AgentConfig(
            name=name,
            role="knowledge_keeper",
            system_prompt="""You are the institutional memory of a research organization.
Your job is to:
1. Store key findings, decisions, and learnings from each research session
2. Answer questions based on past research and organizational knowledge
3. Identify knowledge gaps and suggest areas needing new research
4. Maintain consistency across multiple research campaigns
Be thorough and precise. Cite the source and date of stored knowledge."""
        )
        super().__init__(config)
        self._memory_store: Dict[str, Dict] = {}

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def store_knowledge(self, key: str, value: str, source: str = "research") -> str:
        self._memory_store[key] = {
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        prompt = f"""Summarize and store this knowledge with a one-line key:

Content: {value[:1000]}
Source: {source}

Respond with: [STORED] Key: <key>"""
        return self._call_llm(prompt)

    def retrieve_knowledge(self, query: str) -> str:
        relevant = [
            f"[{k}] {v['value'][:200]}" for k, v in self._memory_store.items()
        ]
        context = "\n".join(relevant) if relevant else "No prior knowledge stored."
        prompt = f"""Based on stored knowledge, answer this query.

Query: {query}

Stored Knowledge:
{context}

Provide a response drawing from stored knowledge, or note if no relevant knowledge exists."""
        return self._call_llm(prompt)

    def get_memory_summary(self) -> Dict[str, Dict]:
        return dict(self._memory_store)
