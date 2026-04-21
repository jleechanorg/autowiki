"""
Base Agent class for Project Chimera.
All specialized agents inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    message_type: str = "task"  # task, result, critique, etc.
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    role: str
    model: str = "minimax-m2.7"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""
    tools: List[str] = []


class BaseAgent(ABC):
    """
    Abstract base class for all Chimera agents.
    Each agent has a role, can process messages, and communicate.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid.uuid4())[:8]
        self.name = config.name
        self.role = config.role
        self.history: List[AgentMessage] = []
        self._llm_client = None  # Will be set by orchestrator or mock

    def set_llm_client(self, client: Any):
        """Inject LLM client (real or mock)."""
        self._llm_client = client

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent's role."""
        pass

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Call the LLM with the given prompt. Uses mock if no client."""
        if self._llm_client is None:
            return self._mock_llm_response(prompt)
        try:
            system_prompt = self.get_system_prompt()
            messages = [{"role": "user", "content": prompt}]
            # MiniMaxClient uses Anthropic /v1/messages endpoint
            if hasattr(self._llm_client, "messages_create"):
                resp = self._llm_client.messages_create(
                    messages=messages,
                    system=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **kwargs,
                )
                # Handle thinking block if present
                content = resp.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            return block["text"]
                    return str(content[0]) if content else ""
                return str(content)
            # Fallback: OpenAI-style chat.completions
            response = self._llm_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM Error] {str(e)}"

    def _mock_llm_response(self, prompt: str) -> str:
        """Fallback mock response for testing without API key."""
        role_responses = {
            "router": "Task decomposed into: [Literature Review], [Technical Analysis], [Market Impact]. Priority: High.",
            "literature_miner": "Found 12 relevant papers. Key findings: Solid-state electrolytes show promise with 450Wh/kg density. Citations: [1,2,3].",
            "domain_expert": "Technical analysis: Current solid-state batteries face dendrite formation issues. Proposed solution: LLZO ceramic separator with AI-optimized doping.",
            "critic": "Potential flaw: Dendrite mitigation claims lack long-term cycling data (>500 cycles). Risk: Overstated energy density by 15%.",
            "fact_checker": "Verified: 3 papers confirm 450Wh/kg. 2 papers show dendrite issues in similar setups. Overall: 85% factual accuracy.",
            "synthesizer": "Integrated report: Promising technology with 2-3 year commercialization timeline. Key barriers: manufacturing scale-up.",
            "quality_gate": "Score: 8.2/10. Strengths: Well-sourced. Weaknesses: Limited economic analysis. Approved with minor revisions.",
            "explainer": "Beginner: Solid-state batteries are safer and hold more energy than regular ones. Expert: Uses ceramic separators instead of liquid.",
            "gnn_coordinator": "Optimal topology: 5 miners → 3 experts → 2 critics → 1 synthesizer. Sparsity: 0.65. Estimated tokens: 12.4k."
        }
        return role_responses.get(self.role, f"Processed task for {self.role}. Result: Task completed successfully.")

    def send_message(self, receiver: str, content: str, msg_type: str = "task") -> AgentMessage:
        """Create and log a message."""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type
        )
        self.history.append(msg)
        return msg

    def receive_message(self, message: AgentMessage) -> str:
        """Process incoming message and generate response."""
        self.history.append(message)
        prompt = f"Received from {message.sender} ({message.message_type}): {message.content}\n\nRespond as {self.role}."
        response = self._call_llm(prompt)
        return response

    def process_task(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main entry point for agent to process a task."""
        prompt = f"Task: {task}\nContext: {context or {}}"
        result = self._call_llm(prompt)
        return {
            "agent": self.name,
            "role": self.role,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    def __repr__(self):
        return f"<{self.role}Agent: {self.name} ({self.id})>"