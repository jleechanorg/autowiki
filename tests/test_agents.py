"""
Tests for Chimera Agents.
"""

import pytest
from chimera.agents import RouterAgent, LiteratureMinerAgent, CriticAgent
from chimera.agents.base import AgentConfig


def test_base_agent_mock():
    agent = RouterAgent()
    result = agent.process_task("Test query about batteries")
    assert "agent" in result
    assert result["role"] == "router"
    assert len(result["result"]) > 20


def test_router_decomposition():
    router = RouterAgent()
    decomp = router.decompose_task("Analyze solid-state batteries")
    assert "decomposition" in decomp
    assert "query" in decomp


def test_critic_critique():
    critic = CriticAgent()
    critique = critic.critique("Solid-state batteries will reach 500Wh/kg by 2027.")
    assert "Major Concerns" in critique or "flaw" in critique.lower() or len(critique) > 50


def test_agent_message_passing():
    from chimera.agents.base import AgentMessage
    msg = AgentMessage(sender="Router", receiver="Miner", content="Find papers")
    assert msg.message_type == "task"
    assert "Router" in str(msg)