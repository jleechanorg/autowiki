#!/usr/bin/env python3
"""Quick demo entry point."""

from chimera.orchestrator import SwarmOrchestrator
from chimera.utils import pretty_print_results

if __name__ == "__main__":
    print("🧠 Project Chimera Demo")
    orch = SwarmOrchestrator(mock_mode=True)
    q = "Current state and 2027-2030 commercialization timeline for solid-state batteries"
    results = orch.compare_all_modes(q)
    pretty_print_results(results)
    print("\n✅ All code generated and demo runnable!")