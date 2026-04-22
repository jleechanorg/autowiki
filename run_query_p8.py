#!/usr/bin/env python3
"""
Per-query benchmark runner for P8 parallel execution.
Takes query number (1-15), runs all 3 modes, writes result to checkpoint_q{N}.json
"""
import os
import sys
import json
import re
import time
from datetime import datetime

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/run_query_p8.log"

from chimera.orchestrator import SwarmOrchestrator

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.io/anthropic"
MODEL = "minimax-m2.7"

HARD_QUERIES = [
    "Design a comprehensive investment portfolio strategy for a 35-year-old with moderate risk tolerance, incorporating real estate, equities, fixed income, and alternative assets with specific allocation percentages and rebalancing rules",
    "Analyze the geopolitical implications of the Russia-Ukraine war on global energy markets through 2030, including supply chain disruptions, price volatility, and strategic responses by the EU, US, and China",
    "Evaluate the top 5 emerging programming languages in 2026 for building production AI systems — include concurrency models, typing systems, ecosystem maturity, and ML framework support",
    "Develop a detailed manufacturing scale-up plan for a startup going from 100 to 100,000 units/month, covering supply chain, equipment, workforce, quality control, and regulatory compliance",
    "Compare and critique the architectural patterns of 3 major AI frameworks (pick specific versions: e.g. LangChain 0.3, LlamaIndex 0.11, AutoGen 0.4) — what are the tradeoffs in agent orchestration, memory, and tool use?",
    "Create a competitive analysis for a B2B SaaS product in the project management space — include market size, top 5 competitors, differentiation strategy, and pricing model with 3-year revenue projection",
    "Synthesize findings from 10 conflicting academic papers on a complex topic (e.g. Does creatine affect cognitive function? What is the best diet for longevity?) — identify consensus and genuine disagreements",
    "Write a technical due diligence report for acquiring a 50-person software company — cover tech debt, architecture decisions, IP ownership, key person risk, and integration challenges",
    "Design a multi-cloud architecture for a healthcare startup that must comply with HIPAA, SOC2, and GDPR — include data residency, encryption, access control, and vendor selection criteria",
    "Compare CUDA, ROCm, and open-source GPU acceleration frameworks for training large models — benchmark methodology, hardware requirements, and community support",
    "Evaluate nuclear fusion approaches: tokamak vs stellarator vs inertial confinement vs magnetized target fusion — technical readiness, timeline to breakeven, and commercial viability",
    "Create a comprehensive security audit checklist for a FinTech app handling transactions up to $1M — cover OWASP, PCI-DSS, fraud detection, and regulatory requirements",
    "Develop a pricing strategy for an enterprise AI coding assistant — include value-based pricing justification, competitor price comparison, and tiered packaging",
    "Analyze the long-term effects of climate policy on agricultural output in the Midwest US through 2050 — crop yield projections, water availability, and adaptation strategies",
    "Analyze the 2026 US presidential election: campaign strategies, polling analysis, key swing states, fundraising totals, and potential policy shifts based on outcomes",
]

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def call_minimax(messages, system="", max_tokens=4096, timeout=180, max_retries=3):
    import httpx
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            client = httpx.Client(timeout=timeout)
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
            if system:
                body["system"] = system
            resp = client.post(f"{BASE_URL}/v1/messages", headers=headers, json=body)
            client.close()
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"]
            elif resp.status_code >= 500:
                time.sleep(base_delay * (2 ** attempt))
                base_delay = min(base_delay * 2, 30)
                continue
            else:
                return f"[API Error {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            time.sleep(base_delay * (2 ** attempt))
            base_delay = min(base_delay * 2, 30)
    return f"[API Error] All retries failed"

def run_single_mode(query):
    system = "You are a research assistant. Provide a comprehensive, well-structured report."
    messages = [{"role": "user", "content": query}]
    return call_minimax(messages, system)

def run_fixed_mode(query):
    orch = SwarmOrchestrator(use_gnn=False, mock_mode=False)
    result = orch.run_research(query, mode="fixed")
    return result.get("output", "[No output]")

def run_gnn_mode(query):
    orch = SwarmOrchestrator(use_gnn=True, mock_mode=False)
    result = orch.run_research(query, mode="gnn")
    return result.get("output", "[No output]")

def score_output(query, output):
    error_flag = output.startswith("[API Error") or "timeout" in output.lower()[:100]
    if error_flag:
        return {"error_flag": True, "is_error": True, "overall": 0.0, "factual": 0.0, "coverage": 0.0,
                "insight": 0.0, "evidence": 0.0, "actionability": 0.0, "structure": 0.0}

    prompt = f"""Score this research output on the query.

QUERY: {query[:200]}

OUTPUT:
{output[:3000]}

SCORING DIMENSIONS (100 pts total):
1. Accuracy & Uncertainty (15pts): Accurate + explicitly flags thin evidence, identifies source contradictions
2. Coverage Breadth & Depth (20pts): All subtopics + edge cases + counterarguments + open questions
3. Insight & Originality (25pts) [HEAVIEST]: Non-obvious relationships synthesized across sources
4. Evidence Chain Quality (15pts): Primary sourcing + explicit evidence→inference→conclusion chain
5. Actionability (15pts): Specific with owner + conditions + verification criteria
6. Structure & Readability (10pts): Executive summary + value-add tables + clear hierarchy

CRITICAL: 5 = BASELINE FLOOR (meets minimum). 8 = EXCEPTIONAL (does something NON-OBVIOUS).

Format your response exactly as:
DIMENSION_SCORES:
Accuracy: [score]/15
Coverage: [score]/20
Insight: [score]/25
Evidence: [score]/15
Actionability: [score]/15
Structure: [score]/10
TOTAL: [sum]/100
OVERALL: [total converted to 10-point scale]"""
    system = "You are an expert research evaluator. Score accurately. 5=baseline, 8=exceptional."
    result_text = call_minimax([{"role": "user", "content": prompt}], system=system, max_tokens=500)

    scores = {"error_flag": False, "is_error": False, "overall": 5.0, "factual": 5.0, "coverage": 5.0,
              "insight": 5.0, "evidence": 5.0, "actionability": 5.0, "structure": 5.0}
    for line in result_text.split('\n'):
        if ':' in line:
            parts = line.split(':')
            dim = parts[0].strip().lower()
            if len(parts) > 1:
                nums = re.findall(r'[\d.]+', parts[1])
                if nums:
                    val = float(nums[0])
                    if 'accuracy' in dim:
                        scores['factual'] = val / 15 * 10
                    elif 'coverage' in dim:
                        scores['coverage'] = val / 20 * 10
                    elif 'insight' in dim:
                        scores['insight'] = val / 25 * 10
                    elif 'evidence' in dim:
                        scores['evidence'] = val / 15 * 10
                    elif 'action' in dim:
                        scores['actionability'] = val / 15 * 10
                    elif 'structure' in dim:
                        scores['structure'] = val / 10 * 10
                    elif 'overall' in dim.lower() and 'score' not in dim.lower():
                        scores['overall'] = val
    if scores['overall'] == 5.0:
        scores['overall'] = (scores['factual']*0.15 + scores['coverage']*0.20 +
                            scores['insight']*0.25 + scores['evidence']*0.15 +
                            scores['actionability']*0.15 + scores['structure']*0.10)
    return scores

def run_query(qnum):
    query = HARD_QUERIES[qnum - 1]
    result = {"query_num": qnum, "query": query, "modes": {}}
    log(f"Q{qnum}: Starting")
    for mode, fn in [("single", run_single_mode), ("fixed", run_fixed_mode), ("gnn", run_gnn_mode)]:
        log(f"Q{qnum}/{mode}: Running...")
        output = fn(query)
        scores = score_output(query, output)
        result["modes"][mode] = {
            "output": output[:2000] if len(output) < 2000 else output[:2000],
            "tokens": len(output.split()),
            "output_length": len(output),
            "scores": scores
        }
        log(f"Q{qnum}/{mode}: Done, score={scores.get('overall', '?')}")
    out_file = f"{LOG_DIR}/checkpoint_p8_q{qnum}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Q{qnum}: Saved to {out_file}")
    return result

if __name__ == "__main__":
    qnum = int(sys.argv[1])
    log(f"Starting P8 query {qnum}/15")
    result = run_query(qnum)
    log(f"P8 query {qnum} COMPLETE")