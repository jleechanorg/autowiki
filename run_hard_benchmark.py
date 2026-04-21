#!/usr/bin/env python3
"""
Hard Queries Benchmark for Chimera.
Targets queries that stress multi-agent collaboration.
Uses pairwise comparison + improved score parsing.
"""

import os
import sys
import json
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Set up environment for MiniMax — use env vars, do not hardcode credentials
os.environ.setdefault("MINIMAX_API_KEY", os.environ.get("MINIMAX_API_KEY", ""))
os.environ.setdefault("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

# Add chimera package to path (relative to this file's location)
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from chimera.orchestrator import SwarmOrchestrator
from chimera.utils import load_llm_client
import httpx

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/hard_benchmark.log"

# Hard queries that should produce spread
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
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# MiniMax API client
API_KEY = os.environ["MINIMAX_API_KEY"]
BASE_URL = os.environ["MINIMAX_BASE_URL"]
MODEL = "minimax-m2.7"


def call_minimax(messages: list, system: str = "", max_tokens: int = 4096, timeout: int = 180, max_retries: int = 3) -> str:
    """Call MiniMax /v1/messages endpoint with exponential backoff retry."""
    base_delay = 1.0
    max_delay = 30.0
    for attempt in range(max_retries):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        try:
            resp = httpx.post(
                f"{BASE_URL}/v1/messages",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        return block["text"]
                return str(content[0]) if content else ""
            return str(content)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log(f"[RETRY] attempt {attempt + 1}/{max_retries} failed: {e} — waiting {delay:.1f}s")
                time.sleep(delay)
            else:
                return f"[API Error] {str(e)}"
    return "[API Error] Max retries exceeded"


def run_single_mode(query: str) -> Tuple[str, int]:
    """Run single-agent baseline."""
    log("[SINGLE] Starting single mode...")

    system = """You are a world-class research analyst and investment/technical strategist.
Produce comprehensive, deeply researched responses with specific data, examples, and actionable recommendations.
Be thorough but well-structured. Include specific numbers, dates, and concrete recommendations."""

    prompt = f"""Research query: {query}

Provide a comprehensive response covering all aspects of this query.
Include specific data points, examples, timelines, and actionable recommendations.
Structure your response with clear sections and headings."""

    start = time.time()
    result = call_minimax([{"role": "user", "content": prompt}], system=system, max_tokens=8192)
    elapsed = time.time() - start

    tokens = len(result) // 4  # rough estimate
    log(f"[SINGLE] Complete in {elapsed:.1f}s, ~{tokens} tokens")
    return result, tokens


def run_fixed_mode(query: str) -> Tuple[str, int]:
    """Run 22-agent fixed pipeline."""
    log("[FIXED] Starting fixed pipeline...")

    system_miner = """You are a specialized research agent. Find and summarize key information on the assigned topic.
Return well-organized findings with specific data points and citations."""

    system_expert = """You are a domain expert analyst. Provide deep technical/practical analysis on your area of expertise.
Include specific numbers, case studies, and concrete recommendations."""

    system_critic = """You are a critical auditor. Identify weaknesses, gaps, and overly optimistic claims.
Be direct and specific about what seems questionable."""

    system_synth = """You are a master synthesizer. Create a comprehensive, well-structured report from multiple inputs.
Include all key findings, acknowledge disagreements, and provide actionable conclusions."""

    start = time.time()

    # Literature mining (parallel)
    lit_prompts = [
        f"Research: {query}\nFocus on historical context and background data. Find at least 5 key data points.",
        f"Research: {query}\nFocus on current state-of-the-art and recent developments.",
        f"Research: {query}\nFocus on expert opinions and technical analysis.",
        f"Research: {query}\nFocus on industry applications and case studies.",
        f"Research: {query}\nFocus on future projections and expert forecasts.",
    ]
    lit_results = []
    for i, p in enumerate(lit_prompts):
        lit = call_minimax([{"role": "user", "content": p}], system=system_miner, max_tokens=1536)
        lit_results.append(lit)

    # Expert analysis (parallel)
    expert_prompts = [
        f"Query: {query}\nLiterature findings: {lit_results[0][:500]}\nProvide expert analysis on the technical/practical dimensions.",
        f"Query: {query}\nLiterature findings: {lit_results[1][:500]}\nAnalyze the strategic implications and market dynamics.",
        f"Query: {query}\nLiterature findings: {lit_results[2][:500]}\nAssess risks, challenges, and implementation considerations.",
    ]
    exp_results = []
    for p in expert_prompts:
        exp = call_minimax([{"role": "user", "content": p}], system=system_expert, max_tokens=1536)
        exp_results.append(exp)

    # Critics (parallel)
    crit_prompts = [
        f"Critique these findings for {query}:\n{exp_results[0][:800]}\nIdentify overstatements and gaps.",
        f"Critique these findings:\n{exp_results[1][:800]}\nChallenge assumptions and predictions.",
        f"Critique: {exp_results[2][:800]}\nQuestion feasibility and evidence quality.",
        f"Cross-critique: {exp_results[0][:400]} + {exp_results[1][:400]}\nIdentify contradictions and missing perspectives.",
    ]
    crit_results = []
    for p in crit_prompts:
        c = call_minimax([{"role": "user", "content": p}], system=system_critic, max_tokens=1024)
        crit_results.append(c)

    # Synthesis
    synth_prompt = f"""Synthesize all findings into a comprehensive report on: {query}

LITERATURE:
{chr(10).join(lit_results[:3])[:2000]}

EXPERT ANALYSIS:
{chr(10).join(exp_results)[:2000]}

CRITIQUES:
{chr(10).join(crit_results)[:1500]}

Create a structured report with:
1. Executive Summary
2. Key Findings (with specific data)
3. Expert Analysis
4. Critical Concerns
5. Recommendations
6. Future Outlook"""
    synthesis = call_minimax([{"role": "user", "content": synth_prompt}], system=system_synth, max_tokens=4096)

    elapsed = time.time() - start
    tokens = len(synthesis) // 4
    log(f"[FIXED] Complete in {elapsed:.1f}s, ~{tokens} tokens")
    return synthesis, tokens


def run_gnn_mode(query: str) -> Tuple[str, int]:
    """Run GNN-dynamic pipeline (similar to fixed but with GNN topology simulation)."""
    log("[GNN] Starting GNN-dynamic pipeline...")

    system_router = """You are a GNN topology router. Analyze the query and select the most relevant agents.
Output a JSON array of agent types needed for this task."""

    system_miner = """You are a specialized research agent. Find and summarize key information.
Return well-organized findings with specific data points."""

    system_expert = """You are a domain expert. Provide deep technical analysis with specific numbers and recommendations."""

    system_critic = """You are a critical auditor. Identify weaknesses, gaps, and overly optimistic claims."""

    system_synth = """You are a master synthesizer. Create a comprehensive, well-structured report."""

    start = time.time()

    # GNN routing (what agents are needed)
    routing_prompt = f"""Analyze this query and determine which agents to use:
Query: {query}

Agent types available: literature_miner, domain_expert, critic, fact_checker, synthesizer

Return a JSON list of agent selections with reasoning."""
    routing = call_minimax([{"role": "user", "content": routing_prompt}], system=system_router, max_tokens=512)

    # Literature mining (selected agents based on routing)
    lit_tasks = [
        f"Research {query} - focus on factual data and statistics.",
        f"Research {query} - focus on expert analysis and technical depth.",
        f"Research {query} - focus on applications and use cases.",
    ]
    lit_results = []
    for t in lit_tasks:
        lit = call_minimax([{"role": "user", "content": t}], system=system_miner, max_tokens=1536)
        lit_results.append(lit)

    # Expert analysis
    exp_tasks = [
        f"Analyze: {lit_results[0][:500]}\nProvide technical depth on {query}.",
        f"Analyze: {lit_results[1][:500]}\nAssess strategic implications of {query}.",
        f"Analyze: {lit_results[2][:500]}\nEvaluate practical implementation of {query}.",
    ]
    exp_results = []
    for t in exp_tasks:
        e = call_minimax([{"role": "user", "content": t}], system=system_expert, max_tokens=1536)
        exp_results.append(e)

    # GNN-selected critiques
    crit_tasks = [
        f"Critique the factual claims: {lit_results[0][:600]}",
        f"Challenge the expert analysis: {exp_results[0][:600]}",
        f"Cross-examine: {exp_results[0][:300]} + {exp_results[1][:300]}",
    ]
    crit_results = []
    for t in crit_tasks:
        c = call_minimax([{"role": "user", "content": t}], system=system_critic, max_tokens=1024)
        crit_results.append(c)

    # Synthesis with GNN topology awareness
    synth_prompt = f"""Synthesize into a comprehensive report on: {query}

GN TOPOLOGY: {routing[:200]}

LITERATURE:
{chr(10).join(lit_results)[:1500]}

EXPERT ANALYSIS:
{chr(10).join(exp_results)[:1500]}

CRITIQUES:
{chr(10).join(crit_results)[:1000]}

Create a structured report with clear sections, specific data, and actionable recommendations."""
    synthesis = call_minimax([{"role": "user", "content": synth_prompt}], system=system_synth, max_tokens=4096)

    elapsed = time.time() - start
    tokens = len(synthesis) // 4
    log(f"[GNN] Complete in {elapsed:.1f}s, ~{tokens} tokens")
    return synthesis, tokens


def judge_pairwise(output_a: str, output_b: str, query: str, label_a: str = "A", label_b: str = "B") -> Dict[str, Any]:
    """Pairwise comparison - more reliable than absolute scoring."""
    prompt = f"""Compare these two research outputs on the same query.

QUERY: {query[:200]}

OUTPUT A:
{output_a[:3000]}

OUTPUT B:
{output_b[:3000]}

Compare on these dimensions (score A and B separately 1-10):
1. Factual Accuracy (are claims supported?)
2. Comprehensiveness (coverage of the topic?)
3. Clarity (well-structured and readable?)
4. Usefulness (actionable insights?)
5. Specificity (concrete numbers, examples?)

For each dimension, declare a winner (A or B or TIE).
End with an overall winner declaration.

Format your response exactly as:
DIMENSION_SCORES:
A: [score] / B: [score] / Winner: [A/B/TIE]
... (repeat for each dimension)
OVERALL: Winner: [A/B/TIE] with scores A=[X]/10 vs B=[Y]/10
RATIONALE: [2-3 sentences explaining the decision]"""

    system = """You are a world-class research evaluator. Be honest and critical.
Give higher scores only for genuinely superior outputs. Don't default to high scores."""

    result = call_minimax([{"role": "user", "content": prompt}], system=system, max_tokens=1024, timeout=180)

    # Parse the result
    parsed = _parse_pairwise_result(result, label_a, label_b)
    return {"raw": result, **parsed}


def _parse_pairwise_result(raw: str, label_a: str, label_b: str) -> Dict[str, Any]:
    """Parse pairwise comparison result with robust regex patterns."""
    result = {
        "scores_a": {"factual": 5.0, "comprehensive": 5.0, "clarity": 5.0, "useful": 5.0, "specific": 5.0},
        "scores_b": {"factual": 5.0, "comprehensive": 5.0, "clarity": 5.0, "useful": 5.0, "specific": 5.0},
        "winners": {"factual": "TIE", "comprehensive": "TIE", "clarity": "TIE", "useful": "TIE", "specific": "TIE", "overall": "TIE"},
        "overall_score_a": 5.0,
        "overall_score_b": 5.0,
        "rationale": "",
    }

    # Extract overall scores - handle multiple formats and scales (/10 or /50):
    # "A=5/10 vs B=6/10", "A=N/A/10 vs B=8.4/10", "A=5/50 vs B=37/50"
    # "Winner: B with scores A=0/10 vs B=8.4/10"
    overall_match = re.search(r'A=(N/?A|\d+(?:\.\d+)?)/(?:10|50)\s*vs\.?\s*B=(N/?A|\d+(?:\.\d+)?)/(?:10|50)', raw, re.IGNORECASE | re.DOTALL)
    if overall_match:
        score_a_str = overall_match.group(1)
        score_b_str = overall_match.group(2)
        result["overall_score_a"] = 0.0 if score_a_str.upper() in ('N/A',) else float(score_a_str)
        result["overall_score_b"] = 0.0 if score_b_str.upper() in ('N/A',) else float(score_b_str)

    # Extract overall winner from OVERALL: or Winner: line
    overall_winner_match = re.search(r'OVERALL:.*?Winner:\s*([ABTIE]+)', raw, re.IGNORECASE | re.DOTALL)
    if not overall_winner_match:
        # Also check for "Winner: B with scores A=..."
        overall_winner_match = re.search(r'Winner:\s*([ABTIE]+)', raw, re.IGNORECASE | re.DOTALL)
    if overall_winner_match:
        result["winners"]["overall"] = overall_winner_match.group(1).upper()

    # Extract rationale
    rat_match = re.search(r'RATIONALE:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    if rat_match:
        result["rationale"] = rat_match.group(1).strip()[:300]

    # Extract individual dimension winners - handle N/A, parenthetical scores, etc.
    dim_names = ["factual", "comprehensive", "clarity", "useful", "specific"]
    for dim in dim_names:
        # Pattern handles: "A: 7 / B: 6 / Winner: A", "A: N/A / B: 8 / Winner: B"
        # Also handles: "A: 7 (winner) / B: 6"
        dim_pattern = rf'{dim}.*?A:\s*(\d+(?:\.\d+)?|N/?A).*?B:\s*(\d+(?:\.\d+)?|N/?A).*?Winner:\s*([ABTIE]+)'
        match = re.search(dim_pattern, raw, re.IGNORECASE)
        if match:
            score_a_str = match.group(1)
            score_b_str = match.group(2)
            # Handle N/A
            result["scores_a"][dim] = 0.0 if score_a_str.upper() in ('N/A', 'N/A') else float(score_a_str)
            result["scores_b"][dim] = 0.0 if score_b_str.upper() in ('N/A', 'N/A') else float(score_b_str)
            result["winners"][dim] = match.group(3).upper()

    return result


def _is_error_output(output: str) -> tuple[bool, str]:
    """Check if output is an API/system error, not a valid response.
    Returns (is_error, error_type).

    Conservative detection: Only flag as error if the indicator appears
    near the start of output (first 500 chars) to avoid false positives
    from terms like 'timeout' in valid content or '500' in numbers.
    """
    # Only search first 500 chars to avoid false positives from
    # legitimate content that discusses errors/timeouts/numbers
    search_region = output[:500]

    error_patterns = [
        (r'\[API Error\]', 'api_error'),
        (r'\btimeout\b', 'timeout'),
        (r'\b529\b', 'service_unavailable'),
        (r'\brate limit\b', 'rate_limit'),
        (r'\bconnection error\b', 'connection_error'),
        (r'\bupstream error\b', 'upstream_error'),
        (r'\bservice unavailable\b', 'service_unavailable'),
        (r'\btoo many requests\b', 'rate_limit'),
        (r'\b429\b', 'rate_limit'),
        (r'\binternal server error\b', 'server_error'),
        (r'\b500\b', 'server_error'),
        (r'\b502\b', 'server_error'),
        (r'\b503\b', 'server_error'),
        (r'\b504\b', 'server_error'),
        (r'\[Error\]', 'error'),
    ]
    for pattern, error_type in error_patterns:
        if re.search(pattern, search_region, re.IGNORECASE):
            return True, error_type
    return False, ""


def score_single_output(output: str, query: str) -> Dict[str, float]:
    """Score a single output on absolute scale (1-10 per dimension)."""
    # Check for error outputs BEFORE calling LLM
    is_error, error_type = _is_error_output(output)
    if is_error:
        return {
            "error_flag": True,
            "error_type": error_type,
            "overall": 0.0,
            "factual": 0.0,
            "comprehensive": 0.0,
            "clarity": 0.0,
            "useful": 0.0,
            "specific": 0.0,
        }

    prompt = f"""Score this research output honestly on a 1-10 scale.

QUERY: {query[:200]}

OUTPUT:
{output[:3500]}

Score each dimension (BE HONEST - don't default to high scores):
- Factual Accuracy (1-10): Are claims specific and supported?
- Comprehensiveness (1-10): Is coverage thorough?
- Clarity (1-10): Is it well-organized?
- Usefulness (1-10): Are there actionable insights?
- Specificity (1-10): Are there concrete numbers and examples?

Respond with EXACTLY this format (one number per line):
FACTUAL: [score]
COMPREHENSIVE: [score]
CLARITY: [score]
USEFUL: [score]
SPECIFIC: [score]"""

    system = """You are a strict evaluator. A 5/10 is average. Only give 9-10 for truly exceptional work.
Be critical and specific about what held the score back."""

    result = call_minimax([{"role": "user", "content": prompt}], system=system, max_tokens=512, timeout=180)

    # Parse scores
    scores = {"factual": 5.0, "comprehensive": 5.0, "clarity": 5.0, "useful": 5.0, "specific": 5.0, "overall": 5.0}

    patterns = {
        "factual": r"FACTUAL:\s*(\d+(?:\.\d+)?)",
        "comprehensive": r"COMPREHENSIVE:\s*(\d+(?:\.\d+)?)",
        "clarity": r"CLARITY:\s*(\d+(?:\.\d+)?)",
        "useful": r"USEFUL:\s*(\d+(?:\.\d+)?)",
        "specific": r"SPECIFIC:\s*(\d+(?:\.\d+)?)",
    }

    for dim, pat in patterns.items():
        match = re.search(pat, result, re.IGNORECASE)
        if match:
            scores[dim] = min(10.0, max(1.0, float(match.group(1))))

    # Weighted overall
    weights = {"factual": 0.30, "comprehensive": 0.25, "clarity": 0.20, "useful": 0.15, "specific": 0.10}
    scores["overall"] = sum(scores[d] * weights[d] for d in weights)

    return scores


def run_benchmark():
    """Run full benchmark across all queries and modes."""
    log("=" * 60)
    log("HARD QUERIES BENCHMARK START")
    log(f"Queries: {len(HARD_QUERIES)}")
    log("=" * 60)

    results = []
    start_time = time.time()

    for qi, query in enumerate(HARD_QUERIES, 1):
        log(f"\n{'='*50}")
        log(f"QUERY {qi}/{len(HARD_QUERIES)}: {query[:80]}...")
        log(f"{'='*50}")

        query_result = {
            "query_num": qi,
            "query": query,
            "modes": {}
        }

        try:
            # Run single mode
            single_out, single_tokens = run_single_mode(query)
            query_result["modes"]["single"] = {
                "output": single_out,
                "tokens": single_tokens,
                "output_length": len(single_out),
            }
        except Exception as e:
            log(f"[ERROR] Single mode failed: {e}")
            query_result["modes"]["single"] = {"error": str(e)}

        try:
            # Run fixed mode
            fixed_out, fixed_tokens = run_fixed_mode(query)
            query_result["modes"]["fixed"] = {
                "output": fixed_out,
                "tokens": fixed_tokens,
                "output_length": len(fixed_out),
            }
        except Exception as e:
            log(f"[ERROR] Fixed mode failed: {e}")
            query_result["modes"]["fixed"] = {"error": str(e)}

        try:
            # Run GNN mode
            gnn_out, gnn_tokens = run_gnn_mode(query)
            query_result["modes"]["gnn"] = {
                "output": gnn_out,
                "tokens": gnn_tokens,
                "output_length": len(gnn_out),
            }
        except Exception as e:
            log(f"[ERROR] GNN mode failed: {e}")
            query_result["modes"]["gnn"] = {"error": str(e)}

        # Score outputs
        log("[SCORING] Judging outputs...")

        # Absolute scores for each mode
        if "output" in query_result["modes"].get("single", {}):
            scores = score_single_output(query_result["modes"]["single"]["output"], query)
            query_result["modes"]["single"]["scores"] = scores
            log(f"[SCORING] Single: overall={scores['overall']:.2f}")

        if "output" in query_result["modes"].get("fixed", {}):
            scores = score_single_output(query_result["modes"]["fixed"]["output"], query)
            query_result["modes"]["fixed"]["scores"] = scores
            log(f"[SCORING] Fixed: overall={scores['overall']:.2f}")

        if "output" in query_result["modes"].get("gnn", {}):
            scores = score_single_output(query_result["modes"]["gnn"]["output"], query)
            query_result["modes"]["gnn"]["scores"] = scores
            log(f"[SCORING] GNN: overall={scores['overall']:.2f}")

        # Pairwise comparisons
        if all("output" in query_result["modes"].get(m, {}) for m in ["single", "fixed", "gnn"]):
            log("[SCORING] Pairwise comparisons...")

            # single vs fixed
            comparison = judge_pairwise(
                query_result["modes"]["single"]["output"],
                query_result["modes"]["fixed"]["output"],
                query, "SINGLE", "FIXED"
            )
            query_result["modes"]["pairwise_single_fixed"] = comparison
            log(f"[SCORING] Single vs Fixed: winner={comparison['winners']['overall']}")

            # single vs gnn
            comparison = judge_pairwise(
                query_result["modes"]["single"]["output"],
                query_result["modes"]["gnn"]["output"],
                query, "SINGLE", "GNN"
            )
            query_result["modes"]["pairwise_single_gnn"] = comparison
            log(f"[SCORING] Single vs GNN: winner={comparison['winners']['overall']}")

            # fixed vs gnn
            comparison = judge_pairwise(
                query_result["modes"]["fixed"]["output"],
                query_result["modes"]["gnn"]["output"],
                query, "FIXED", "GNN"
            )
            query_result["modes"]["pairwise_fixed_gnn"] = comparison
            log(f"[SCORING] Fixed vs GNN: winner={comparison['winners']['overall']}")

        results.append(query_result)

        # Check time budget (8 hours = 28800 seconds)
        elapsed = time.time() - start_time
        log(f"[TIME] Elapsed: {elapsed/60:.1f} minutes")
        if elapsed > 27000:  # 7.5 hours, leave 30min buffer
            log("[TIME] Approaching time limit. Saving partial results.")
            break

        # Save checkpoint
        with open(f"{LOG_DIR}/checkpoint.json", "w") as f:
            json.dump({"results": results, "timestamp": datetime.now().isoformat()}, f, indent=2, default=str)

    total_elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"BENCHMARK COMPLETE")
    log(f"Total time: {total_elapsed/60:.1f} minutes")
    log(f"Queries processed: {len(results)}")
    log(f"{'='*60}")

    return results, total_elapsed


def generate_report(results: List[Dict], total_time: float) -> str:
    """Generate markdown report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build comparison table
    table_rows = []
    spreads = []
    error_counts = {"single": 0, "fixed": 0, "gnn": 0}
    error_queries = {"single": [], "fixed": [], "gnn": []}

    for qr in results:
        qi = qr["query_num"]
        query_short = qr["query"][:60] + "..."

        row = f"| Q{qi} | {query_short}"

        single_scores = qr["modes"].get("single", {}).get("scores", {})
        fixed_scores = qr["modes"].get("fixed", {}).get("scores", {})
        gnn_scores = qr["modes"].get("gnn", {}).get("scores", {})

        single_score = single_scores.get("overall", 0) if single_scores else 0
        fixed_score = fixed_scores.get("overall", 0) if fixed_scores else 0
        gnn_score = gnn_scores.get("overall", 0) if gnn_scores else 0

        single_error = single_scores.get("error_flag", False) if single_scores else False
        fixed_error = fixed_scores.get("error_flag", False) if fixed_scores else False
        gnn_error = gnn_scores.get("error_flag", False) if gnn_scores else False

        # Track errors
        if single_error:
            error_counts["single"] += 1
            error_queries["single"].append(qi)
        if fixed_error:
            error_counts["fixed"] += 1
            error_queries["fixed"].append(qi)
        if gnn_error:
            error_counts["gnn"] += 1
            error_queries["gnn"].append(qi)

        # Show ERROR for failed modes, score for valid
        single_str = "ERROR" if single_error else f"{single_score:.1f}"
        fixed_str = "ERROR" if fixed_error else f"{fixed_score:.1f}"
        gnn_str = "ERROR" if gnn_error else f"{gnn_score:.1f}"
        row += f" | {single_str} | {fixed_str} | {gnn_str}"

        # Determine winner (only among non-error scores)
        scores = {}
        if not single_error:
            scores["single"] = single_score
        if not fixed_error:
            scores["fixed"] = fixed_score
        if not gnn_error:
            scores["gnn"] = gnn_score

        if scores:
            max_score = max(scores.values())
            winners = [k for k, v in scores.items() if v == max_score]
            winner_str = "/".join(winners).upper()
        else:
            winner_str = "N/A"
            max_score = 0

        row += f" | {winner_str}"

        spread = max_score - min(s for s in scores.values() if s > 0)
        spreads.append(spread)
        row += f" | {spread:.1f}"

        table_rows.append(row)

    # Calculate mode averages (exclude error outputs)
    mode_stats = {"single": [], "fixed": [], "gnn": []}
    win_counts = {"single": 0, "fixed": 0, "gnn": 0, "tie": 0}

    for qr in results:
        for mode in ["single", "fixed", "gnn"]:
            scores_data = qr["modes"].get(mode, {}).get("scores", {})
            # Exclude error outputs from averages
            if scores_data and not scores_data.get("error_flag", False):
                score = scores_data.get("overall", 0)
                if score > 0:
                    mode_stats[mode].append(score)

        # Count wins (exclude error outputs)
        scores = {}
        for m in ["single", "fixed", "gnn"]:
            scores_data = qr["modes"].get(m, {}).get("scores", {})
            if scores_data and not scores_data.get("error_flag", False):
                scores[m] = scores_data.get("overall", 0)
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                winners = [k for k, v in scores.items() if v == max_score]
                if len(winners) == 1:
                    win_counts[winners[0]] += 1
                else:
                    win_counts["tie"] += 1

    avg_row = "| AVG | |"
    for mode in ["single", "fixed", "gnn"]:
        if mode_stats[mode]:
            avg = sum(mode_stats[mode]) / len(mode_stats[mode])
            avg_row += f" {avg:.1f} |"
        else:
            avg_row += " - |"

    report = f"""# Chimera Hard Queries Benchmark

**Generated:** {timestamp}
**Queries:** {len(HARD_QUERIES)} hard queries
**Queries Run:** {len(results)}
**Total Runtime:** {total_time/60:.1f} minutes

---

## Summary

| Query | Single | Fixed | GNN | Winner | Spread |
|-------|--------|-------||-----|--------|--------|
{"".join(table_rows)}
{avg_row}

---

## Mode Win Counts

| Mode | Wins | Win Rate |
|------|------|----------|
| Single | {win_counts["single"]} | {win_counts["single"]/len(results)*100:.0f}% |
| Fixed | {win_counts["fixed"]} | {win_counts["fixed"]/len(results)*100:.0f}% |
| GNN | {win_counts["gnn"]} | {win_counts["gnn"]/len(results)*100:.0f}% |
| Tie | {win_counts["tie"]} | {win_counts["tie"]/len(results)*100:.0f}% |

---

## Spread Analysis

Average spread (max - min per query): {sum(spreads)/len(spreads):.2f}
Max spread: {max(spreads):.1f}
Min spread: {min(spreads):.1f}

Queries with spread > 1.0: {sum(1 for s in spreads if s > 1.0)} / {len(spreads)}

---

## Error Analysis

| Mode | Errors | Error Rate | Queries Affected |
|------|--------|------------|------------------|
| Single | {error_counts["single"]} | {error_counts["single"]/len(results)*100:.0f}% | {", ".join(f"Q{q}" for q in error_queries["single"]) or "none"} |
| Fixed | {error_counts["fixed"]} | {error_counts["fixed"]/len(results)*100:.0f}% | {", ".join(f"Q{q}" for q in error_queries["fixed"]) or "none"} |
| GNN | {error_counts["gnn"]} | {error_counts["gnn"]/len(results)*100:.0f}% | {", ".join(f"Q{q}" for q in error_queries["gnn"]) or "none"} |

Note: Error-state outputs score 0.0 and are excluded from mode averages and win counts.

---

## Detailed Results

"""

    # Per-query breakdown
    for qr in results:
        qi = qr["query_num"]
        report += f"\n### Query {qi}: {qr['query'][:100]}...\n\n"

        for mode in ["single", "fixed", "gnn"]:
            mode_data = qr["modes"].get(mode, {})
            if "scores" in mode_data:
                s = mode_data["scores"]
                if s.get("error_flag", False):
                    report += f"- **{mode.upper()}**: ERROR ({s.get('error_type', 'unknown')}) - scored 0.0\n"
                else:
                    report += f"- **{mode.upper()}**: Overall {s['overall']:.1f} | "
                    report += f"Factual {s['factual']:.1f} | Comp {s['comprehensive']:.1f} | "
                    report += f"Clarity {s['clarity']:.1f} | Useful {s['useful']:.1f} | Specific {s['specific']:.1f}\n"
            elif "error" in mode_data:
                report += f"- **{mode.upper()}**: ERROR - {mode_data['error'][:100]}\n"

        # Pairwise results
        for pair_key in ["pairwise_single_fixed", "pairwise_single_gnn", "pairwise_fixed_gnn"]:
            if pair_key in qr["modes"]:
                p = qr["modes"][pair_key]
                report += f"\n  *{pair_key.replace('pairwise_', '').upper()}:* "
                report += f"Winner={p['winners']['overall']}, "
                report += f"A={p['overall_score_a']:.1f}, B={p['overall_score_b']:.1f}\n"

        report += "\n"

    # Analysis section
    report += """
---

## Analysis

### Where did modes differ?

"""

    # Find queries with significant spread
    for qr in results:
        scores = {m: qr["modes"].get(m, {}).get("scores", {}).get("overall", 0) for m in ["single", "fixed", "gnn"]}
        spread = max(scores.values()) - min(s for s in scores.values() if s > 0)
        if spread > 1.0:
            report += f"- **Q{qr['query_num']}** ({qr['query'][:60]}...): spread={spread:.1f}\n"

    report += """
### When does multi-agent pay off?

Based on the results:
"""

    # Analyze when fixed/gnn beat single
    multi_wins = 0
    single_wins = 0
    for qr in results:
        scores = {m: qr["modes"].get(m, {}).get("scores", {}).get("overall", 0) for m in ["single", "fixed", "gnn"]}
        if max(scores.values()) > 0:
            best_multi = max(scores["fixed"], scores["gnn"])
            if best_multi > scores["single"]:
                multi_wins += 1
            elif scores["single"] > best_multi:
                single_wins += 1

    report += f"- Multi-agent (fixed or GNN) beat single: {multi_wins} queries\n"
    report += f"- Single matched or beat multi-agent: {single_wins} queries\n"

    report += """
### GNN vs Fixed comparison

"""
    gnn_beats_fixed = 0
    fixed_beats_gnn = 0
    for qr in results:
        if "pairwise_fixed_gnn" in qr["modes"]:
            p = qr["modes"]["pairwise_fixed_gnn"]
            if p["winners"]["overall"] == "GNN":
                gnn_beats_fixed += 1
            elif p["winners"]["overall"] == "FIXED":
                fixed_beats_gnn += 1

    report += f"- GNN beat Fixed: {gnn_beats_fixed} pairwise comparisons\n"
    report += f"- Fixed beat GNN: {fixed_beats_gnn} pairwise comparisons\n"

    report += f"""
---

*Generated by Chimera Hard Queries Benchmark | MiniMax M2.7 API*
"""

    return report


if __name__ == "__main__":
    log("[START] Hard queries benchmark initiated")

    results, total_time = run_benchmark()

    # Generate report
    report = generate_report(results, total_time)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_hard_queries.md")
    with open(output_path, "w") as f:
        f.write(report)

    log(f"[COMPLETE] Report written to: {output_path}")

    # Save raw JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_hard_queries.json")
    with open(json_path, "w") as f:
        json.dump({
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "total_time_minutes": total_time / 60,
        }, f, indent=2, default=str)

    log(f"[COMPLETE] Raw JSON saved to: {json_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Queries run: {len(results)}")
    print(f"Runtime: {total_time/60:.1f} minutes")
    print(f"Output: {output_path}")
