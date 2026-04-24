#!/usr/bin/env python3
"""
P10: Improved behavioral anchor scoring with:
1. Structured per-anchor YES/NO (A-O) in JSON output
2. Explicit error tagging with error_type
3. Per-anchor breakdown for post-run analysis (especially Q5/Q6 single wins)
"""
import os, sys, json, re, time
from datetime import datetime

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/run_query_p10.log"

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

ANCHOR_LABELS = {
    'A': 'specific_allocation_percentages',
    'B': 'primary_source_citations',
    'C': 'contradiction_identification',
    'D': 'concrete_examples',
    'E': 'specific_tools_versions',
    'F': 'uncertainty_acknowledgment',
    'G': 'specific_actions_with_conditions',
    'H': 'synthesizes_3plus_sources',
    'I': 'limitation_boundary_conditions',
    'J': 'working_code_pseudocode_formula',
    'K': 'specific_companies_institutions',
    'L': 'timeline_milestone_dates',
    'M': 'compares_alternatives_quantitative',
    'N': 'risk_threat_probability',
    'O': 'framework_taxonomy_methodology'
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def call_minimax(messages, system="", max_tokens=4096, timeout=120, max_retries=3):
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
                data = resp.json()
                content = data.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            return block["text"]
                return str(content[0]) if content else ""
            elif resp.status_code == 529:
                time.sleep(base_delay * (2 ** attempt))
                base_delay = min(base_delay * 2, 30)
                continue
            elif resp.status_code >= 500:
                time.sleep(base_delay * (2 ** attempt))
                base_delay = min(base_delay * 2, 30)
                continue
            else:
                return f"[API Error {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            time.sleep(base_delay * (2 ** attempt))
            base_delay = min(base_delay * 2, 30)
    return f"[API Error] All retries failed after {max_retries} attempts"

def run_single_mode(query):
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    return call_minimax(messages, system, max_tokens=4096, timeout=180)

def run_fixed_mode(query):
    system = "You are a research assistant. Provide a comprehensive, well-structured report."
    messages = [{"role": "user", "content": query}]
    base_output = call_minimax(messages, system, max_tokens=2048, timeout=180)
    critic_prompt = f"""CRITIQUE the following research report. Identify:
1. Missing specific data points or numbers
2. Unsupported claims or thin evidence
3. Gaps in coverage
4. Actions that lack clear owners/conditions

REPORT:
{base_output[:3000]}

Provide your critique as a numbered list."""
    critic_output = call_minimax([{"role": "user", "content": critic_prompt}], system="You are a critical research auditor.", max_tokens=1024, timeout=120)
    synthesis_prompt = f"""ORIGINAL REPORT:
{base_output[:3000]}

CRITIC REVIEW:
{critic_output[:1000]}

Based on the critic's feedback, produce an IMPROVED version of the report that addresses the gaps identified. Keep well-supported sections intact."""
    improved_output = call_minimax([{"role": "user", "content": synthesis_prompt}], system="You are an expert research synthesizer.", max_tokens=2048, timeout=180)
    return f"{improved_output}\n\n[Critic Review: {critic_output[:500]}]"

def run_gnn_mode(query):
    perspectives = [
        ("technical", "You are a technical specialist. Focus on specific tools, frameworks, versions, and implementation details."),
        ("business", "You are a business analyst. Focus on market size, competitors, revenue, and strategic considerations."),
        ("risk", "You are a risk analyst. Focus on threats, failure modes, constraints, and uncertainty.")
    ]
    outputs = []
    for name, system in perspectives:
        prompt = f"Research query: {query}\n\nProvide your perspective as a {name} specialist with specific data and analysis."
        output = call_minimax([{"role": "user", "content": prompt}], system=system, max_tokens=1024, timeout=90)
        outputs.append(f"[{name.upper()}] {output}")
    combined = "\n\n".join(outputs)
    synthesis_prompt = f"""SYNTHESIZE the following diverse perspectives into a coherent report.

PERSPECTIVES:
{combined[:4000]}

QUERY: {query}

Produce a unified, coherent report that integrates these perspectives. Ensure specific data points are included."""
    final_output = call_minimax([{"role": "user", "content": synthesis_prompt}], system="You are an expert research synthesizer.", max_tokens=2048, timeout=180)
    return final_output

def detect_error_type(output):
    """Classify error type for explicit tagging."""
    if output.startswith("[API Error"):
        if "529" in output:
            return "timeout"
        return "api_error"
    if "[No output]" in output or len(output) < 50:
        return "no_output"
    return None

def score_output_behavioral_p10(query, output):
    """P10: Score with explicit error tagging and per-anchor breakdown."""
    error_type = detect_error_type(output)

    if error_type:
        return {
            "error_flag": True,
            "is_error": True,
            "error_type": error_type,
            "overall": 0.0,
            "behavior_count": 0,
            "behaviors": [],
            "anchor_scores": {letter: False for letter in 'ABCDEFGHIJKLMNO'},
            "raw_judge_response": ""
        }

    scoring_prompt = """SCORING TASK: Evaluate this research output for specific BEHAVIORAL INDICATORS.

QUERY: {query_short}

OUTPUT:
{output_short}

BEHAVIORAL ANCHORS — for each, determine YES or NO:
[A] Lists specific allocation percentages or numbers (e.g. "60% equities, 20% bonds")
[B] Cites primary sources with URLs, DOIs, or specific document titles
[C] Identifies a contradiction or disagreement between two sources/claims
[D] Provides a concrete example, case study, or counterexample
[E] Names a specific tool, library, framework, or product with version number
[F] Acknowledges uncertainty, flags thin evidence, or states confidence level
[G] Suggests a specific action with conditions, owner, and verification criteria
[H] Synthesizes information from 3 or more sources in a single paragraph
[I] Identifies a limitation, boundary condition, or scope of the analysis
[J] Provides working code, pseudocode, formula, or quantitative model
[K] Names specific companies, people, or institutions with their roles
[L] Provides timeline, schedule, or milestone with dates or phases
[M] Compares alternatives on specific dimensions with quantitative tradeoffs
[N] Flags a risk or threat with probability or severity estimate
[O] Uses a framework, taxonomy, or structured methodology (e.g. SWOT, Porter's Five Forces)

SCORING TABLE:
0-2 behaviors = 4.0 | 3-4 behaviors = 5.5 | 5-6 behaviors = 7.0 | 7-8 behaviors = 8.5 | 9+ behaviors = 9.5

For EACH letter A through O, respond with YES or NO.

Format EXACTLY as (one per line):
BEHAVIOR_A: YES
BEHAVIOR_B: NO
...
BEHAVIOR_O: YES or NO
BEHAVIOR_COUNT: [total number of YES]
SCORE: [final score]""".format(query_short=query[:200], output_short=output[:4000])

    system = "You are an expert research evaluator. Count behaviors precisely."
    result_text = call_minimax([{"role": "user", "content": scoring_prompt}], system=system, max_tokens=1200, timeout=120)

    # Parse per-anchor YES/NO
    anchor_scores = {}
    for letter in 'ABCDEFGHIJKLMNO':
        anchor_scores[letter] = False

    behaviors_found = []
    for line in result_text.split('\n'):
        line_upper = line.strip().upper()
        for letter in 'ABCDEFGHIJKLMNO':
            if f'BEHAVIOR_{letter}:' in line_upper:
                if 'YES' in line_upper:
                    anchor_scores[letter] = True
                    behaviors_found.append(letter)

    # Find behavior count and score
    behavior_count = 0
    final_score = None
    for line in result_text.split('\n'):
        line_upper = line.strip().upper()
        if 'BEHAVIOR_COUNT:' in line_upper:
            nums = re.findall(r'\d+', line)
            if nums:
                behavior_count = int(nums[0])
        if 'SCORE:' in line_upper:
            nums = re.findall(r'[\d.]+', line)
            if nums:
                final_score = float(nums[0])

    if behavior_count == 0:
        behavior_count = sum(1 for v in anchor_scores.values() if v)

    if final_score is None:
        if behavior_count <= 2:
            final_score = 4.0
        elif behavior_count <= 4:
            final_score = 5.5
        elif behavior_count <= 6:
            final_score = 7.0
        elif behavior_count <= 8:
            final_score = 8.5
        else:
            final_score = 9.5

    return {
        "error_flag": False,
        "is_error": False,
        "error_type": None,
        "overall": final_score,
        "behavior_count": behavior_count,
        "behaviors": behaviors_found,
        "anchor_scores": anchor_scores,
        "anchor_labels": {letter: ANCHOR_LABELS[letter] for letter in 'ABCDEFGHIJKLMNO'},
        "raw_judge_response": result_text[:500]
    }

def run_query(qnum):
    query = HARD_QUERIES[qnum - 1]
    result = {"query_num": qnum, "query": query, "modes": {}}
    log(f"Q{qnum}: Starting P10")
    for mode, fn in [("single", run_single_mode), ("fixed", run_fixed_mode), ("gnn", run_gnn_mode)]:
        log(f"Q{qnum}/{mode}: Running...")
        output = fn(query)
        error_type = detect_error_type(output)
        scores = score_output_behavioral_p10(query, output)
        result["modes"][mode] = {
            "output": output[:2000] if len(output) > 2000 else output,
            "tokens": len(output.split()),
            "output_length": len(output),
            "error_type": error_type,
            "scores": scores
        }
        log(f"Q{qnum}/{mode}: Done, score={scores.get('overall','?')}, error={error_type or 'none'}, behaviors={scores.get('behavior_count','?')}")
    out_file = f"{LOG_DIR}/checkpoint_p10_q{qnum}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Q{qnum}: Saved to {out_file}")

    # Commit after each query
    import subprocess
    subprocess.run(["git", "add", out_file], cwd="/home/jleechan/projects_other/autowiki")
    commit_msg = f"feat(chimera): P10 Q{qnum} complete — scores saved"
    subprocess.run(["git", "commit", "-m", commit_msg, "-q"], cwd="/home/jleechan/projects_other/autowiki")
    subprocess.run(["git", "push", "-q"], cwd="/home/jleechan/projects_other/autowiki")
    log(f"Q{qnum}: Committed and pushed")

    return result

if __name__ == "__main__":
    qnum = int(sys.argv[1])
    log(f"Starting P10 query {qnum}/15")
    result = run_query(qnum)
    log(f"P10 query {qnum} COMPLETE")