#!/usr/bin/env python3
"""
P11: Chimera Benchmark with Major Improvements

KEY IMPROVEMENTS over P10:
1. GNN Architecture Fix:
   - 4 perspectives instead of 3 (added "regulatory/legal")
   - Increased timeout to 180s per perspective
   - Improved synthesis prompt requiring specific data points

2. Cascade Mode:
   - 4-stage pipeline: generate → critique → revise → final_review
   - Separate from existing 3-stage (generate → critique → synthesize)

3. Scoring Rubric Improvements:
   - Add "depth" anchor tracking per section
   - Track "quantitative specificity" separately
   - 20-point rubric: 5 depth dims + 15 behavioral anchors

4. Circuit Breaker Error Recovery:
   - If mode fails 2x consecutively, extend timeout and retry
   - Track consecutive failures per mode

5. Behavioral Anchors A-T (20 total):
   A: specific_allocation_percentages
   B: primary_source_citations
   C: contradiction_identification
   D: concrete_examples
   E: specific_tools_versions
   F: uncertainty_acknowledgment
   G: specific_actions_with_conditions
   H: synthesizes_3plus_sources
   I: limitation_boundary_conditions
   J: working_code_pseudocode_formula
   K: specific_companies_institutions
   L: timeline_milestone_dates
   M: compares_alternatives_quantitative
   N: risk_threat_probability
   O: framework_taxonomy_methodology
   P: depth_section_completeness  (NEW - tracks thoroughness)
   Q: depth_cross_referencing    (NEW - links sections)
   R: depth_nuance_recognition    (NEW - distinguishes subtle cases)
   S: quantitative_specificity     (NEW - numbers vs adjectives)
   T: source_diversity            (NEW - multiple source types)
"""
import os, sys, json, re, time
from datetime import datetime
from collections import defaultdict

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/run_query_p11.log"

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.io/anthropic"
MODEL = "minimax-m2.7"

# Circuit breaker state
consecutive_failures = defaultdict(int)
circuit_breaker_timeout = {
    "single": 180,
    "fixed": 240,
    "gnn": 300,
    "cascade": 300
}

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
    'O': 'framework_taxonomy_methodology',
    'P': 'depth_section_completeness',
    'Q': 'depth_cross_referencing',
    'R': 'depth_nuance_recognition',
    'S': 'quantitative_specificity',
    'T': 'source_diversity'
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def call_minimax(messages, system="", max_tokens=4096, timeout=120, max_retries=3, mode_name="default"):
    import httpx
    base_delay = 1.0

    # Circuit breaker: extend timeout if mode failed 2x before
    effective_timeout = circuit_breaker_timeout.get(mode_name, timeout)
    if consecutive_failures[mode_name] >= 2:
        effective_timeout = int(effective_timeout * 1.5)
        log(f"  Circuit breaker active for {mode_name}: timeout={effective_timeout}s")

    for attempt in range(max_retries):
        try:
            client = httpx.Client(timeout=effective_timeout)
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
                consecutive_failures[mode_name] += 1
                time.sleep(base_delay * (2 ** attempt))
                base_delay = min(base_delay * 2, 30)
                continue
            elif resp.status_code >= 500:
                consecutive_failures[mode_name] += 1
                time.sleep(base_delay * (2 ** attempt))
                base_delay = min(base_delay * 2, 30)
                continue
            else:
                consecutive_failures[mode_name] += 1
                return f"[API Error {resp.status_code}] {resp.text[:200]}"
        except Exception as e:
            consecutive_failures[mode_name] += 1
            time.sleep(base_delay * (2 ** attempt))
            base_delay = min(base_delay * 2, 30)

    consecutive_failures[mode_name] += 1
    return f"[API Error] All retries failed after {max_retries} attempts"

def reset_circuit_breaker(mode_name):
    """Call after successful execution."""
    if consecutive_failures[mode_name] > 0:
        log(f"  Circuit breaker reset for {mode_name}")
    consecutive_failures[mode_name] = 0

def run_single_mode(query):
    """Single-pass generation."""
    reset_circuit_breaker("single")
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    return call_minimax(messages, system, max_tokens=4096, timeout=180, mode_name="single")

def run_fixed_mode(query):
    """3-stage: generate → critique → synthesize (same as P10)."""
    reset_circuit_breaker("fixed")
    system = "You are a research assistant. Provide a comprehensive, well-structured report."
    messages = [{"role": "user", "content": query}]
    base_output = call_minimax(messages, system, max_tokens=2048, timeout=180, mode_name="fixed")

    critic_prompt = f"""CRITIQUE the following research report. Identify:
1. Missing specific data points or numbers
2. Unsupported claims or thin evidence
3. Gaps in coverage
4. Actions that lack clear owners/conditions

REPORT:
{base_output[:3000]}

Provide your critique as a numbered list."""
    critic_output = call_minimax([{"role": "user", "content": critic_prompt}],
                                 system="You are a critical research auditor.",
                                 max_tokens=1024, timeout=120, mode_name="fixed")

    synthesis_prompt = f"""ORIGINAL REPORT:
{base_output[:3000]}

CRITIC REVIEW:
{critic_output[:1000]}

Based on the critic's feedback, produce an IMPROVED version of the report that addresses the gaps identified. Keep well-supported sections intact."""
    improved_output = call_minimax([{"role": "user", "content": synthesis_prompt}],
                                   system="You are an expert research synthesizer.",
                                   max_tokens=2048, timeout=180, mode_name="fixed")
    return f"{improved_output}\n\n[Critic Review: {critic_output[:500]}]"

def run_gnn_mode(query):
    """
    GNN Mode with 4 perspectives (P11 improvement):
    - technical: tools, frameworks, implementations
    - business: market, revenue, strategy
    - risk: threats, failure modes, constraints
    - regulatory: legal, compliance, standards (NEW)
    - Extended timeout (180s) to prevent timeouts
    - Improved synthesis prompt requiring specific data
    """
    reset_circuit_breaker("gnn")

    perspectives = [
        ("technical", "You are a technical specialist. Focus on specific tools, frameworks, versions, implementation details, code patterns, and architectural tradeoffs."),
        ("business", "You are a business analyst. Focus on market size, competitors, revenue projections, ROI calculations, and strategic considerations with specific numbers."),
        ("risk", "You are a risk analyst. Focus on threats, failure modes, probability estimates, severity assessments, and uncertainty quantification."),
        ("regulatory", "You are a regulatory specialist. Focus on laws, compliance requirements, standards bodies, certification processes, and legal constraints with specific citations.")
    ]

    outputs = []
    for name, system in perspectives:
        prompt = f"""RESEARCH QUERY: {query}

Provide your {name.upper()} PERSPECTIVE with:
1. Specific data points, numbers, and dates
2. Named tools, companies, or frameworks where relevant
3. Concrete examples or case studies
4. Any constraints or limitations you identify

Be thorough - aim for 400+ words."""
        output = call_minimax([{"role": "user", "content": prompt}],
                             system=system,
                             max_tokens=1024,
                             timeout=180,  # Increased from 90s
                             mode_name="gnn")
        outputs.append(f"[{name.upper()}] {output}")

    combined = "\n\n".join(outputs)

    # Improved synthesis prompt requiring specific data integration
    synthesis_prompt = f"""SYNTHESIZE the following four perspectives into a unified report.

PERSPECTIVES:
{combined[:5000]}

QUERY: {query}

INSTRUCTIONS:
1. Produce a COHERENT, UNIFIED report (not a list of perspectives)
2. For EVERY major claim, cite SPECIFIC data from the perspectives above
3. Include concrete numbers, percentages, dates, and named entities
4. Acknowledge where perspectives disagree and why
5. Structure with clear headers and cross-references between sections
6. End with a summary of key insights and remaining uncertainties

FORMAT: Use markdown with specific data prominently featured."""
    final_output = call_minimax([{"role": "user", "content": synthesis_prompt}],
                                system="You are an expert research synthesizer with a mandate for specificity.",
                                max_tokens=2048,
                                timeout=180,
                                mode_name="gnn")
    return final_output

def run_cascade_mode(query):
    """
    NEW 4-stage cascade pipeline (P11 innovation):
    1. generate: Initial comprehensive report
    2. critique: Identify specific gaps
    3. revise: Address each gap specifically
    4. final_review: Verify all improvements were made

    This differs from fixed (3-stage) by adding explicit gap-filling loop.
    """
    reset_circuit_breaker("cascade")

    # Stage 1: Generate
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    stage1 = call_minimax(messages, system, max_tokens=2048, timeout=180, mode_name="cascade")

    # Stage 2: Critique (more detailed than fixed mode)
    critic_prompt = f"""CRITIQUE the following research report with EXTREME SPECIFICITY.

For each gap, provide:
- The EXACT section/paragraph where the gap exists
- What specific data is missing
- What a well-supported claim would look like

REPORT:
{stage1[:4000]}

FORMAT:
GAP 1: [exact location] | [what's missing] | [what's needed]
GAP 2: ...
VERDICT: [overall quality 1-10 with justification]"""
    stage2 = call_minimax([{"role": "user", "content": critic_prompt}],
                         system="You are a precision research auditor.",
                         max_tokens=1024, timeout=120, mode_name="cascade")

    # Stage 3: Revise (gap-filling focused)
    revise_prompt = f"""REVISION TASK: Address each gap identified in the critique.

ORIGINAL REPORT:
{stage1[:4000]}

CRITIQUE:
{stage2[:2000]}

For each GAP identified, provide specific content that fills it.
Use actual numbers, dates, sources, and concrete examples.
If a claim cannot be supported, remove it and note the limitation."""
    stage3 = call_minimax([{"role": "user", "content": revise_prompt}],
                         system="You are a research specialist focused on evidence quality.",
                         max_tokens=2048, timeout=180, mode_name="cascade")

    # Stage 4: Final Review (verification)
    final_prompt = f"""FINAL REVIEW: Verify that all gaps were addressed.

ORIGINAL REPORT:
{stage1[:2000]}

CRITIQUE (gaps identified):
{stage2[:1500]}

REVISED REPORT:
{stage3[:2000]}

CHECKLIST (for each gap from critique):
[ ] Gap 1 addressed: YES/NO with evidence
[ ] Gap 2 addressed: YES/NO with evidence
...

Return a clean, final version that incorporates all revisions.
Mark any gaps that could not be filled with [UNRESOLVED: reason]."""
    final_output = call_minimax([{"role": "user", "content": final_prompt}],
                                system="You are a final quality reviewer.",
                                max_tokens=2048, timeout=180, mode_name="cascade")

    return f"{final_output}\n\n[Stage 2 Critique: {stage2[:300]}]\n[Stage 3 Revisions: Applied]]"

def detect_error_type(output):
    """Classify error type for explicit tagging."""
    if output.startswith("[API Error"):
        if "529" in output:
            return "timeout"
        return "api_error"
    if "[No output]" in output or len(output) < 50:
        return "no_output"
    return None

def score_output_p11(query, output):
    """
    P11 Scoring: 20 anchors (A-T)
    - A-O: Original 15 behavioral anchors
    - P: depth_section_completeness
    - Q: depth_cross_referencing
    - R: depth_nuance_recognition
    - S: quantitative_specificity
    - T: source_diversity

    Returns detailed breakdown with depth tracking.
    """
    error_type = detect_error_type(output)

    if error_type:
        return {
            "error_flag": True,
            "is_error": True,
            "error_type": error_type,
            "overall": 0.0,
            "behavior_count": 0,
            "depth_score": 0.0,
            "quant_score": 0.0,
            "behaviors": [],
            "anchor_scores": {letter: False for letter in 'ABCDEFGHIJKLMNOPQRST'},
            "raw_judge_response": ""
        }

    scoring_prompt = """SCORING TASK: Evaluate this research output for BEHAVIORAL INDICATORS and DEPTH.

QUERY: {query_short}

OUTPUT:
{output_short}

BEHAVIORAL ANCHORS (A-O) — for each, determine YES or NO:
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

DEPTH ANCHORS (P-T) — rate each 0-1:
[P] Section Completeness: Does each major section have substantive content (not just headers)?
[Q] Cross-Referencing: Are there references between sections (e.g. "as discussed in Section 2")?
[R] Nuance Recognition: Does the output distinguish subtle cases, edge conditions, or exceptions?
[S] Quantitative Specificity: Are numerical claims specific (e.g. "23.4%" not "about 25%")?
[T] Source Diversity: Does the output reference multiple types of sources (academic, industry, government)?

For A-O: respond YES or NO
For P-T: respond with a score from 0.0 to 1.0

Format EXACTLY as (one per line):
BEHAVIOR_A: YES
BEHAVIOR_B: NO
...
BEHAVIOR_O: YES or NO
DEPTH_P: 0.7
DEPTH_Q: 0.3
DEPTH_R: 0.8
DEPTH_S: 0.9
DEPTH_T: 0.4
BEHAVIOR_COUNT: [total number of YES for A-O]
DEPTH_AVG: [average of P-T scores]
SCORE: [final overall score 0-10]""".format(query_short=query[:200], output_short=output[:4500])

    system = "You are an expert research evaluator. Count behaviors precisely. Rate depth dimensions carefully."
    result_text = call_minimax([{"role": "user", "content": scoring_prompt}],
                              system=system,
                              max_tokens=1200, timeout=120)

    # Parse behavioral anchors
    anchor_scores = {letter: False for letter in 'ABCDEFGHIJKLMNO'}
    depth_scores = {'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0, 'T': 0.0}

    behaviors_found = []
    for line in result_text.split('\n'):
        line_upper = line.strip().upper()
        # Behavioral anchors A-O
        for letter in 'ABCDEFGHIJKLMNO':
            if f'BEHAVIOR_{letter}:' in line_upper:
                if 'YES' in line_upper:
                    anchor_scores[letter] = True
                    behaviors_found.append(letter)
        # Depth anchors P-T (look for pattern like "DEPTH_P: 0.7" after colon)
        for letter in 'PQRST':
            if f'DEPTH_{letter}:' in line_upper:
                # Extract number after colon, handle floats like 0.7
                parts = line.split(':')
                if len(parts) >= 2:
                    val_str = parts[1].strip()
                    # Find first valid float
                    match = re.search(r'(\d+\.?\d*)', val_str)
                    if match:
                        depth_scores[letter] = float(match.group(1))

    # Find behavior count and scores
    behavior_count = sum(1 for v in anchor_scores.values() if v)
    depth_avg = sum(depth_scores.values()) / 5.0
    quant_score = depth_scores['S']  # Quantitative specificity

    final_score = None
    for line in result_text.split('\n'):
        line_upper = line.strip().upper()
        if 'SCORE:' in line_upper:
            # Extract first number after SCORE:
            match = re.search(r'SCORE:.*?(\d+\.?\d*)', line)
            if match:
                final_score = float(match.group(1))

    # Fallback scoring: combine behavior count with depth
    if final_score is None:
        # Original mapping
        if behavior_count <= 2:
            behavior_base = 4.0
        elif behavior_count <= 4:
            behavior_base = 5.5
        elif behavior_count <= 6:
            behavior_base = 7.0
        elif behavior_count <= 8:
            behavior_base = 8.5
        else:
            behavior_base = 9.5
        # Add depth bonus (up to 0.5)
        final_score = min(10.0, behavior_base + depth_avg * 0.5)

    return {
        "error_flag": False,
        "is_error": False,
        "error_type": None,
        "overall": final_score,
        "behavior_count": behavior_count,
        "depth_score": depth_avg,
        "quant_score": quant_score,
        "behaviors": behaviors_found,
        "anchor_scores": {**anchor_scores, **depth_scores},
        "anchor_labels": {letter: ANCHOR_LABELS[letter] for letter in 'ABCDEFGHIJKLMNOPQRST'},
        "raw_judge_response": result_text[:500]
    }

def run_query(qnum):
    query = HARD_QUERIES[qnum - 1]
    result = {"query_num": qnum, "query": query, "modes": {}}
    log(f"Q{qnum}: Starting P11 with cascade, 4-perspective GNN, circuit breaker")
    log(f"  Circuit breaker state: {dict(consecutive_failures)}")

    # Run all 4 modes
    mode_results = []
    for mode, fn in [
        ("single", run_single_mode),
        ("fixed", run_fixed_mode),
        ("gnn", run_gnn_mode),
        ("cascade", run_cascade_mode)
    ]:
        log(f"Q{qnum}/{mode}: Running...")
        output = fn(query)
        error_type = detect_error_type(output)
        scores = score_output_p11(query, output)
        result["modes"][mode] = {
            "output": output[:2000] if len(output) > 2000 else output,
            "tokens": len(output.split()),
            "output_length": len(output),
            "error_type": error_type,
            "scores": scores
        }
        log(f"Q{qnum}/{mode}: Done, score={scores.get('overall','?')}, error={error_type or 'none'}, behaviors={scores.get('behavior_count','?')}, depth={scores.get('depth_score','?')}")
        mode_results.append((mode, scores.get('overall', 0), error_type))

    # Save checkpoint
    out_file = f"{LOG_DIR}/checkpoint_p11_q{qnum}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Q{qnum}: Saved to {out_file}")

    # Commit after each query
    import subprocess
    subprocess.run(["git", "add", out_file], cwd="/home/jleechan/projects_other/autowiki")
    commit_msg = f"feat(chimera): P11 Q{qnum} complete — cascade+gnn4p+circuit_breaker"
    subprocess.run(["git", "commit", "-m", commit_msg, "-q"], cwd="/home/jleechan/projects_other/autowiki")
    subprocess.run(["git", "push", "-q"], cwd="/home/jleechan/projects_other/autowiki")
    log(f"Q{qnum}: Committed and pushed")

    return result, mode_results

if __name__ == "__main__":
    qnum = int(sys.argv[1])
    log(f"Starting P11 query {qnum}/15")
    result, mode_results = run_query(qnum)
    log(f"P11 query {qnum} COMPLETE")
    log(f"Results: {mode_results}")
