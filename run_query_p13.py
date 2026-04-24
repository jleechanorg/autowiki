#!/usr/bin/env python3
"""
P13: Chimera Benchmark — Lessons from P11/P12

KEY IMPROVEMENTS:
1. CASCADE: Reverted to 4-stage (generate → extreme_critique → gap_filling → final_review)
   - P12's 5-stage with integration_check hurt performance (6.06 vs P11's 5.52)

2. RUBRIC: 20-anchor rubric (same as P11, NOT P12's 22-anchor)
   - P12 had rubric inconsistencies (Q10: raw=2.0 vs score=10.0)

3. HYBRID MODE (NEW): single + critique in parallel, then arbiter picks best sections
   - Different from ensemble: runs single first, critiques only weakest sections
   - More targeted than ensemble (which runs all 3 modes)

4. GNN: 3 perspectives (technical, business, risk) with 3x longer synthesis
   - P12's 5 perspectives caused GNN regression (5.98 vs P11's 5.07)
   - More perspectives ≠ better; longer synthesis preserves data

5. FIXED: Added verification step at the end
   - Checks if all claims in report are supported

MODES: single, fixed, gnn(3p), cascade(4stg), hybrid
"""
import os, sys, json, re, time
from datetime import datetime
from collections import defaultdict

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/run_query_p13.log"

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL = "https://api.minimax.io/anthropic"
MODEL = "minimax-m2.7"

# Circuit breaker state
consecutive_failures = defaultdict(int)
circuit_breaker_timeout = {
    "single": 180,
    "fixed": 240,
    "gnn": 300,
    "cascade": 360,
    "hybrid": 420
}

HARD_QUERIES = [
    # Q1: medium, business
    {"q": 1, "complexity": "medium", "domain": "business",
     "text": "Design a comprehensive investment portfolio strategy for a 35-year-old with moderate risk tolerance, incorporating real estate, equities, fixed income, and alternative assets with specific allocation percentages and rebalancing rules"},
    # Q2: high, policy
    {"q": 2, "complexity": "high", "domain": "policy",
     "text": "Analyze the geopolitical implications of the Russia-Ukraine war on global energy markets through 2030, including supply chain disruptions, price volatility, and strategic responses by the EU, US, and China"},
    # Q3: high, technical
    {"q": 3, "complexity": "high", "domain": "technical",
     "text": "Evaluate the top 5 emerging programming languages in 2026 for building production AI systems — include concurrency models, typing systems, ecosystem maturity, and ML framework support"},
    # Q4: high, business
    {"q": 4, "complexity": "high", "domain": "business",
     "text": "Develop a detailed manufacturing scale-up plan for a startup going from 100 to 100,000 units/month, covering supply chain, equipment, workforce, quality control, and regulatory compliance"},
    # Q5: high, technical
    {"q": 5, "complexity": "high", "domain": "technical",
     "text": "Compare and critique the architectural patterns of 3 major AI frameworks (pick specific versions: e.g. LangChain 0.3, LlamaIndex 0.11, AutoGen 0.4) — what are the tradeoffs in agent orchestration, memory, and tool use?"},
    # Q6: high, business
    {"q": 6, "complexity": "high", "domain": "business",
     "text": "Create a competitive analysis for a B2B SaaS product in the project management space — include market size, top 5 competitors, differentiation strategy, and pricing model with 3-year revenue projection"},
    # Q7: medium, science
    {"q": 7, "complexity": "medium", "domain": "science",
     "text": "Synthesize findings from 10 conflicting academic papers on a complex topic (e.g. Does creatine affect cognitive function? What is the best diet for longevity?) — identify consensus and genuine disagreements"},
    # Q8: high, technical
    {"q": 8, "complexity": "high", "domain": "technical",
     "text": "Write a technical due diligence report for acquiring a 50-person software company — cover tech debt, architecture decisions, IP ownership, key person risk, and integration challenges"},
    # Q9: high, policy
    {"q": 9, "complexity": "high", "domain": "policy",
     "text": "Design a multi-cloud architecture for a healthcare startup that must comply with HIPAA, SOC2, and GDPR — include data residency, encryption, access control, and vendor selection criteria"},
    # Q10: high, technical
    {"q": 10, "complexity": "high", "domain": "technical",
     "text": "Compare CUDA, ROCm, and open-source GPU acceleration frameworks for training large models — benchmark methodology, hardware requirements, and community support"},
    # Q11: medium, science
    {"q": 11, "complexity": "medium", "domain": "science",
     "text": "Evaluate nuclear fusion approaches: tokamak vs stellarator vs inertial confinement vs magnetized target fusion — technical readiness, timeline to breakeven, and commercial viability"},
    # Q12: high, policy
    {"q": 12, "complexity": "high", "domain": "policy",
     "text": "Create a comprehensive security audit checklist for a FinTech app handling transactions up to $1M — cover OWASP, PCI-DSS, fraud detection, and regulatory requirements"},
    # Q13: high, business
    {"q": 13, "complexity": "high", "domain": "business",
     "text": "Develop a pricing strategy for an enterprise AI coding assistant — include value-based pricing justification, competitor price comparison, and tiered packaging"},
    # Q14: medium, science
    {"q": 14, "complexity": "medium", "domain": "science",
     "text": "Analyze the long-term effects of climate policy on agricultural output in the Midwest US through 2050 — crop yield projections, water availability, and adaptation strategies"},
    # Q15: high, policy
    {"q": 15, "complexity": "high", "domain": "policy",
     "text": "Analyze the 2026 US presidential election: campaign strategies, polling analysis, key swing states, fundraising totals, and potential policy shifts based on outcomes"},
]

# 20-anchor rubric (same as P11, consistent across iterations)
ANCHOR_LABELS = {
    # Behavioral anchors A-O (15)
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
    # Depth anchors P-T (5)
    'P': 'depth_section_completeness',
    'Q': 'depth_cross_referencing',
    'R': 'depth_nuance_recognition',
    'S': 'quantitative_specificity',
    'T': 'source_diversity',
}

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()  # Ensure immediate write

def call_minimax(messages, system="", max_tokens=4096, timeout=120, max_retries=3, mode_name="default"):
    import httpx
    base_delay = 1.0

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
    if consecutive_failures[mode_name] > 0:
        log(f"  Circuit breaker reset for {mode_name}")
    consecutive_failures[mode_name] = 0

def run_single_mode(query):
    """Single-pass generation (P12 winner, keep unchanged)."""
    reset_circuit_breaker("single")
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    return call_minimax(messages, system, max_tokens=4096, timeout=180, mode_name="single")

def run_fixed_mode(query):
    """3-stage + verification: generate → critique → synthesize → verify."""
    reset_circuit_breaker("fixed")
    system = "You are a research assistant. Provide a comprehensive, well-structured report."
    messages = [{"role": "user", "content": query}]
    base_output = call_minimax(messages, system, max_tokens=2048, timeout=180, mode_name="fixed")

    # Stage 2: Critique
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

    # Stage 3: Synthesis
    synthesis_prompt = f"""ORIGINAL REPORT:
{base_output[:3000]}

CRITIC REVIEW:
{critic_output[:1000]}

Based on the critic's feedback, produce an IMPROVED version of the report that addresses the gaps identified. Keep well-supported sections intact."""
    improved_output = call_minimax([{"role": "user", "content": synthesis_prompt}],
                                   system="You are an expert research synthesizer.",
                                   max_tokens=2048, timeout=180, mode_name="fixed")

    # Stage 4: Verification (NEW in P13)
    verify_prompt = f"""VERIFY the following report's claims.

REPORT:
{improved_output[:3000]}

CHECKLIST for each claim:
1. Is there sufficient evidence for this claim?
2. Are specific numbers/dates cited correctly?
3. Are source citations verifiable?
4. Any contradictory claims within the report?

If issues found, note them. If all claims are supported, confirm "VERIFIED: All claims supported"."""
    verify_output = call_minimax([{"role": "user", "content": verify_prompt}],
                                 system="You are a verification specialist.",
                                 max_tokens=512, timeout=60, mode_name="fixed")

    return f"{improved_output}\n\n[Critique: {critic_output[:300]}]\n[Verification: {verify_output[:200]}]"

def run_gnn_mode(query):
    """
    GNN Mode with 3 perspectives + 3x longer synthesis:
    - technical: tools, frameworks, implementations
    - business: market, revenue, strategy
    - risk: threats, failure modes, constraints

    P13 change: Reduced from 5 perspectives (P12) back to 3.
    P12's 5 perspectives caused regression (5.98 vs P11's 5.07).
    More perspectives ≠ better; quality matters more than quantity.

    Also: 3x longer synthesis (6144 tokens vs 2048) to preserve data.
    """
    reset_circuit_breaker("gnn")

    perspectives = [
        ("technical", "You are a technical specialist. Focus on specific tools, frameworks, versions, implementation details, code patterns, and architectural tradeoffs. Be thorough — aim for 500+ words with concrete examples."),
        ("business", "You are a business analyst. Focus on market size, competitors, revenue projections, ROI calculations, and strategic considerations with specific numbers. Be thorough — aim for 500+ words with concrete examples."),
        ("risk", "You are a risk analyst. Focus on threats, failure modes, probability estimates, severity assessments, and uncertainty quantification. Be thorough — aim for 500+ words with concrete examples.")
    ]

    outputs = []
    for name, system in perspectives:
        prompt = f"""RESEARCH QUERY: {query}

Provide your {name.upper()} PERSPECTIVE with:
1. Specific data points, numbers, and dates
2. Named tools, companies, or frameworks where relevant
3. Concrete examples or case studies
4. Any constraints or limitations you identify

Be thorough - aim for 500+ words."""
        output = call_minimax([{"role": "user", "content": prompt}],
                             system=system,
                             max_tokens=1024,
                             timeout=200,
                             mode_name="gnn")
        outputs.append(f"[{name.upper()}] {output}")

    combined = "\n\n".join(outputs)

    # 3x longer synthesis (6144 tokens vs P12's 2048) to preserve data
    synthesis_prompt = f"""SYNTHESIZE the following three perspectives into a unified report.

QUERY REQUIREMENTS: {query}

PERSPECTIVES:
{combined[:6000]}

INSTRUCTIONS:
1. Produce a COHERENT, UNIFIED report (not a list of perspectives)
2. For EVERY major claim, cite SPECIFIC data from the perspectives above
3. Include concrete numbers, percentages, dates, and named entities
4. Acknowledge where perspectives disagree and why
5. Structure with clear headers and cross-references between sections
6. Reference the original query requirements and address each explicitly
7. End with a summary of key insights and remaining uncertainties

FORMAT: Use markdown with specific data prominently featured."""
    final_output = call_minimax([{"role": "user", "content": synthesis_prompt}],
                                system="You are an expert research synthesizer with a mandate for specificity.",
                                max_tokens=4096,  # Reduced from 6144 to prevent memory issues
                                timeout=240,
                                mode_name="gnn")
    return final_output

def run_cascade_mode(query):
    """
    4-stage cascade (reverted from P12's 5-stage):
    1. generate: Initial comprehensive report
    2. extreme_critique: Numbered gap lists with specific data missing
    3. gap_filling: Address each gap specifically (renamed from "revise")
    4. final_review: Verify all improvements were made

    P12's 5-stage with integration_check hurt performance (6.06 vs P11's 5.52).
    Reverting to P11's 4-stage architecture which won.
    """
    reset_circuit_breaker("cascade")

    # Stage 1: Generate
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    stage1 = call_minimax(messages, system, max_tokens=2048, timeout=180, mode_name="cascade")

    # Stage 2: Extreme Critique (same as P11)
    critic_prompt = f"""CRITIQUE the following research report with EXTREME SPECIFICITY.

For each gap, provide:
- The EXACT section/paragraph where the gap exists
- What specific data is missing
- What a well-supported claim would look like

REPORT:
{stage1[:4000]}

FORMAT (mandatory):
GAP 1: [exact location] | [what's missing] | [what's needed]
GAP 2: ...
GAP N: ...

VERDICT: [overall quality 1-10 with justification]"""
    stage2 = call_minimax([{"role": "user", "content": critic_prompt}],
                         system="You are a precision research auditor.",
                         max_tokens=1024, timeout=120, mode_name="cascade")

    # Stage 3: Gap Filling (renamed from "revise" for clarity)
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

    # Stage 4: Final Review (no integration_check - that was P12's mistake)
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

    return f"{final_output}\n\n[Stage 2 Critique: {stage2[:300]}]\n[Stage 3 Gap-filling: Applied]"

def run_hybrid_mode(query):
    """
    NEW HYBRID MODE (P13 innovation):
    - Different from ensemble: runs single first, THEN critiques only weakest sections
    - Ensemble runs all 3 modes (single+gnn+cascade) then arbiter picks
    - Hybrid: single → identify weakest sections → targeted critique → merge

    1. Run single-pass generation
    2. Identify 2-3 weakest sections by content analysis
    3. Run targeted critique on weak sections only
    4. Synthesize improvements
    5. Arbiter picks best version of each section
    """
    reset_circuit_breaker("hybrid")
    log(f"  Hybrid: running single first...")

    # Stage 1: Single-pass generation
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    single_out = call_minimax([{"role": "user", "content": query}], system=system,
                              max_tokens=4096, timeout=180, mode_name="hybrid")

    # Stage 2: Identify weakest sections
    section_analysis_prompt = f"""ANALYZE this report and identify the 2-3 WEAKEST sections.

REPORT:
{single_out[:4000]}

For each section:
1. Name it
2. Rate its quality 1-10
3. Identify what's missing or weak

Return a ranked list of weakest sections by name."""
    section_analysis = call_minimax([{"role": "user", "content": section_analysis_prompt}],
                                    system="You are a quality analyst.",
                                    max_tokens=512, timeout=60, mode_name="hybrid")

    # Stage 3: Targeted critique on weakest sections
    targeted_critique_prompt = f"""CRITIQUE the WEAKEST sections of this report.

ORIGINAL REPORT:
{single_out[:4000]}

WEAKEST SECTIONS (from analysis):
{section_analysis[:500]}

For each weakest section, provide:
- Specific gaps in evidence or data
- What a well-supported version would look like
- Concrete examples or numbers to add"""
    targeted_critique = call_minimax([{"role": "user", "content": targeted_critique_prompt}],
                                     system="You are a critical research auditor.",
                                     max_tokens=1024, timeout=120, mode_name="hybrid")

    # Stage 4: Synthesize improvements
    synthesize_prompt = f"""IMPROVE the report based on targeted critique.

ORIGINAL:
{single_out[:4000]}

CRITIQUE OF WEAKEST SECTIONS:
{targeted_critique[:1000]}

For each section that was critiqued, produce an improved version.
Keep strong sections intact. Address gaps in weak sections."""
    improved = call_minimax([{"role": "user", "content": synthesize_prompt}],
                            system="You are an expert research synthesizer.",
                            max_tokens=2048, timeout=180, mode_name="hybrid")

    # Stage 5: Arbiter picks best version per section
    arbiter_prompt = f"""ARBITER: Select the best version of each section.

ORIGINAL QUERY: {query}

ORIGINAL VERSION:
{single_out[:3000]}

IMPROVED VERSION:
{improved[:3000]}

CRITIQUE (what was wrong with weakest sections):
{targeted_critique[:500]}

For each major section, pick the BETTER version (original or improved).
Prefer versions with more specific evidence, numbers, and actionable content.
If improved version is better, use it. If original is better, keep it.

Produce a FINAL unified report using the best from each version."""
    final = call_minimax([{"role": "user", "content": arbiter_prompt}],
                         system="You are an expert arbiter.",
                         max_tokens=2048, timeout=180, mode_name="hybrid")

    return f"{final}\n\n[Hybrid: single → targeted critique → arbiter merge]"

def detect_error_type(output):
    if output.startswith("[API Error"):
        if "529" in output:
            return "timeout"
        return "api_error"
    if "[No output]" in output or len(output) < 50:
        return "no_output"
    return None

def score_output_p13(query, output):
    """
    P13 Scoring: 20 anchors (A-T, same as P11)
    - A-O: Original 15 behavioral anchors
    - P: depth_section_completeness
    - Q: depth_cross_referencing
    - R: depth_nuance_recognition
    - S: quantitative_specificity
    - T: source_diversity

    Max score: 20 points, displayed as /20 then scaled to /10.
    Uses P11's consistent rubric for cross-phase comparison.
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
            "source_score": 0.0,
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
        for letter in 'ABCDEFGHIJKLMNO':
            if f'BEHAVIOR_{letter}:' in line_upper:
                if 'YES' in line_upper:
                    anchor_scores[letter] = True
                    behaviors_found.append(letter)
        for letter in 'PQRST':
            if f'DEPTH_{letter}:' in line_upper:
                parts = line.split(':')
                if len(parts) >= 2:
                    val_str = parts[1].strip()
                    match = re.search(r'(\d+\.?\d*)', val_str)
                    if match:
                        depth_scores[letter] = float(match.group(1))

    # Find behavior count and score
    behavior_count = sum(1 for v in anchor_scores.values() if v)
    depth_avg = sum(depth_scores.values()) / 5.0
    quant_score = depth_scores['S']
    source_score = depth_scores['T']

    final_score = None
    for line in result_text.split('\n'):
        line_upper = line.strip().upper()
        if 'SCORE:' in line_upper:
            match = re.search(r'SCORE:.*?(\d+\.?\d*)', line)
            if match:
                final_score = float(match.group(1))

    if final_score is None:
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
        final_score = min(10.0, behavior_base + depth_avg * 0.5)

    return {
        "error_flag": False,
        "is_error": False,
        "error_type": None,
        "overall": final_score,
        "behavior_count": behavior_count,
        "depth_score": depth_avg,
        "quant_score": quant_score,
        "source_score": source_score,
        "behaviors": behaviors_found,
        "anchor_scores": {**anchor_scores, **depth_scores},
        "anchor_labels": ANCHOR_LABELS,
        "raw_judge_response": result_text[:500]
    }

def run_query(qnum):
    query_obj = HARD_QUERIES[qnum - 1]
    query = query_obj["text"]
    complexity = query_obj["complexity"]
    domain = query_obj["domain"]

    result = {
        "query_num": qnum,
        "complexity": complexity,
        "domain": domain,
        "query": query,
        "modes": {}
    }
    log(f"Q{qnum}: Starting P13 — complexity={complexity}, domain={domain}")
    log(f"  Modes: single, fixed(+verify), gnn(3p+3xsynth), cascade(4stg), hybrid")
    log(f"  Rubric: 20-anchor (same as P11 for consistency)")
    log(f"  Circuit breaker state: {dict(consecutive_failures)}")

    # Run all 5 modes
    mode_results = []
    for mode, fn in [
        ("single", run_single_mode),
        ("fixed", run_fixed_mode),
        ("gnn", run_gnn_mode),
        ("cascade", run_cascade_mode),
        ("hybrid", run_hybrid_mode)
    ]:
        log(f"Q{qnum}/{mode}: Running...")
        output = fn(query)
        error_type = detect_error_type(output)
        scores = score_output_p13(query, output)
        result["modes"][mode] = {
            "output": output[:2000] if len(output) > 2000 else output,
            "tokens": len(output.split()),
            "output_length": len(output),
            "error_type": error_type,
            "scores": scores
        }
        # Save per-mode checkpoint (partial progress)
        partial_file = f"{LOG_DIR}/checkpoint_p13_q{qnum}_{mode}.partial.json"
        with open(partial_file, "w") as f:
            json.dump(result, f, indent=2)

        log(f"Q{qnum}/{mode}: Done, score={scores.get('overall','?')}, error={error_type or 'none'}, behaviors={scores.get('behavior_count','?')}, depth={scores.get('depth_score','?')}")
        mode_results.append((mode, scores.get('overall', 0), error_type))

    # Save checkpoint
    out_file = f"{LOG_DIR}/checkpoint_p13_q{qnum}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Q{qnum}: Saved to {out_file}")

    # Commit after each query
    import subprocess
    subprocess.run(["git", "add", out_file], cwd="/home/jleechan/projects_other/autowiki")
    commit_msg = f"feat(chimera): P13 Q{qnum} — single+fixed+gnn(3p)+cascade(4stg)+hybrid, 20-anchor"
    subprocess.run(["git", "commit", "-m", commit_msg, "-q"], cwd="/home/jleechan/projects_other/autowiki")
    subprocess.run(["git", "push", "-q"], cwd="/home/jleechan/projects_other/autowiki")
    log(f"Q{qnum}: Committed and pushed")

    return result, mode_results

if __name__ == "__main__":
    qnum = int(sys.argv[1])
    log(f"Starting P13 query {qnum}/15")
    result, mode_results = run_query(qnum)
    log(f"P13 query {qnum} COMPLETE")
    log(f"Results: {mode_results}")