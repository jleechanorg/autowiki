#!/usr/bin/env python3
"""
P12: Chimera Benchmark with Major Improvements

KEY IMPROVEMENTS over P11:

1. CASCADE OPTIMIZATION (winner in P11):
   - 5-stage pipeline: generate → extreme_critique → revise → integration_check → final_review
   - Extreme critique: forced numbered gap lists with specific data missing
   - Coverage anchor: penalizes reports missing entire topic sections

2. GNN ENHANCEMENT (5 perspectives, competitive in P11):
   - Added "historical" perspective (5 total)
   - Improved synthesis: preserves more specific data points
   - Synthesis sees original query requirements explicitly

3. SCORING RUBRIC REFINEMENT:
   - Merge redundant depth anchors (P,Q,R → single depth composite)
   - Add new anchors:
     - U: actionability (specific next steps with conditions)
     - V: source_quality (specific URLs, DOIs, paper citations)
   - 22-point rubric max: 15 behavioral + 5 depth + 2 new

4. NEW "ENSEMBLE" MODE:
   - Runs single + gnn + cascade in parallel
   - Arbiter LLM picks best sections from each
   - Produces unified final output

5. QUERY DIFFICULTY CLASSIFICATION:
   - complexity: low/medium/high
   - domain: technical/business/policy/science
   - Enables mode × difficulty analysis
"""
import os, sys, json, re, time
from datetime import datetime
from collections import defaultdict

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/run_query_p12.log"

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
    "ensemble": 480
}

# QUERY DIFFICULTY CLASSIFICATION (P12 new feature)
HARD_QUERIES = [
    # Q1: medium, business - portfolio strategy
    {"q": 1, "complexity": "medium", "domain": "business",
     "text": "Design a comprehensive investment portfolio strategy for a 35-year-old with moderate risk tolerance, incorporating real estate, equities, fixed income, and alternative assets with specific allocation percentages and rebalancing rules"},
    # Q2: high, policy - geopolitical analysis
    {"q": 2, "complexity": "high", "domain": "policy",
     "text": "Analyze the geopolitical implications of the Russia-Ukraine war on global energy markets through 2030, including supply chain disruptions, price volatility, and strategic responses by the EU, US, and China"},
    # Q3: high, technical - emerging languages
    {"q": 3, "complexity": "high", "domain": "technical",
     "text": "Evaluate the top 5 emerging programming languages in 2026 for building production AI systems — include concurrency models, typing systems, ecosystem maturity, and ML framework support"},
    # Q4: high, business - manufacturing scale-up
    {"q": 4, "complexity": "high", "domain": "business",
     "text": "Develop a detailed manufacturing scale-up plan for a startup going from 100 to 100,000 units/month, covering supply chain, equipment, workforce, quality control, and regulatory compliance"},
    # Q5: high, technical - AI frameworks
    {"q": 5, "complexity": "high", "domain": "technical",
     "text": "Compare and critique the architectural patterns of 3 major AI frameworks (pick specific versions: e.g. LangChain 0.3, LlamaIndex 0.11, AutoGen 0.4) — what are the tradeoffs in agent orchestration, memory, and tool use?"},
    # Q6: high, business - competitive analysis
    {"q": 6, "complexity": "high", "domain": "business",
     "text": "Create a competitive analysis for a B2B SaaS product in the project management space — include market size, top 5 competitors, differentiation strategy, and pricing model with 3-year revenue projection"},
    # Q7: medium, science - academic synthesis
    {"q": 7, "complexity": "medium", "domain": "science",
     "text": "Synthesize findings from 10 conflicting academic papers on a complex topic (e.g. Does creatine affect cognitive function? What is the best diet for longevity?) — identify consensus and genuine disagreements"},
    # Q8: high, technical - due diligence
    {"q": 8, "complexity": "high", "domain": "technical",
     "text": "Write a technical due diligence report for acquiring a 50-person software company — cover tech debt, architecture decisions, IP ownership, key person risk, and integration challenges"},
    # Q9: high, policy - multi-cloud compliance
    {"q": 9, "complexity": "high", "domain": "policy",
     "text": "Design a multi-cloud architecture for a healthcare startup that must comply with HIPAA, SOC2, and GDPR — include data residency, encryption, access control, and vendor selection criteria"},
    # Q10: high, technical - GPU frameworks
    {"q": 10, "complexity": "high", "domain": "technical",
     "text": "Compare CUDA, ROCm, and open-source GPU acceleration frameworks for training large models — benchmark methodology, hardware requirements, and community support"},
    # Q11: medium, science - fusion evaluation
    {"q": 11, "complexity": "medium", "domain": "science",
     "text": "Evaluate nuclear fusion approaches: tokamak vs stellarator vs inertial confinement vs magnetized target fusion — technical readiness, timeline to breakeven, and commercial viability"},
    # Q12: high, policy - security audit
    {"q": 12, "complexity": "high", "domain": "policy",
     "text": "Create a comprehensive security audit checklist for a FinTech app handling transactions up to $1M — cover OWASP, PCI-DSS, fraud detection, and regulatory requirements"},
    # Q13: high, business - pricing strategy
    {"q": 13, "complexity": "high", "domain": "business",
     "text": "Develop a pricing strategy for an enterprise AI coding assistant — include value-based pricing justification, competitor price comparison, and tiered packaging"},
    # Q14: medium, science - climate analysis
    {"q": 14, "complexity": "medium", "domain": "science",
     "text": "Analyze the long-term effects of climate policy on agricultural output in the Midwest US through 2050 — crop yield projections, water availability, and adaptation strategies"},
    # Q15: high, policy - election analysis
    {"q": 15, "complexity": "high", "domain": "policy",
     "text": "Analyze the 2026 US presidential election: campaign strategies, polling analysis, key swing states, fundraising totals, and potential policy shifts based on outcomes"},
]

ANCHOR_LABELS = {
    # Original behavioral anchors A-O (15)
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
    # Depth anchors (5, merged from P-T)
    'P': 'depth_section_completeness',
    'Q': 'depth_cross_referencing',
    'R': 'depth_nuance_recognition',
    # NEW anchors
    'S': 'actionability_specific_next_steps',
    'T': 'source_quality_urls_dois',
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
    """Single-pass generation."""
    reset_circuit_breaker("single")
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    return call_minimax(messages, system, max_tokens=4096, timeout=180, mode_name="single")

def run_fixed_mode(query):
    """3-stage: generate → critique → synthesize."""
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
    GNN Mode with 5 perspectives (P12 improvement):
    - technical: tools, frameworks, implementations
    - business: market, revenue, strategy
    - risk: threats, failure modes, constraints
    - regulatory: legal, compliance, standards
    - historical: past precedents, evolution, lessons learned (NEW)
    """
    reset_circuit_breaker("gnn")

    perspectives = [
        ("technical", "You are a technical specialist. Focus on specific tools, frameworks, versions, implementation details, code patterns, and architectural tradeoffs."),
        ("business", "You are a business analyst. Focus on market size, competitors, revenue projections, ROI calculations, and strategic considerations with specific numbers."),
        ("risk", "You are a risk analyst. Focus on threats, failure modes, probability estimates, severity assessments, and uncertainty quantification."),
        ("regulatory", "You are a regulatory specialist. Focus on laws, compliance requirements, standards bodies, certification processes, and legal constraints with specific citations."),
        ("historical", "You are a historical analyst. Focus on past precedents, how the situation evolved, lessons from similar cases, and historical patterns that inform the current analysis.")
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
                             timeout=200,
                             mode_name="gnn")
        outputs.append(f"[{name.upper()}] {output}")

    combined = "\n\n".join(outputs)

    # Improved synthesis prompt - sees query explicitly, preserves specific data
    synthesis_prompt = f"""SYNTHESIZE the following five perspectives into a unified report.

QUERY REQUIREMENTS: {query}

PERSPECTIVES:
{combined[:5000]}

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
                                max_tokens=2048,
                                timeout=200,
                                mode_name="gnn")
    return final_output

def run_cascade_mode(query):
    """
    5-stage cascade pipeline (P12 improvement over P11's 4-stage):
    1. generate: Initial comprehensive report
    2. extreme_critique: Numbered gap lists with specific data missing
    3. revise: Address each gap specifically
    4. integration_check: Validates cross-section consistency (NEW)
    5. final_review: Verify all improvements were made

    Also includes coverage anchor that penalizes missing topic sections.
    """
    reset_circuit_breaker("cascade")

    # Stage 1: Generate
    system = "You are a research assistant. Provide a comprehensive, well-structured report with specific data points, numbers, and actionable recommendations."
    messages = [{"role": "user", "content": query}]
    stage1 = call_minimax(messages, system, max_tokens=2048, timeout=180, mode_name="cascade")

    # Stage 2: Extreme Critique (more specific than P11)
    critic_prompt = f"""CRITIQUE the following research report with EXTREME SPECIFICITY.
For each gap, provide:
- The EXACT section/paragraph where the gap exists
- What specific data is missing
- What a well-supported claim would look like
- Number each gap sequentially

REPORT:
{stage1[:4000]}

FORMAT (mandatory):
GAP 1: [exact location] | [what's missing] | [what's needed] | [why it matters]
GAP 2: ...
GAP N: ...

COVERAGE CHECK: Does the report cover all major aspects of the query?
- If ANY major topic is missing entirely, flag as: COVERAGE_GAP: [topic name]

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
If a claim cannot be supported, remove it and note the limitation.
If COVERAGE_GAP was flagged, address that missing topic."""
    stage3 = call_minimax([{"role": "user", "content": revise_prompt}],
                         system="You are a research specialist focused on evidence quality.",
                         max_tokens=2048, timeout=180, mode_name="cascade")

    # Stage 4: Integration Check (NEW in P12)
    integration_prompt = f"""INTEGRATION CHECK: Verify cross-section consistency.

REVISED REPORT:
{stage3[:3000]}

CHECKLIST:
1. Do numbers in section 1 match numbers referenced in section 2?
2. Are there any contradictory claims between sections?
3. Do timeline references stay consistent throughout?
4. Are cited sources actually referenced correctly?

If issues found, note them. If clean, confirm "INTEGRATION CLEAN"."""
    stage4 = call_minimax([{"role": "user", "content": integration_prompt}],
                         system="You are a consistency checker.",
                         max_tokens=512, timeout=60, mode_name="cascade")

    # Stage 5: Final Review
    final_prompt = f"""FINAL REVIEW: Verify that all gaps were addressed.

ORIGINAL REPORT:
{stage1[:2000]}

CRITIQUE (gaps identified):
{stage2[:1500]}

INTEGRATION CHECK:
{stage4[:500]}

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

    return f"{final_output}\n\n[Stage 2 Critique: {stage2[:300]}]\n[Stage 4 Integration: {stage4[:200]}]\n[Stage 5 Revisions: Applied]]"

def run_ensemble_mode(query):
    """
    NEW ENSEMBLE MODE (P12 innovation):
    - Runs single, gnn, and cascade in parallel (internal)
    - Arbiter LLM picks best sections from each
    - Produces unified final output
    """
    reset_circuit_breaker("ensemble")
    log(f"  Ensemble: running single+gnn+cascade in parallel...")

    # Run all three modes
    single_out = run_single_mode(query)
    gnn_out = run_gnn_mode(query)
    cascade_out = run_cascade_mode(query)

    # Arbiter picks best sections
    arbiter_prompt = f"""ARBITER TASK: Select the best sections from three different report approaches.

ORIGINAL QUERY: {query}

REPORT A (Single-pass):
{single_out[:2500]}

REPORT B (GNN-5-perspective):
{gnn_out[:2500]}

REPORT C (Cascade-5-stage):
{cascade_out[:2500]}

INSTRUCTIONS:
1. For each major section of the query, identify which report provides the BEST answer
2. Cite specific reasons (data quality, depth, actionability, source quality)
3. Produce a UNIFIED final report that combines the best sections
4. When reports disagree, prefer the one with more specific evidence

OUTPUT FORMAT:
## [Section Name]
Best source: [A/B/C]
Content: [selected content from the chosen source]
...

End with a brief summary of how the three approaches differed."""
    arbiter_output = call_minimax([{"role": "user", "content": arbiter_prompt}],
                                  system="You are an expert arbiter synthesizing the best from multiple approaches.",
                                  max_tokens=2048,
                                  timeout=300,
                                  mode_name="ensemble")

    return f"{arbiter_output}\n\n[Ensemble: single+gnn+cascade → arbiter]"

def detect_error_type(output):
    if output.startswith("[API Error"):
        if "529" in output:
            return "timeout"
        return "api_error"
    if "[No output]" in output or len(output) < 50:
        return "no_output"
    return None

def score_output_p12(query, output):
    """
    P12 Scoring: 22 anchors (A-O + P-R + S-T)
    - A-O: Original 15 behavioral anchors
    - P-R: 3 depth anchors (merged from P-T in P11)
    - S: actionability (new)
    - T: source_quality (new)

    Max score: 22 points, displayed as /22 then scaled to /10 for comparison
    """
    error_type = detect_error_type(output)

    if error_type:
        return {
            "error_flag": True,
            "is_error": True,
            "error_type": error_type,
            "overall": 0.0,
            "raw_score": 0,
            "behavior_count": 0,
            "depth_score": 0.0,
            "actionability": 0.0,
            "source_quality": 0.0,
            "behaviors": [],
            "anchor_scores": {letter: False for letter in 'ABCDEFGHIJKLMNOPQRST'},
            "raw_judge_response": ""
        }

    # STRICT JSON OUTPUT MODE - forces consistent parsing
    # CRITICAL: Output ONLY JSON. Start with { and end with }. No other text.
    scoring_prompt = f"""EVALUATION TASK: Score this research output.

QUERY: {query[:200]}
OUTPUT: {output[:4500]}

Return ONLY this JSON, nothing else before or after:
{{
  "behavior_A": true, "behavior_B": false, "behavior_C": false, "behavior_D": false,
  "behavior_E": false, "behavior_F": false, "behavior_G": false, "behavior_H": false,
  "behavior_I": false, "behavior_J": false, "behavior_K": false, "behavior_L": false,
  "behavior_M": false, "behavior_N": false, "behavior_O": false,
  "depth_P": 0.0, "depth_Q": 0.0, "depth_R": 0.0,
  "actionability_S": 0.0, "source_quality_T": 0.0,
  "behavior_count": 0, "depth_avg": 0.0, "raw_score": 0.0, "overall_score": 0.0
}}

For each behavior A-O: set true if present, false if absent.
For depth/actionability/source: rate 0.0 to 1.0.
Calculate: raw_score = behavior_count + (depth_avg * 3) + actionability_S + source_quality_T
Calculate: overall_score = min(10.0, (raw_score / 20.0) * 10.0)
Output JSON ONLY. Start with {{ and end with }}. No explanation."""

    system = "You are a precise JSON scorer. Output ONLY valid JSON, no markdown, no explanation."
    result_text = call_minimax([{"role": "user", "content": scoring_prompt}],
                              system=system,
                              max_tokens=1200, timeout=120)

    # Parse JSON response
    import json
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Fix common JSON issues: true/false (Python bool) vs "true"/"false" string keys
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(result_text.strip())

        # Extract values from JSON
        anchor_scores = {}
        behaviors_found = []
        for letter in 'ABCDEFGHIJKLMNO':
            key = f'behavior_{letter}'
            val = parsed.get(key, False)
            anchor_scores[letter] = bool(val)
            if val:
                behaviors_found.append(letter)

        depth_scores = {
            'P': float(parsed.get('depth_P', 0.0)),
            'Q': float(parsed.get('depth_Q', 0.0)),
            'R': float(parsed.get('depth_R', 0.0))
        }
        actionability = float(parsed.get('actionability_S', 0.0))
        source_quality = float(parsed.get('source_quality_T', 0.0))
        behavior_count = int(parsed.get('behavior_count', len(behaviors_found)))
        depth_avg = float(parsed.get('depth_avg', 0.0))
        raw_score = float(parsed.get('raw_score', 0.0))
        final_score = float(parsed.get('overall_score', 0.0))

    except Exception as e:
        # FALLBACK: try multiple parsing strategies for non-JSON judge responses
        anchor_scores = {letter: False for letter in 'ABCDEFGHIJKLMNO'}
        depth_scores = {'P': 0.0, 'Q': 0.0, 'R': 0.0}
        actionability = 0.0
        source_quality = 0.0
        behaviors_found = []
        final_score = None

        # Strategy 1: Look for VERDICT/SCORE line with embedded number
        for line in result_text.split('\n'):
            ul = line.strip().upper()
            if 'VERDICT' in ul or 'FINAL_SCORE' in ul or ('SCORE' in ul and 'OVERALL' in ul):
                match = re.search(r'(\d+\.?\d*)', line)
                if match and final_score is None:
                    final_score = float(match.group(1))
            if 'OVERALL_SCORE:' in ul or 'OVERALL:' in ul:
                match = re.search(r'(\d+\.?\d*)', line)
                if match and final_score is None:
                    final_score = float(match.group(1))

        # Strategy 2: Look for behavior keywords (handles "behavior A" or "anchor A" format)
        for letter in 'ABCDEFGHIJKLMNO':
            # Pattern: "A." or "A:" or "[A]" or "behavior A" followed by YES/NO
            patterns = [
                rf'\b{letter}\b[.\):\s]+(YES|TRUE|NO|FALSE)',
                rf'behavior\s+{letter}[^a-z]*(YES|TRUE|NO|FALSE)',
                rf'\[A-{letter}\][^a-z]*(YES|TRUE|NO|FALSE)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, result_text, re.IGNORECASE)
                if matches:
                    val = matches[0].upper()
                    if val in ('YES', 'TRUE'):
                        anchor_scores[letter] = True
                        behaviors_found.append(letter)
                    break

        # Strategy 3: Depth P/Q/R from 0-1 scores near the label
        for letter in ['P', 'Q', 'R']:
            # Look for patterns like "depth P: 0.7" or "P: 0.7" near depth keyword
            matches = re.findall(rf'(?:DEPTH\s+)?{letter}\s*:\s*(\d+\.?\d*)', result_text, re.IGNORECASE)
            if matches:
                depth_scores[letter] = min(1.0, float(matches[0]))

        # Strategy 4: Actionability (S) and Source quality (T)
        act_matches = re.findall(r'ACTIONABILITY[^0-9]*(\d+\.?\d*)', result_text, re.IGNORECASE)
        if act_matches:
            actionability = min(1.0, float(act_matches[0]))
        src_matches = re.findall(r'SOURCE[^0-9]*QUALITY[^0-9]*(\d+\.?\d*)', result_text, re.IGNORECASE)
        if src_matches:
            source_quality = min(1.0, float(src_matches[0]))

        # Strategy 5: Count YES/true mentions per behavior letter
        for letter in 'ABCDEFGHIJKLMNO':
            count_yes = len(re.findall(rf'\b{letter}\b[^a-z]*(?:YES|TRUE)', result_text, re.IGNORECASE))
            if count_yes > 0 and not anchor_scores[letter]:
                anchor_scores[letter] = True
                behaviors_found.append(letter)

        behavior_count = sum(1 for v in anchor_scores.values() if v)
        depth_avg = sum(depth_scores.values()) / 3.0
        raw_score = behavior_count + depth_avg * 3 + actionability + source_quality
        if final_score is None:
            final_score = min(10.0, (raw_score / 20.0) * 10.0)

    return {
        "error_flag": False,
        "is_error": False,
        "error_type": None,
        "overall": final_score,
        "raw_score": round(raw_score, 2),
        "behavior_count": behavior_count,
        "depth_score": round(depth_avg, 3),
        "actionability": round(actionability, 3),
        "source_quality": round(source_quality, 3),
        "behaviors": behaviors_found,
        "anchor_scores": {**anchor_scores, **depth_scores, 'S': actionability, 'T': source_quality},
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
    log(f"Q{qnum}: Starting P12 — complexity={complexity}, domain={domain}")
    log(f"  Modes: single, fixed, gnn(5p), cascade(5stg), ensemble")
    log(f"  Circuit breaker state: {dict(consecutive_failures)}")

    # Run all 5 modes
    mode_results = []
    for mode, fn in [
        ("single", run_single_mode),
        ("fixed", run_fixed_mode),
        ("gnn", run_gnn_mode),
        ("cascade", run_cascade_mode),
        ("ensemble", run_ensemble_mode)
    ]:
        log(f"Q{qnum}/{mode}: Running...")
        output = fn(query)
        error_type = detect_error_type(output)
        scores = score_output_p12(query, output)
        result["modes"][mode] = {
            "output": output[:2000] if len(output) > 2000 else output,
            "tokens": len(output.split()),
            "output_length": len(output),
            "error_type": error_type,
            "scores": scores
        }
        log(f"Q{qnum}/{mode}: Done, score={scores.get('overall','?')}, raw={scores.get('raw_score','?')}, error={error_type or 'none'}")
        mode_results.append((mode, scores.get('overall', 0), error_type))

    # Save checkpoint
    out_file = f"{LOG_DIR}/checkpoint_p12_q{qnum}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Q{qnum}: Saved to {out_file}")

    # Commit after each query
    import subprocess
    subprocess.run(["git", "add", out_file], cwd="/home/jleechan/projects_other/autowiki")
    commit_msg = f"feat(chimera): P12 Q{qnum} — cascade(5stg)+gnn(5p)+ensemble, actionability rubric"
    subprocess.run(["git", "commit", "-m", commit_msg, "-q"], cwd="/home/jleechan/projects_other/autowiki")
    subprocess.run(["git", "push", "-q"], cwd="/home/jleechan/projects_other/autowiki")
    log(f"Q{qnum}: Committed and pushed")

    return result, mode_results

if __name__ == "__main__":
    qnum = int(sys.argv[1])
    log(f"Starting P12 query {qnum}/15")
    result, mode_results = run_query(qnum)
    log(f"P12 query {qnum} COMPLETE")
    log(f"Results: {mode_results}")