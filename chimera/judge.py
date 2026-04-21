"""
AI Judge / LLM-as-a-Judge for Chimera evaluation.
Provides multi-dimensional scoring with explanations.
"""

import re
from typing import Dict, Any, Tuple
from .agents.base import BaseAgent, AgentConfig

# Rubric dimensions (100 pts total)
# Key design principle: 5 = BASELINE FLOOR (meets minimum), 8 = EXCEPTIONAL (does something non-obvious)
# Gap between 5→8 is behavioral, not just "better"
RUBRIC_WEIGHTS = {
    "accuracy_and_uncertainty": 15,   # Discriminates dangerous inaccuracies from honest uncertainty
    "coverage_breadth_depth": 20,      # Surface coverage vs edge cases + counterarguments
    "insight_and_originality": 25,     # HEAVIEST — discriminates adequate from excellent
    "evidence_chain_quality": 15,      # Citation quality + inference chain
    "actionability": 15,              # Vague vs specific recommendations
    "structure_and_readability": 10,   # Baseline clarity vs executive summary + tables
}

# Behavioral anchors for each dimension
# Each level specifies WHAT THE OUTPUT SAYS OR DOES, not subjective quality adjectives
DIMENSION_ANCHORS = {
    "accuracy_and_uncertainty": {
        5: "All factual claims are accurate. No fabricated citations. No dangerous inaccuracies.",
        6: "Accurate with implicit uncertainty (e.g., 'the study suggests' without overstating).",
        7: "Accurate + distinguishes between evidence-backed claims and speculation in most cases.",
        8: "Accurate + explicitly flags where evidence is thin, identifies contradictions between sources, distinguishes speculation from findings.",
        9: "8 + proactively qualifies uncertain claims with confidence levels, flags query framing limitations.",
        10: "9 + provides working code examples that execute without errors, or functional implementations."
    },
    "coverage_breadth_depth": {
        5: "Covers all major subtopics from the query. Surface-level treatment of each.",
        6: "All major subtopics covered with some depth on 1-2 key areas.",
        7: "All subtopics + identifies 1-2 edge cases or open questions within coverage.",
        8: "All subtopics + systematically addresses edge cases, counterarguments, and open questions within each area.",
        9: "8 + synthesizes connections across subtopics that aren't obvious from any single source.",
        10: "9 + identifies what the query itself failed to ask (meta-coverage)."
    },
    "insight_and_originality": {
        5: "Logical connections between facts. Appropriate conclusions drawn from presented evidence.",
        6: "Connections are explicit and well-explained.",
        7: "Identifies relationships between sources that are not immediately obvious.",
        8: "Non-obvious relationships synthesized across sources in ways they don't do themselves. Identifies contradictions between sources.",
        9: "8 + produces a novel synthesis or framework not present in any single source.",
        10: "9 + generates testable hypotheses or actionable predictions from the synthesis."
    },
    "evidence_chain_quality": {
        5: "Cites sources to support claims. Evidence is referenced.",
        6: "Sources are cited with enough context to evaluate the claim.",
        7: "Primary sourcing (not just secondary summaries), explicit link from evidence to conclusion.",
        8: "Primary sources + explicit evidence→inference→conclusion chain for each major claim.",
        9: "8 + traces claims back to raw data or primary sources, identifies where sources conflict.",
        10: "9 + reconstructs the original methodology and evaluates whether the evidence actually supports the conclusion."
    },
    "actionability": {
        5: "Provides recommendations or next steps. May be vague or generic.",
        6: "Recommendations are specific but lack owner, conditions, or verification.",
        7: "Specific recommendations with at least owner OR conditions.",
        8: "Specific recommendations with owner + conditions + verification criteria.",
        9: "8 + prioritizes recommendations, provides trade-off analysis.",
        10: "9 + includes a concrete implementation plan or code snippet that could execute."
    },
    "structure_and_readability": {
        5: "Clear baseline with section headers. Readable.",
        6: "Well-organized with clear headings and logical flow.",
        7: "6 + includes a summary or key takeaways section.",
        8: "Executive summary + value-add tables/figures + excellent hierarchy. Conclusion first when appropriate.",
        9: "8 + narrative flows from most to least important, with visual aids for complex comparisons.",
        10: "9 + interactive or self-referential structure (e.g., summary table that answers the query directly)."
    },
}


def _is_error_output(output: str) -> Tuple[bool, str]:
    """Check if output is an API/system error, not a valid response.
    Returns (is_error, error_type).
    Search limited to FIRST 500 CHARS to avoid false positives from valid content
    (e.g., "500" in "S&P 500 Index", "timeout" in "timeout_ms" variable names).
    """
    # Only search the prefix where API error indicators actually appear
    search_region = output[:500]

    error_patterns = [
        (r'\[API Error\]', 'api_error'),
        (r'\btimeout\b', 'timeout'),
        (r'\b529\b', 'service_unavailable'),
        (r'rate limit', 'rate_limit'),
        (r'connection error', 'connection_error'),
        (r'upstream error', 'upstream_error'),
        (r'service unavailable', 'service_unavailable'),
        (r'too many requests', 'rate_limit'),
        (r'\b429\b', 'rate_limit'),
        (r'internal server error', 'server_error'),
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


class AIJudge(BaseAgent):
    """Specialized judge agent for scoring research outputs."""

    def __init__(self, name: str = "AIJudge"):
        config = AgentConfig(
            name=name,
            role="judge",
            model="minimax-m2.7",
            temperature=0.3,  # Low temp for consistent scoring
            max_tokens=2048,
            system_prompt="""You are a world-class AI research evaluator.
You score outputs on a 1-10 scale across 6 dimensions (100 pts total) with BRUTAL honesty.
CRITICAL RULE: 5 = BASELINE FLOOR (meets minimum acceptable). 8 = EXCEPTIONAL (does something NON-OBVIOUS).
If an output only meets the baseline, it scores 5 — NOT higher.

SCORING ANCHORS (per dimension):
Accuracy & Uncertainty (15pts):
  5=All accurate, no fabricated citations. 8=Accurate + explicitly flags thin evidence, identifies source contradictions.
Coverage Breadth & Depth (20pts):
  5=Covers all subtopics, surface level. 8=All subtopics + edge cases, counterarguments, open questions.
Insight & Originality (25pts) [HEAVIEST]:
  5=Logical connections, appropriate conclusions. 8=Non-obvious relationships synthesized across sources.
Evidence Chain Quality (15pts):
  5=Cites sources, supports claims. 8=Primary sourcing + explicit evidence→inference→conclusion chain.
Actionability (15pts):
  5=Vague recommendations. 8=Specific action + owner + conditions + verification criteria.
Structure & Readability (10pts):
  5=Clear baseline, section headers. 8=Executive summary + value-add tables + excellent hierarchy.

ACCURACY GATE: If the output contains DANGEROUS inaccuracies (factual errors that could cause harm if acted upon),
cap the maximum composite score at 3/10 regardless of other dimensions.

You always provide:
1. Numerical scores per dimension with BEHAVIORAL justification (what the text SAYS or DOES that earns that score)
2. Specific evidence from the text for each score
3. Top 3 actionable improvements
You are calibrated against human expert raters (target: >85% correlation)."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_scores(self, raw: str) -> Dict[str, float]:
        """Parse dimension scores from LLM output.
        Defaults to 5.0 (baseline floor) if parsing fails — not a high score."""
        result = {dim: 5.0 for dim in RUBRIC_WEIGHTS}

        # Patterns to extract scores — be specific to avoid false matches
        patterns = {
            "accuracy_and_uncertainty": r"[Aa]ccuracy.*?(\d+(?:\.\d+)?)",
            "coverage_breadth_depth": r"[Cc]overage.*?(\d+(?:\.\d+)?)",
            "insight_and_originality": r"[Ii]nsight.*?(\d+(?:\.\d+)?)",
            "evidence_chain_quality": r"[Ee]vidence.*?(\d+(?:\.\d+)?)",
            "actionability": r"[Aa]ctionability.*?(\d+(?:\.\d+)?)",
            "structure_and_readability": r"[Ss]tructure.*?(\d+(?:\.\d+)?)",
        }

        for dim, pat in patterns.items():
            match = re.search(pat, raw)
            if match:
                result[dim] = min(10.0, max(1.0, float(match.group(1))))

        return result

    def score(self, output: str, reference: str = None) -> Dict[str, Any]:
        """Score a research output using real LLM judgment."""
        # Check for error outputs BEFORE calling LLM
        is_error, error_type = _is_error_output(output)
        if is_error:
            return {
                "raw_judgment": f"[Error detected - {error_type}]",
                "overall_score": 0.0,
                "is_error": True,
                "error_type": error_type,
                "error_flag": True,
                "dimensions": {dim: 0.0 for dim in RUBRIC_WEIGHTS}
            }

        prompt = f"""Score the following research output on a 1-10 scale for each dimension.

Output to evaluate:
{output[:4000]}

SCORING SCALE (100 pts total):
1. Accuracy & Uncertainty (15pts): Is everything factually accurate? Does it honestly flag thin evidence?
2. Coverage Breadth & Depth (20pts): Are all subtopics covered? Edge cases? Counterarguments?
3. Insight & Originality (25pts) [HEAVIEST]: Are connections non-obvious? Does it synthesize across sources?
4. Evidence Chain Quality (15pts): Primary sources? Explicit evidence→conclusion chain?
5. Actionability (15pts): Specific recommendations with owner + conditions + verification?
6. Structure & Readability (10pts): Executive summary? Value-add tables? Clear hierarchy?

CRITICAL: 5 = baseline floor (meets minimum). 8 = exceptional (does something NON-OBVIOUS).
If output only meets the baseline, it scores 5 — NOT higher.

ACCURACY GATE: If the output contains DANGEROUS inaccuracies (factual errors that could cause harm),
cap max composite at 3/10 regardless of other dimensions. Flag this explicitly.

For each dimension give:
- Score (1-10) with BEHAVIORAL justification (what the text SAYS or DOES that earns that score)
- One-sentence evidence from the text

Finally: Overall Score (sum of dimension scores, 0-100 scale) and Top 3 improvements."""
        result = self._call_llm(prompt)

        # Parse scores from output
        dimensions = self._parse_scores(result)

        # Check for dangerous inaccuracies in the raw output → accuracy gate
        dangerous_patterns = [
            r'fails to distinguish', r'fabricated', r'fake', r'completely wrong',
            r'contradicted by', r'cannot be verified', r'no evidence suggests',
        ]
        has_dangerous = any(re.search(p, output, re.IGNORECASE) for p in dangerous_patterns)

        # Compute overall (sum of raw dimension scores, not weighted — 0-100 scale)
        raw_total = sum(dimensions.values())

        if has_dangerous:
            # Accuracy gate: cap at 3/10
            capped_overall = min(raw_total / 100 * 10, 3.0)
            dimensions["accuracy_and_uncertainty"] = min(dimensions["accuracy_and_uncertainty"], 3.0)
        else:
            capped_overall = raw_total / 10  # Convert 0-100 to 0-10 scale for compatibility

        return {
            "raw_judgment": result,
            "overall_score": round(capped_overall, 2),
            "dimensions": {k: round(v, 2) for k, v in dimensions.items()},
            "raw_total": raw_total,  # 0-100 scale before capping
            "accuracy_gate_triggered": has_dangerous,
        }

    def compare(self, output_a: str, output_b: str, label_a: str = "A", label_b: str = "B") -> Dict:
        """Pairwise comparison (more reliable than absolute scoring)."""
        # Check both outputs for errors first
        is_error_a, error_type_a = _is_error_output(output_a)
        is_error_b, error_type_b = _is_error_output(output_b)

        # If both are errors, it's a tie
        if is_error_a and is_error_b:
            return {
                "comparison": "[Both outputs are errors]",
                "winner": "TIE",
                "error_a": True,
                "error_b": True,
                "error_type_a": error_type_a,
                "error_type_b": error_type_b,
            }

        # If one is an error, the valid one wins automatically
        if is_error_a:
            return {
                "comparison": f"[Output {label_a} is an error: {error_type_a}]",
                "winner": label_b,
                "error_a": True,
                "error_b": False,
                "error_type_a": error_type_a,
                "error_type_b": None,
            }
        if is_error_b:
            return {
                "comparison": f"[Output {label_b} is an error: {error_type_b}]",
                "winner": label_a,
                "error_a": False,
                "error_b": True,
                "error_type_a": None,
                "error_type_b": error_type_b,
            }

        prompt = f"""Compare Output {label_a} vs Output {label_b}.

{label_a}:
{output_a[:2500]}

{label_b}:
{output_b[:2500]}

Which is better overall? Score each dimension 1-10 and declare a winner with BEHAVIORAL justification.

Scoring dimensions (100 pts total):
- Accuracy & Uncertainty (15pts): Accurate + flags thin evidence
- Coverage Breadth & Depth (20pts): All subtopics + edge cases + counterarguments
- Insight & Originality (25pts): Non-obvious relationships synthesized across sources
- Evidence Chain Quality (15pts): Primary sourcing + explicit evidence→conclusion chain
- Actionability (15pts): Specific with owner + conditions + verification
- Structure & Readability (10pts): Executive summary + value-add tables

CRITICAL: 5 = baseline floor. 8 = does something NON-OBVIOUS.
If scores are within 0.5 of each other overall, declare a TIE and explain why.

Format your response:
A=(score)/10 vs B=(score)/10
Dimension scores:
- Accuracy: A=(score) B=(score)
- Coverage: A=(score) B=(score)
- Insight: A=(score) B=(score)
- Evidence: A=(score) B=(score)
- Actionability: A=(score) B=(score)
- Structure: A=(score) B=(score)
OVERALL: A=(total) vs B=(total) — Winner: A/B/TIE
JUSTIFICATION: (specific behavioral reasons)"""
        result = self._call_llm(prompt)

        # Parse winner from output
        winner = label_b  # default
        if re.search(rf"{label_a}.*?(?:wins?|better|superior)", result, re.IGNORECASE | re.DOTALL):
            winner = label_a
        elif re.search(rf"{label_b}.*?(?:wins?|better|superior)", result, re.IGNORECASE | re.DOTALL):
            winner = label_b

        return {"comparison": result, "winner": winner}