"""
AI Judge / LLM-as-a-Judge for Chimera evaluation.
Provides multi-dimensional scoring with explanations.
"""

import re
from typing import Dict, Any
from .agents.base import BaseAgent, AgentConfig

# Rubric weights
RUBRIC_WEIGHTS = {
    "factual_accuracy": 0.30,
    "comprehensiveness": 0.25,
    "clarity": 0.20,
    "usefulness": 0.15,
    "efficiency": 0.10,
}


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
You score outputs on a 1-10 scale across 5 dimensions with brutal honesty.
You always provide:
1. Numerical scores
2. Specific evidence from the text
3. Actionable improvement suggestions
You are calibrated against human expert raters (target: >85% correlation)."""
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return self.config.system_prompt

    def _parse_scores(self, raw: str) -> Dict[str, float]:
        """Parse dimension scores from LLM output."""
        result = {
            "factual_accuracy": 5.0,
            "comprehensiveness": 5.0,
            "clarity": 5.0,
            "usefulness": 5.0,
            "efficiency": 5.0,
        }

        # Patterns to extract scores
        patterns = {
            "factual_accuracy": r"[Ff]actual.*?(\d+(?:\.\d+)?)",
            "comprehensiveness": r"[Cc]omprehensive.*?(\d+(?:\.\d+)?)",
            "clarity": r"[Cc]larity.*?(\d+(?:\.\d+)?)",
            "usefulness": r"[Uu]seful.*?(\d+(?:\.\d+)?)",
            "efficiency": r"[Ee]fficiency.*?(\d+(?:\.\d+)?)",
        }

        for dim, pat in patterns.items():
            match = re.search(pat, raw)
            if match:
                result[dim] = min(10.0, max(1.0, float(match.group(1))))

        return result

    def score(self, output: str, reference: str = None) -> Dict[str, Any]:
        """Score a research output using real LLM judgment."""
        prompt = f"""Score the following research output on a 1-10 scale for each dimension.

Output to evaluate:
{output[:4000]}

Dimensions:
1. Factual Accuracy (30% weight): Grounded, no hallucinations
2. Comprehensiveness & Depth (25% weight): Coverage and depth
3. Clarity & Structure (20% weight): Organization and readability
4. Usefulness & Actionability (15% weight): Practical value
5. Efficiency (10% weight): Concise without sacrificing quality

For each dimension give:
- Score (1-10)
- One-sentence justification with evidence

Finally: Overall Score (weighted average) and Top 3 improvements."""
        result = self._call_llm(prompt)

        # Parse scores from output
        dimensions = self._parse_scores(result)

        # Compute weighted overall
        overall = sum(dimensions[k] * RUBRIC_WEIGHTS[k] for k in RUBRIC_WEIGHTS)

        return {
            "raw_judgment": result,
            "overall_score": round(overall, 2),
            "dimensions": {
                "factual_accuracy": round(dimensions["factual_accuracy"], 2),
                "comprehensiveness": round(dimensions["comprehensiveness"], 2),
                "clarity": round(dimensions["clarity"], 2),
                "usefulness": round(dimensions["usefulness"], 2),
                "efficiency": round(dimensions["efficiency"], 2),
            }
        }

    def compare(self, output_a: str, output_b: str, label_a: str = "A", label_b: str = "B") -> Dict:
        """Pairwise comparison (more reliable than absolute scoring)."""
        prompt = f"""Compare Output {label_a} vs Output {label_b}.

{label_a}:
{output_a[:2500]}

{label_b}:
{output_b[:2500]}

Which is better overall? Give scores 1-10 for each dimension and declare a winner with justification.
Scoring dimensions:
- Factual Accuracy
- Comprehensiveness & Depth
- Clarity & Structure
- Usefulness
- Efficiency"""
        result = self._call_llm(prompt)

        # Parse winner from output
        winner = label_b  # default
        if re.search(rf"{label_a}.*?(?:wins?|better|superior)", result, re.IGNORECASE | re.DOTALL):
            winner = label_a
        elif re.search(rf"{label_b}.*?(?:wins?|better|superior)", result, re.IGNORECASE | re.DOTALL):
            winner = label_b

        return {"comparison": result, "winner": winner}