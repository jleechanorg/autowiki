#!/usr/bin/env python3
"""
P12 Aggregate: Compute statistics for P12 benchmark run.

Produces:
- Mean scores per mode (single, fixed, gnn, cascade, ensemble)
- Error rates per mode
- Win counts
- Per-query breakdown
- Complexity × Domain × Mode analysis
- Comparison with P11
"""
import json, os, sys
from datetime import datetime
from collections import defaultdict

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
OUTPUT_FILE = f"{LOG_DIR}/checkpoint_p12_aggregate.json"

def load_checkpoint(qnum):
    path = f"{LOG_DIR}/checkpoint_p12_q{qnum}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def aggregate_p12():
    results = []
    mode_scores = {"single": [], "fixed": [], "gnn": [], "cascade": [], "ensemble": []}
    mode_errors = {"single": 0, "fixed": 0, "gnn": 0, "cascade": 0, "ensemble": 0}
    mode_depths = {"single": [], "fixed": [], "gnn": [], "cascade": [], "ensemble": []}
    mode_action = {"single": [], "fixed": [], "gnn": [], "cascade": [], "ensemble": []}
    mode_source = {"single": [], "fixed": [], "gnn": [], "cascade": [], "ensemble": []}

    # Complexity × Domain × Mode breakdown
    complexity_analysis = defaultdict(lambda: defaultdict(list))
    domain_analysis = defaultdict(lambda: defaultdict(list))

    for qnum in range(1, 16):
        data = load_checkpoint(qnum)
        if not data:
            continue

        complexity = data.get("complexity", "unknown")
        domain = data.get("domain", "unknown")

        q_result = {"q": qnum, "complexity": complexity, "domain": domain}
        for mode in ["single", "fixed", "gnn", "cascade", "ensemble"]:
            mode_data = data["modes"].get(mode, {})
            scores = mode_data.get("scores", {})
            error = mode_data.get("error_type")

            score = scores.get("overall", 0.0) if not scores.get("is_error") else 0.0
            depth = scores.get("depth_score", 0.0)
            action = scores.get("actionability", 0.0)
            source = scores.get("source_quality", 0.0)
            raw = scores.get("raw_score", 0.0)

            q_result[mode] = score
            q_result[f"{mode}_raw"] = raw
            q_result[f"{mode}_err"] = error

            mode_scores[mode].append(score)
            mode_depths[mode].append(depth)
            mode_action[mode].append(action)
            mode_source[mode].append(source)

            if error:
                mode_errors[mode] += 1

            # Track by complexity and domain
            complexity_analysis[complexity][mode].append(score)
            domain_analysis[domain][mode].append(score)

        results.append(q_result)

    # Compute aggregates
    modes = ["single", "fixed", "gnn", "cascade", "ensemble"]
    summary = {}

    for mode in modes:
        scores = [s for s in mode_scores[mode] if s > 0]
        if scores:
            summary[mode] = {
                "mean": round(sum(scores) / len(scores), 2),
                "count": len(scores),
                "error_count": mode_errors[mode],
                "error_rate": round(mode_errors[mode] / 15, 3),
                "depth_mean": round(sum(mode_depths[mode]) / len(mode_depths[mode]), 3),
                "action_mean": round(sum(mode_action[mode]) / len(mode_action[mode]), 3),
                "source_mean": round(sum(mode_source[mode]) / len(mode_source[mode]), 3)
            }

    # Win counts
    win_counts = {mode: 0 for mode in modes}
    for r in results:
        best_mode = max(modes, key=lambda m: r.get(m, 0))
        win_counts[best_mode] += 1

    # Mode vs mode comparison
    comparisons = {}
    for i, m1 in enumerate(modes):
        for m2 in modes[i+1:]:
            m1_wins = sum(1 for r in results if r.get(m1, 0) > r.get(m2, 0))
            m2_wins = sum(1 for r in results if r.get(m2, 0) > r.get(m1, 0))
            ties = sum(1 for r in results if r.get(m1, 0) == r.get(m2, 0))
            comparisons[f"{m1}_vs_{m2}"] = {
                f"{m1}_wins": m1_wins,
                f"{m2}_wins": m2_wins,
                "ties": ties
            }

    # Complexity analysis
    complexity_summary = {}
    for complexity, mode_scores_dict in complexity_analysis.items():
        complexity_summary[complexity] = {}
        for mode, scores in mode_scores_dict.items():
            if scores:
                complexity_summary[complexity][mode] = round(sum(scores) / len(scores), 2)

    # Domain analysis
    domain_summary = {}
    for domain, mode_scores_dict in domain_analysis.items():
        domain_summary[domain] = {}
        for mode, scores in mode_scores_dict.items():
            if scores:
                domain_summary[domain][mode] = round(sum(scores) / len(scores), 2)

    output = {
        "mode": "p12",
        "timestamp": datetime.now().isoformat(),
        "queries_tested": len(results),
        "results": results,
        "summary": summary,
        "win_counts": win_counts,
        "comparisons": comparisons,
        "complexity_analysis": complexity_summary,
        "domain_analysis": domain_summary,
        "improvements": [
            "5-stage cascade (generate → extreme_critique → revise → integration_check → final_review)",
            "5-perspective GNN (added historical)",
            "Ensemble mode (single+gnn+cascade → arbiter)",
            "22-anchor rubric with actionability + source_quality",
            "Query difficulty classification (complexity + domain)"
        ]
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("=" * 60)
    print("P12 BENCHMARK RESULTS")
    print("=" * 60)
    for mode in modes:
        s = summary.get(mode, {})
        print(f"\n{mode.upper()}:")
        print(f"  Mean Score: {s.get('mean', 'N/A')}")
        print(f"  Error Rate: {s.get('error_rate', 'N/A')} ({s.get('error_count', 0)} errors)")
        print(f"  Depth Score: {s.get('depth_mean', 'N/A')}")
        print(f"  Actionability: {s.get('action_mean', 'N/A')}")
        print(f"  Source Quality: {s.get('source_mean', 'N/A')}")
        print(f"  Win Count: {win_counts.get(mode, 0)}")

    print("\n" + "=" * 60)
    print("MODE COMPARISONS")
    print("=" * 60)
    for comp, result in comparisons.items():
        print(f"{comp}: {result}")

    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60)
    for complexity, scores in complexity_summary.items():
        print(f"\n{complexity.upper()}:")
        for mode, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {score}")

    print("\n" + "=" * 60)
    print("DOMAIN ANALYSIS")
    print("=" * 60)
    for domain, scores in domain_summary.items():
        print(f"\n{domain.upper()}:")
        for mode, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {score}")

    print(f"\nSaved to: {OUTPUT_FILE}")
    return output

if __name__ == "__main__":
    aggregate_p12()