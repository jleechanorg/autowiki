#!/usr/bin/env python3
"""
P11 Aggregate: Compute statistics for P11 benchmark run.

Produces:
- Mean scores per mode (single, fixed, gnn, cascade)
- Error rates per mode
- Win counts
- Per-query breakdown
- Comparison with P10
"""
import json, os, sys
from datetime import datetime

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"
OUTPUT_FILE = f"{LOG_DIR}/checkpoint_p11_aggregate.json"

def load_checkpoint(qnum):
    path = f"{LOG_DIR}/checkpoint_p11_q{qnum}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def aggregate_p11():
    results = []
    mode_scores = {"single": [], "fixed": [], "gnn": [], "cascade": []}
    mode_errors = {"single": 0, "fixed": 0, "gnn": 0, "cascade": 0}
    mode_depths = {"single": [], "fixed": [], "gnn": [], "cascade": []}
    mode_quants = {"single": [], "fixed": [], "gnn": [], "cascade": []}

    for qnum in range(1, 16):
        data = load_checkpoint(qnum)
        if not data:
            continue

        q_result = {"q": qnum}
        for mode in ["single", "fixed", "gnn", "cascade"]:
            mode_data = data["modes"].get(mode, {})
            scores = mode_data.get("scores", {})
            error = mode_data.get("error_type")

            score = scores.get("overall", 0.0) if not scores.get("is_error") else 0.0
            depth = scores.get("depth_score", 0.0)
            quant = scores.get("quant_score", 0.0)

            q_result[mode] = score
            q_result[f"{mode}_err"] = error

            mode_scores[mode].append(score)
            mode_depths[mode].append(depth)
            mode_quants[mode].append(quant)

            if error:
                mode_errors[mode] += 1

        results.append(q_result)

    # Compute aggregates
    modes = ["single", "fixed", "gnn", "cascade"]
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
                "quant_mean": round(sum(mode_quants[mode]) / len(mode_quants[mode]), 3)
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

    output = {
        "mode": "p11",
        "timestamp": datetime.now().isoformat(),
        "queries_tested": len(results),
        "results": results,
        "summary": summary,
        "win_counts": win_counts,
        "comparisons": comparisons,
        "improvements": [
            "4-perspective GNN (added regulatory)",
            "Circuit breaker error recovery",
            "4-stage cascade mode",
            "20-anchor scoring (A-T with depth)",
            "Extended timeout 180s per perspective"
        ]
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("=" * 60)
    print("P11 BENCHMARK RESULTS")
    print("=" * 60)
    for mode in modes:
        s = summary.get(mode, {})
        print(f"\n{mode.upper()}:")
        print(f"  Mean Score: {s.get('mean', 'N/A')}")
        print(f"  Error Rate: {s.get('error_rate', 'N/A')} ({s.get('error_count', 0)} errors)")
        print(f"  Depth Score: {s.get('depth_mean', 'N/A')}")
        print(f"  Quant Score: {s.get('quant_mean', 'N/A')}")
        print(f"  Win Count: {win_counts.get(mode, 0)}")

    print("\n" + "=" * 60)
    print("MODE COMPARISONS")
    print("=" * 60)
    for comp, result in comparisons.items():
        print(f"{comp}: {result}")

    print(f"\nSaved to: {OUTPUT_FILE}")
    return output

if __name__ == "__main__":
    aggregate_p11()
