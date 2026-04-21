#!/usr/bin/env python3
"""Aggregate benchmark results from benchmark_hard_queries.json and compute per-mode statistics."""

import json
import sys
from pathlib import Path
from collections import defaultdict

BENCHMARK_FILE = Path(__file__).parent / "benchmark_hard_queries.json"


def is_error_state(mode_data: dict) -> bool:
    """Check if a mode output is in error state."""
    output = mode_data.get("output", "")
    error_indicators = ["[API Error]", "timeout", "529", "error", "Error"]
    return any(indicator.lower() in output.lower() for indicator in error_indicators)


def compute_statistics(data: dict) -> dict:
    """Compute per-mode statistics from benchmark data."""
    results = data.get("results", [])

    # Track per-mode stats
    modes = ["single", "fixed", "gnn"]
    stats = {mode: {
        "scores": [],
        "wins": 0,
        "error_count": 0,
        "total_count": 0
    } for mode in modes}

    for query in results:
        query_num = query.get("query_num", "?")
        modes_data = query.get("modes", {})

        total_count = len(modes_data)
        if total_count == 0:
            continue

        # Identify error states
        errors = {}
        for mode in modes:
            if mode in modes_data:
                errors[mode] = is_error_state(modes_data[mode])
                stats[mode]["total_count"] += 1
                if errors[mode]:
                    stats[mode]["error_count"] += 1

        # Get overall scores (excluding errors)
        valid_scores = {}
        for mode in modes:
            if mode in modes_data and not errors[mode]:
                score = modes_data[mode].get("scores", {}).get("overall")
                if score is not None:
                    valid_scores[mode] = score
                    stats[mode]["scores"].append(score)

        # Determine winner (among valid, non-error outputs)
        if valid_scores:
            max_score = max(valid_scores.values())
            winners = [m for m, s in valid_scores.items() if s == max_score]
            for winner in winners:
                stats[winner]["wins"] += 1

    return stats


def print_report(stats: dict):
    """Print formatted statistics report."""
    modes = ["single", "fixed", "gnn"]

    print("=" * 60)
    print("BENCHMARK STATISTICS REPORT")
    print("=" * 60)

    for mode in modes:
        s = stats[mode]
        scores = s["scores"]
        n_valid = len(scores)
        n_total = s["total_count"]
        n_errors = s["error_count"]
        error_rate = (n_errors / n_total * 100) if n_total > 0 else 0

        if n_valid > 0:
            avg = sum(scores) / n_valid
            min_s = min(scores)
            max_s = max(scores)
        else:
            avg = 0
            min_s = max_s = 0

        print(f"\n[{mode.upper()}]")
        print(f"  Total queries:     {n_total}")
        print(f"  Error queries:     {n_errors} ({error_rate:.1f}% error rate)")
        print(f"  Valid queries:     {n_valid}")
        print(f"  Average score:     {avg:.2f}" if n_valid > 0 else "  Average score:     N/A (all errors)")
        if n_valid > 0:
            print(f"  Score range:       {min_s:.1f} - {max_s:.1f}")
        print(f"  Win count:         {s['wins']}")

    # Win rate summary
    print("\n" + "-" * 60)
    print("WIN COUNTS:")
    for mode in modes:
        s = stats[mode]
        total_valid_wins = sum(stats[m]["wins"] for m in modes if stats[m]["scores"])
        if s["scores"]:
            win_rate = (s["wins"] / len(s["scores"]) * 100) if s["scores"] else 0
            print(f"  {mode}: {s['wins']} wins ({win_rate:.1f}% of valid queries)")
        else:
            print(f"  {mode}: {s['wins']} wins (no valid queries)")

    # Summary comparison
    print("\n" + "=" * 60)
    print("MODE COMPARISON (averages of valid, non-error outputs only):")
    avgs = []
    for mode in modes:
        scores = stats[mode]["scores"]
        if scores:
            avgs.append((mode, sum(scores) / len(scores)))
    avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (mode, avg) in enumerate(avgs, 1):
        print(f"  {i}. {mode}: {avg:.2f}")


def main():
    if not BENCHMARK_FILE.exists():
        print(f"ERROR: {BENCHMARK_FILE} not found")
        sys.exit(1)

    with open(BENCHMARK_FILE, "r") as f:
        data = json.load(f)

    stats = compute_statistics(data)
    print_report(stats)

    # Return comparison values for verification
    print("\n" + "=" * 60)
    print("VERIFICATION CHECK:")
    single_avg = sum(stats["single"]["scores"]) / len(stats["single"]["scores"]) if stats["single"]["scores"] else 0
    fixed_avg = sum(stats["fixed"]["scores"]) / len(stats["fixed"]["scores"]) if stats["fixed"]["scores"] else 0
    gnn_avg = sum(stats["gnn"]["scores"]) / len(stats["gnn"]["scores"]) if stats["gnn"]["scores"] else 0
    print(f"  Computed averages: single={single_avg:.2f}, fixed={fixed_avg:.2f}, gnn={gnn_avg:.2f}")
    print(f"  Claimed averages: single=4.1, fixed=3.8, gnn=1.5")
    print(f"  MATCH: single={'YES' if abs(single_avg - 4.1) < 0.1 else 'NO'}, "
          f"fixed={'YES' if abs(fixed_avg - 3.8) < 0.1 else 'NO'}, "
          f"gnn={'YES' if abs(gnn_avg - 1.5) < 0.1 else 'NO'}")


if __name__ == "__main__":
    main()
