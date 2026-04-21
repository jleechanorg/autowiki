#!/usr/bin/env python3
"""Aggregate benchmark results from checkpoint.json and compute per-mode statistics."""

import json
import sys
from pathlib import Path
from collections import defaultdict

CHECKPOINT_FILE = Path(__file__).parent / "benchmark_logs" / "checkpoint.json"


def is_error_state(mode_data: dict) -> bool:
    """Check if a mode output is in error state using error_flag from checkpoint.

    Uses the checkpoint's error_flag which is set by _is_error_output()
    with conservative detection (first 500 chars only) to avoid false positives.
    Falls back to regex search on first 500 chars only if error_flag is not present (legacy checkpoints).
    """
    import re
    # Use error_flag from checkpoint if present (set by _is_error_output with 500-char limit)
    if "scores" in mode_data:
        scores = mode_data["scores"]
        # error_flag = True means error, False means valid, None means not computed (legacy)
        if "error_flag" in scores:
            return bool(scores["error_flag"])
    # Fallback: use same patterns as _is_error_output, limited to first 500 chars
    search_region = mode_data.get("output", "")[:500]
    error_patterns = [
        r'\[API Error\]',
        r'\btimeout\b',
        r'\b529\b',
        r'\brate limit\b',
        r'\bconnection error\b',
        r'\bupstream error\b',
        r'\bservice unavailable\b',
        r'\btoo many requests\b',
        r'\b429\b',
        r'\binternal server error\b',
        r'\b500\b',
        r'\b502\b',
        r'\b503\b',
    ]
    for pattern in error_patterns:
        if re.search(pattern, search_region, re.IGNORECASE):
            return True
    return False


def compute_statistics(data: dict) -> dict:
    """Compute per-mode statistics from benchmark data."""
    results = data.get("results", data)  # Handle both dict with 'results' key and raw list

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
    if not CHECKPOINT_FILE.exists():
        print(f"ERROR: {CHECKPOINT_FILE} not found")
        sys.exit(1)

    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)

    stats = compute_statistics(data)
    print_report(stats)

    # Return comparison values for verification
    print("\n" + "=" * 60)
    print("VERIFICATION CHECK (Run 3 values):")
    single_avg = sum(stats["single"]["scores"]) / len(stats["single"]["scores"]) if stats["single"]["scores"] else 0
    fixed_avg = sum(stats["fixed"]["scores"]) / len(stats["fixed"]["scores"]) if stats["fixed"]["scores"] else 0
    gnn_avg = sum(stats["gnn"]["scores"]) / len(stats["gnn"]["scores"]) if stats["gnn"]["scores"] else 0
    print(f"  Computed averages: single={single_avg:.2f}, fixed={fixed_avg:.2f}, gnn={gnn_avg:.2f}")
    # Run 3 actual values from checkpoint analysis
    print(f"  Run 3 actuals:    single=4.73, fixed=5.00, gnn=5.03")
    print(f"  MATCH: single={'YES' if abs(single_avg - 4.73) < 0.1 else 'NO'}, "
          f"fixed={'YES' if abs(fixed_avg - 5.00) < 0.1 else 'NO'}, "
          f"gnn={'YES' if abs(gnn_avg - 5.03) < 0.1 else 'NO'}")


if __name__ == "__main__":
    main()
