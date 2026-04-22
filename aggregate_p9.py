#!/usr/bin/env python3
"""Aggregate P9 benchmark results from checkpoint_p9_q*.json files."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def is_error_state(mode_data: dict) -> bool:
    if "scores" in mode_data:
        scores = mode_data["scores"]
        if "error_flag" in scores:
            return bool(scores["error_flag"])
        if "is_error" in scores:
            return bool(scores["is_error"])
    return False

def compute_statistics(results: list) -> dict:
    stats = defaultdict(lambda: {"scores": [], "errors": 0, "total": 0})
    for r in results:
        for mode in ["single", "fixed", "gnn"]:
            mdata = r["modes"].get(mode, {})
            stats[mode]["total"] += 1
            if is_error_state(mdata):
                stats[mode]["errors"] += 1
                stats[mode]["scores"].append(0.0)
            else:
                score = mdata.get("scores", {}).get("overall", 5.0)
                stats[mode]["scores"].append(score)
    return stats

def pairwise_winners(results: list) -> dict:
    counts = {"single_vs_fixed": {"single": 0, "fixed": 0, "tie": 0},
              "single_vs_gnn": {"single": 0, "gnn": 0, "tie": 0},
              "fixed_vs_gnn": {"fixed": 0, "gnn": 0, "tie": 0}}
    for r in results:
        s = r["modes"]["single"]["scores"].get("overall", 0)
        f = r["modes"]["fixed"]["scores"].get("overall", 0)
        g = r["modes"]["gnn"]["scores"].get("overall", 0)
        if abs(s - f) < 0.5:
            counts["single_vs_fixed"]["tie"] += 1
        elif s > f:
            counts["single_vs_fixed"]["single"] += 1
        else:
            counts["single_vs_fixed"]["fixed"] += 1
        if abs(s - g) < 0.5:
            counts["single_vs_gnn"]["tie"] += 1
        elif s > g:
            counts["single_vs_gnn"]["single"] += 1
        else:
            counts["single_vs_gnn"]["gnn"] += 1
        if abs(f - g) < 0.5:
            counts["fixed_vs_gnn"]["tie"] += 1
        elif f > g:
            counts["fixed_vs_gnn"]["fixed"] += 1
        else:
            counts["fixed_vs_gnn"]["gnn"] += 1
    return counts

def main():
    log_dir = Path(__file__).parent / "benchmark_logs"
    checkpoint_files = sorted(log_dir.glob("checkpoint_p9_q*.json"))
    if not checkpoint_files:
        print("ERROR: No checkpoint_p9_q*.json files found")
        sys.exit(1)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    results = []
    for f in checkpoint_files:
        with open(f) as fh:
            results.append(json.load(fh))
    stats = compute_statistics(results)
    print("\n=== PER-MODE STATISTICS (P9 BEHAVIORAL SCORING) ===")
    for mode, s in stats.items():
        err_rate = s["errors"] / s["total"] * 100 if s["total"] > 0 else 0
        valid_scores = [x for x in s["scores"] if x > 0] if s["scores"] else [0.0]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        print(f"\n{mode.upper()}:")
        print(f"  Error rate: {err_rate:.1f}% ({s['errors']}/{s['total']})")
        print(f"  Valid scores: {sorted(valid_scores)}")
        print(f"  Mean (valid): {avg:.2f}")
        print(f"  Range: {min(valid_scores):.2f} - {max(valid_scores):.2f}" if valid_scores else "  Range: N/A (all errors)")
    pw = pairwise_winners(results)
    print("\n=== PAIRWISE WINNERS ===")
    for pair, counts in pw.items():
        print(f"{pair}: {counts}")
    print("\n=== PER-QUERY SCORES ===")
    print(f"{'Q':>3} {'SINGLE':>8} {'FIXED':>8} {'GNN':>8} {'Winner':>10} {'Behaviors':>12}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["query_num"]):
        q = r["query_num"]
        s = r["modes"]["single"]["scores"].get("overall", 0)
        f = r["modes"]["fixed"]["scores"].get("overall", 0)
        g = r["modes"]["gnn"]["scores"].get("overall", 0)
        behaviors = r["modes"]["fixed"]["scores"].get("behaviors", [])
        scores = [(s, "SINGLE"), (f, "FIXED"), (g, "GNN")]
        winner = max(scores, key=lambda x: x[0])[1] if max(s, f, g) > min(s, f, g) else "TIE"
        print(f"Q{q:>2} {s:>8.2f} {f:>8.2f} {g:>8.2f} {winner:>10} {str(behaviors):>12}")
    out_file = log_dir / "checkpoint_p9.json"
    with open(out_file, "w") as fh:
        json.dump({"results": results, "timestamp": "P9-aggregated"}, fh, indent=2)
    print(f"\nWrote {out_file}")
    print("\n=== ACCEPTANCE CRITERIA ===")
    all_pass = True
    for mode, s in stats.items():
        err_rate = s["errors"] / s["total"] * 100
        ok = err_rate < 5
        print(f"  {mode}: error_rate={err_rate:.1f}% {'PASS' if ok else 'FAIL'}")
        all_pass = all_pass and ok
    valid_scores = [s for mode_s in stats.values() for s in mode_s["scores"] if s > 0]
    score_range = max(valid_scores) - min(valid_scores) if valid_scores else 0
    ok = score_range >= 3
    print(f"  score_range={score_range:.2f} {'PASS' if ok else 'FAIL'} (need >=3)")
    print(f"\n  OVERALL: {'PASS' if all_pass and ok else 'FAIL'}")

if __name__ == "__main__":
    main()
