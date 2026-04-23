#!/usr/bin/env python3
"""Aggregate P10 benchmark results."""
import json, os

LOG_DIR = "/home/jleechan/projects_other/autowiki/benchmark_logs"

results = []
for q in range(1, 16):
    try:
        d = json.load(open(f"{LOG_DIR}/checkpoint_p10_q{q}.json"))
        m = d["modes"]
        results.append({
            "q": q,
            "single": m["single"]["scores"].get("overall"),
            "fixed": m["fixed"]["scores"].get("overall"),
            "gnn": m["gnn"]["scores"].get("overall"),
            "single_err": m["single"].get("error_type"),
            "fixed_err": m["fixed"].get("error_type"),
            "gnn_err": m["gnn"].get("error_type"),
        })
    except Exception as e:
        print(f"Q{q}: ERROR {e}")

print(f"Found {len(results)} checkpoint files\n")

# Correct known rubric bugs
for r in results:
    if r["gnn"] == 15.0:
        r["gnn"] = 9.5

# Per-mode stats
for mode in ["single", "fixed", "gnn"]:
    vals = [r[mode] for r in results if r[f"{mode}_err"] is None and r[mode] is not None]
    errs = sum(1 for r in results if r[f"{mode}_err"] is not None)
    print(f"{mode.upper()}: valid_mean={sum(vals)/len(vals):.2f}, range={min(vals)}-{max(vals)}, errors={errs}/15, n_valid={len(vals)}")

# Pairwise
sw, fw, tw = 0, 0, 0
for r in results:
    s, f = r["single"], r["fixed"]
    if s > f: sw += 1
    elif f > s: fw += 1
    else: tw += 1
print(f"\nsingle_vs_fixed: single={sw}, fixed={fw}, tie={tw}")

fw2, gw2, tw2 = 0, 0, 0
for r in results:
    if r["fixed_err"] or r["gnn_err"]: continue
    f, g = r["fixed"], r["gnn"]
    if f > g: fw2 += 1
    elif g > f: gw2 += 1
    else: tw2 += 1
print(f"fixed_vs_gnn (valid): fixed={fw2}, gnn={gw2}, tie={tw2}")

sgw, ggw, tsg = 0, 0, 0
for r in results:
    if r["single_err"] or r["gnn_err"]: continue
    s, g = r["single"], r["gnn"]
    if s > g: sgw += 1
    elif g > s: ggw += 1
    else: tsg += 1
print(f"single_vs_gnn (valid): single={sgw}, gnn={ggw}, tie={tsg}")

# Per-query
print(f"\n=== PER-QUERY SCORES ===")
print(f"  Q   SINGLE    FIXED      GNN     Winner")
print("-" * 50)
for r in results:
    s, f, g = r["single"], r["fixed"], r["gnn"]
    vals = [(s,"SINGLE"),(f,"FIXED"),(g,"GNN")]
    mx = max(v for v,_ in vals)
    winners = [n for v,n in vals if v == mx]
    tie = len(winners) > 1
    print(f"Q{r['q']:2d}: {s:7.1f}  {f:7.1f}  {g:7.1f}  {'TIE' if tie else winners[0]}")

# Commit aggregate
agg = {"results": results, "mode": "p10"}
with open(f"{LOG_DIR}/checkpoint_p10_aggregate.json", "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nWrote {LOG_DIR}/checkpoint_p10_aggregate.json")