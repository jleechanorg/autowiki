# Chimera P4 Hard Benchmark Evidence Bundle

**Bundle:** `benchmark_logs/`
**Run Date:** 2026-04-20/21
**Run ID:** `hard-benchmark-2026-04-20`
**Git Commit:** `a11e11c4a8a1dbad46150097538c9051f8cb231e`

## Claim → Artifact Map

| Claim | Artifact | Verification |
|-------|----------|--------------|
| 15 queries × 3 modes completed | `checkpoint.json` | 45 mode-results present |
| Error rates: single=13.3%, fixed=0%, gnn=0% | `checkpoint.json` + `aggregate_benchmark.py` | Script output confirms |
| Quality averages: single=4.73, fixed=5.00, gnn=5.03 | `checkpoint.json` + `aggregate_benchmark.py` | Script output confirms |
| Rubric ceiling at 5.0 | `checkpoint.json` scores | 14/15 queries scored 5.0 regardless of content |
| Retry logic 3-attempt exponential backoff | `run_hard_benchmark.py:run_benchmark()` + `hard_benchmark.log` | Log shows 8 retries across Q1-Q6 |
| SHA-256 checksums | `checksums_final.sha256` | Verified match via sha256sum |

## Known Issues

1. **Pairwise JSON corruption**: The `_parse_pairwise_result` function in `run_hard_benchmark.py` had bugs causing dimension scores to default to 5.0 and winners to "TIE" when regex failed to match LLM output. This was fixed in commit `0c41e23` but the checkpoint was written before the fix. The raw text in checkpoint.json is intact; a re-parse script can recover correct values.

2. **Rubric ceiling**: All valid outputs score 5.0 - the rubric lacks discrimination. This is a design limitation, not a bug.

3. **Single mode 13.3% error rate**: Above the 5% threshold - caveat required on single mode quality claims.

## Files

- `checkpoint.json` - Per-query results with scores, pairwise comparisons, raw LLM outputs
- `hard_benchmark.log` - Execution log (292.6 min, 8 retries)
- `metadata.json` - Run provenance (git SHA, model, timeout, retry config)
- `methodology.md` - Methodology documentation
- `checksums_final.sha256` - SHA-256 for all result files
- `aggregate_benchmark.py` - Result aggregation script (computes error rates, averages)

## Run Configuration

- Model: minimax-m2.7
- Timeout: 180s per call
- Retry: 3 attempts with exponential backoff (1s→2s→4s)
- Circuit breaker: trip after >3 consecutive failures
- Cache: disabled
- Warm-up: cold start
- Queries: 15 hard research queries
- Modes: single (baseline), fixed (multi-agent pipeline), GNN (dynamic topology)
