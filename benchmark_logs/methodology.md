# Benchmark Methodology

## Timeout Configuration
- **Total timeout**: 180s per query (increased from 120s)
- Rationale: Complex reasoning tasks require more time for model deliberation

## Retry Strategy
- **Retry attempts**: 3 per query
- **Backoff**: Exponential (1s, 2s, 4s)
- **Trigger**: Any non-200 HTTP response or timeout

## Circuit Breaker
- **Threshold**: >3 consecutive failures
- **Pause duration**: 60s before resuming
- **State**: Tracks consecutive failures, resets on success

## Execution Conditions
- **Warm-up**: No warm-up queries (cold run)
- **Cache**: Disabled (fresh API calls for each query)
- **Reproducibility**: No seed fixing

## Evaluation Method
- **Judge**: LLM-as-judge
- **Rubric**: 5-dimension scoring
  1. Correctness
  2. Completeness
  3. Code quality
  4. Documentation
  5. Edge case handling

## Query Set
- **Total queries**: 15 hard benchmark queries
- **Modes tested**: single, fixed, gnn
- **Expected runtime**: ~45 minutes for full benchmark