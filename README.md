# Auditable rare-event estimation for language-model evaluation

Weighted-ensemble (WE) estimates of rare failure probabilities, with a committed seed, a Merkle-logged transcript, and a verifier that replays it. Everything runs on a laptop CPU with free tools.

**Question.** A splitting estimator gives its operator many hidden choices: seeds, levels, which runs to keep. Can a third party check that a reported rare-event probability came from the run that was committed to?

**What is here**

| File | Purpose |
|---|---|
| `toy_we.py` | Random-walk toy with an exact answer; WE vs Monte Carlo; hash-derived seeds |
| `audited_we.py` | Toy audit with commitment, Merkle log, step-level verifier, four cheats |
| `cell_audit.py` | Toy audit with cheap bookkeeping checks plus model-move replay |
| `det_test.py` | Bitwise reproducibility test for TinyStories-33M on CPU |
| `events.py`, `lm.py`, `exact_truth.py` | Exact ground truth by enumeration (top-50 sampler, 3 tokens) |
| `real_audit_core.py`, `real_audit.py` | WE on the real model, transcript log, verifier, attacks |
| `audit_strategies.py` | Equal-cost comparison of audit strategies, including the feasibility-pruned rule |
| `test_irrelevant.py` | Forging event-irrelevant cells leaves the estimate unchanged |
| `plot_results.py` | Toy detection plot |

**Reproduce** (needs only [uv](https://docs.astral.sh/uv/))

```
uv init audited-we && cd audited-we
uv add numpy matplotlib torch transformers
uv run python toy_we.py
uv run python audited_we.py 100
uv run python cell_audit.py 100
uv run python det_test.py            # run twice, compare digests
uv run python exact_truth.py
uv run python real_audit.py 1000 0 2530 2530
uv run python real_audit.py 200 30 2530 2530
uv run python audit_strategies.py 30 2530
uv run python test_irrelevant.py 10 2530
```

**Headline results** (TinyStories-33M, CPU, float32, 1 thread, torch 2.15.0; event: at least 2 of 3 sampled tokens in a 40-word trigger set; exact probability 1.4225e-05)

- Unbiased estimate: WE mean 1.53e-05 +/- 1.0e-06 over 1,000 runs (z = +1.00 vs exact).
- At equal model moves, WE saw the event in 469 of 1,000 runs; Monte Carlo in 42 of 1,000. Rough variance reduction about 3.6x.
- Bookkeeping checks catch any weight tampering with zero model passes.
- Replaying only event-relevant cells needs about 53 of 460 distinct-prefix model passes (about 11.5%) and caught every forged move that can change the estimate.

**Limits.** One event, three steps, one small model, top-k sampling. The audit assumes bitwise-reproducible sampling (pinned CPU runtime), so it does not cover hosted APIs. The pruning rule needs an event defined as a count threshold. Half of WE runs still return zero, so report averages over replicates.
