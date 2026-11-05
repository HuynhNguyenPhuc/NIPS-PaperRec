"""Paired statistics for per-query metric differences between two recommenders."""

from __future__ import annotations

import numpy as np

# Resamples are drawn in blocks so the index matrix never holds all of them at once.
CHUNK = 1000


def bootstrap_and_permute(d: np.ndarray, boot: int, rng: np.random.Generator) -> dict:
    """
    Paired bootstrap confidence interval and sign-flip permutation p-value for a mean difference.

    Args:
        d: Per-query paired differences (method minus baseline).
        boot: Number of bootstrap resamples and sign-flip draws.
        rng: Random generator.

    Returns:
        stats: Mean, 95% bootstrap interval, two-sided sign-flip p-value.
    """
    n = len(d)
    mean = float(d.mean())
    boot_means = np.empty(boot)
    perm_means = np.empty(boot)

    # Each chunk fills its own slice of both result arrays before the next one starts.
    for start in range(0, boot, CHUNK):
        size = min(CHUNK, boot - start)

        # Bootstrap resamples the queries with replacement, keeping each query's pair intact.
        idx = rng.integers(0, n, size=(size, n))
        boot_means[start:start + size] = d[idx].mean(axis=1)

        # Sign flips test whether the observed mean is larger than a symmetric null would give.
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        perm_means[start:start + size] = np.abs((signs * d).mean(axis=1))

    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    # The +1 terms count the observed statistic itself, so p is never exactly zero.
    p = (np.sum(perm_means >= abs(mean) - 1e-12) + 1) / (boot + 1)

    return {"mean": mean, "ci_low": float(lo), "ci_high": float(hi), "p_signflip": float(p)}


def paired_summary(a: list, b: list, boot: int, rng: np.random.Generator) -> dict:
    """
    Win, tie and loss counts and conditional means for the paired differences a - b.

    Args:
        a: Per-query values of the method under test.
        b: Per-query values of the baseline, same queries in the same order.
        boot: Number of bootstrap resamples.
        rng: Random generator.

    Returns:
        summary: Counts, conditional means and the bootstrap/permutation statistics.
    """
    d = np.asarray(a, float) - np.asarray(b, float)

    # A tolerance keeps floating-point noise from turning a tie into a win or loss.
    wins = d > 1e-9
    losses = d < -1e-9
    ties = ~(wins | losses)

    # Counts and conditional means are reported alongside the inference statistics.
    out = {
        "n": int(len(d)),
        "wins": int(wins.sum()),
        "ties": int(ties.sum()),
        "losses": int(losses.sum()),
        "mean_gain_on_wins": float(d[wins].mean()) if wins.any() else 0.0,
        "mean_loss_on_losses": float(d[losses].mean()) if losses.any() else 0.0,
    }

    # The bootstrap and permutation statistics are added to the same dict.
    out.update(bootstrap_and_permute(d, boot, rng))

    return out
