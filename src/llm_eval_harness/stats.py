"""Statistics for reporting eval results honestly.

Pure and synchronous by design. No I/O, no API calls, no async. Most of the
repositories that depend on this harness need exactly these functions and
nothing else, so keeping them free of the LLM machinery means they work
offline, in CI, and in a unit test without a single mock.

The governing idea: a delta without a confidence interval is not a result.
Two systems scoring 71% and 74% on 100 samples are indistinguishable, and
reporting "+3 points" implies a precision the measurement does not have.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

Number = float


def mean(xs: Sequence[Number]) -> float:
    if not xs:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(xs) / len(xs)


# --- agreement ---------------------------------------------------------


def cohen_kappa(a: Sequence[Number], b: Sequence[Number], *, bins: int = 2) -> float:
    """Cohen's kappa between two raters, after binning continuous scores.

    Why kappa rather than raw agreement: if 90% of your samples are "good",
    two raters who both always say "good" agree 90% of the time while carrying
    no information. Kappa subtracts the agreement expected by chance, so that
    pair scores 0.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for
    systematic disagreement. Landis and Koch call 0.41-0.60 moderate and
    0.61-0.80 substantial; below ~0.4 a judge is not measuring what the human
    is measuring, and any number built on it inherits that.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compute kappa on an empty sample")

    def binned(x: Number) -> int:
        # Clamp so a score of exactly 1.0 lands in the top bin rather than
        # falling off the end.
        return min(bins - 1, max(0, int(x * bins)))

    ax, bx = [binned(x) for x in a], [binned(x) for x in b]
    n = len(ax)

    observed = sum(1 for i in range(n) if ax[i] == bx[i]) / n

    expected = 0.0
    for k in range(bins):
        pa = sum(1 for x in ax if x == k) / n
        pb = sum(1 for x in bx if x == k) / n
        expected += pa * pb

    if expected == 1.0:
        # Both raters used exactly one category. Agreement is total but
        # uninformative; kappa is undefined, and 0.0 is the honest answer.
        return 0.0
    return (observed - expected) / (1.0 - expected)


# --- uncertainty -------------------------------------------------------


def bootstrap_ci(
    values: Sequence[Number],
    *,
    statistic: Callable[[Sequence[Number]], float] = mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Resamples with replacement rather than assuming normality, which matters
    because eval scores are frequently bounded, skewed, or bimodal -- none of
    which a t-interval handles well.

    Seeded so the interval is reproducible: an unseeded CI that shifts between
    runs cannot be committed to a results file.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")

    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_resamples):
        stats.append(statistic([values[rng.randrange(n)] for _ in range(n)]))
    stats.sort()

    alpha = 1.0 - confidence
    lo = stats[max(0, int((alpha / 2) * n_resamples))]
    hi = stats[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


def paired_delta_ci(
    a: Sequence[Number],
    b: Sequence[Number],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on the paired difference b - a. Returns (delta, lo, hi).

    Paired, because both systems are scored on the *same* samples. Treating
    them as independent throws away that pairing and inflates the interval,
    which hides real improvements.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples must be equal length: {len(a)} vs {len(b)}")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return mean(diffs), lo, hi


def is_reportable(delta: float, ci: tuple[float, float]) -> bool:
    """Does the interval exclude zero?

    The rule this harness enforces: if the CI straddles zero, you did not
    measure an improvement, and reporting the point estimate as one is the
    most common way eval results mislead.
    """
    lo, hi = ci
    return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)


def min_n_for_delta(delta: float, sd: float, *, power: float = 0.8,
                    alpha: float = 0.05) -> int:
    """Samples needed to detect `delta` at the given power. Two-sided.

    Answers the question that should precede any eval: is my golden set big
    enough to see the effect I care about? Building a 50-item set to detect a
    2-point difference is wasted effort, and knowing that in advance is
    cheaper than discovering it afterwards.
    """
    if sd <= 0:
        raise ValueError("standard deviation must be positive")
    if delta == 0:
        raise ValueError("cannot size a study for a zero effect")

    # Normal approximation. z for the two-sided alpha and one-sided power.
    z_alpha = _z(1 - alpha / 2)
    z_power = _z(power)
    n = 2 * ((z_alpha + z_power) ** 2) * (sd**2) / (delta**2)
    return math.ceil(n)


def _z(p: float) -> float:
    """Inverse normal CDF, via the Acklam rational approximation.

    Accurate to about 1e-9, which is far beyond what a power calculation
    needs. Hand-rolled to keep this module dependency-free: pulling in scipy
    for a single quantile would make every downstream repository carry it.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
