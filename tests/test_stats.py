"""Tests for the statistics module.

These pin behaviour that eleven downstream repositories rely on, so they test
claims rather than implementation: that kappa punishes uninformative
agreement, that a bootstrap interval is reproducible across runs, and that
the reportability rule actually refuses a delta whose interval straddles zero.

Known-answer checks use values from standard tables, so a refactor that
silently changes the maths fails here rather than in a results file.
"""

from __future__ import annotations

import pytest

from llm_eval_harness.stats import (
    _z,
    bootstrap_ci,
    cohen_kappa,
    is_reportable,
    mean,
    min_n_for_delta,
    paired_delta_ci,
)

# --- agreement ---------------------------------------------------------


def test_kappa_is_one_for_perfect_agreement():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_kappa_is_minus_one_for_systematic_disagreement():
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_kappa_is_zero_when_both_raters_use_one_category():
    """The case raw agreement gets wrong.

    Two raters who both always say "good" agree 100% of the time and convey
    nothing. Raw agreement reports 1.0; kappa reports 0.0, which is the
    honest answer and the reason this harness uses kappa at all.
    """
    assert cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(0.0)


def test_kappa_below_chance_is_negative():
    """Mostly-agreeing raters on a skewed distribution can still score below
    chance -- worth asserting, because a naive reader treats any positive
    agreement as good."""
    a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    b = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    assert cohen_kappa(a, b) < 0.0


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="lengths differ"):
        cohen_kappa([1, 0], [1, 0, 1])


def test_kappa_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        cohen_kappa([], [])


def test_score_of_exactly_one_lands_in_the_top_bin():
    """Off-by-one guard: int(1.0 * 2) == 2 would index past the last bin."""
    assert cohen_kappa([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# --- bootstrap ---------------------------------------------------------


def test_bootstrap_ci_brackets_the_point_estimate():
    values = [0.1, 0.4, 0.5, 0.6, 0.9] * 20
    lo, hi = bootstrap_ci(values)
    assert lo < mean(values) < hi


def test_bootstrap_ci_is_reproducible_for_a_given_seed():
    """An unseeded interval that shifts between runs cannot be committed to a
    results file -- the drift gate would fail on every regeneration."""
    v = [0.2, 0.5, 0.7, 0.9] * 25
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)


def test_the_interval_is_stable_across_seeds_at_full_resamples():
    """Seed-independence is a feature here, not a coincidence.

    At 10k resamples on a well-behaved sample the bootstrap has converged, so
    different seeds land on the same percentiles. That is what makes an
    interval safe to commit to a results file: it survives regeneration on a
    different run without tripping the drift gate.

    Convergence is the claim, so it is asserted at full resamples and
    contrasted below with the small-resample case where seeds do diverge.

    Compared with a tolerance rather than exactly: the resample sets converge
    to the same percentiles, but summing the same floats in a different order
    differs in the last bit. Exact equality passed on Windows and failed on
    the Linux CI runner -- the opposite of the property being asserted.
    """
    v = [0.2, 0.5, 0.7, 0.9] * 25
    a = bootstrap_ci(v, seed=1)
    b = bootstrap_ci(v, seed=2)
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_too_few_resamples_makes_the_interval_seed_dependent():
    """The failure mode the default guards against.

    With only 50 resamples the estimate has not converged and the interval
    moves with the seed -- which is exactly when a committed number starts
    changing between runs.
    """
    v = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
    a = bootstrap_ci(v, n_resamples=50, seed=1)
    b = bootstrap_ci(v, n_resamples=50, seed=99)
    assert a != b


def test_more_samples_narrow_the_interval():
    """The property that makes sample size worth planning."""
    base = [0.3, 0.5, 0.7]
    small = bootstrap_ci(base * 10, seed=0)
    large = bootstrap_ci(base * 200, seed=0)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_zero_variance_gives_a_degenerate_interval():
    lo, hi = bootstrap_ci([0.5] * 50)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_rejects_empty_and_bad_confidence():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.5], confidence=1.5)


# --- paired comparison and the reporting rule --------------------------


def test_paired_delta_detects_a_consistent_improvement():
    a = [0.60] * 50
    b = [0.70] * 50
    delta, lo, hi = paired_delta_ci(a, b)
    assert delta == pytest.approx(0.10)
    assert is_reportable(delta, (lo, hi))


def test_a_delta_whose_interval_straddles_zero_is_not_reportable():
    """The rule the whole harness exists to enforce.

    Noise centred on zero produces a non-zero point estimate. Reporting that
    as an improvement is the most common way eval results mislead.
    """
    a = [0.5, 0.6, 0.4, 0.55, 0.45] * 10
    b = [0.6, 0.5, 0.5, 0.45, 0.55] * 10
    delta, lo, hi = paired_delta_ci(a, b)
    assert lo < 0.0 < hi
    assert not is_reportable(delta, (lo, hi))


def test_reportable_accepts_a_consistent_regression():
    """Direction-agnostic: a reliable *drop* is just as reportable."""
    assert is_reportable(-0.1, (-0.15, -0.05))


def test_paired_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        paired_delta_ci([0.1, 0.2], [0.1])


# --- power -------------------------------------------------------------


def test_inverse_normal_matches_published_quantiles():
    assert _z(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _z(0.800) == pytest.approx(0.841621, abs=1e-5)
    assert _z(0.500) == pytest.approx(0.0, abs=1e-9)
    assert _z(0.025) == pytest.approx(-1.959964, abs=1e-5)


def test_smaller_effects_need_more_samples():
    assert min_n_for_delta(0.01, 0.5) > min_n_for_delta(0.10, 0.5)


def test_noisier_measurements_need_more_samples():
    assert min_n_for_delta(0.05, 1.0) > min_n_for_delta(0.05, 0.2)


def test_sample_size_matches_the_closed_form():
    """n = 2(z_a + z_b)^2 sd^2 / d^2; ~1570 per arm for d=0.05, sd=0.5."""
    assert min_n_for_delta(0.05, 0.5) == 1570


def test_sizing_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="standard deviation"):
        min_n_for_delta(0.1, 0.0)
    with pytest.raises(ValueError, match="zero effect"):
        min_n_for_delta(0.0, 0.5)
