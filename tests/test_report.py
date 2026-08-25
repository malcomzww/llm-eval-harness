"""Tests for markdown rendering.

The property that matters: a committed result must regenerate identically on
a different machine, or the CI drift gate fails for no useful reason. These
tests pin that, and pin the fact that a non-reportable delta is labelled as
such rather than presented as an improvement.
"""

from __future__ import annotations

from llm_eval_harness.judge import JudgeAgreement
from llm_eval_harness.report import render_calibration, render_comparison, render_result
from llm_eval_harness.types import EvalResult, Judgement, RunMeta

META = RunMeta(date="2026-08-25", hardware="test", model="m", seed=0,
               command="make bench", artifact="results/raw.json")


def result(name: str, mean: float, n: int = 10, strata: bool = False) -> EvalResult:
    agg = {"n": float(n), "mean": mean, "ci_lo": mean - 0.1, "ci_hi": mean + 0.1}
    if strata:
        agg |= {"mean__easy": mean + 0.1, "mean__hard": mean - 0.1}
    return EvalResult(
        name=name,
        judgements=[Judgement(sample_id=str(i), score=mean) for i in range(n)],
        aggregate=agg,
        meta=META,
    )


def test_rendering_is_deterministic():
    """Same input, same bytes. The drift gate depends on this."""
    r = result("demo", 0.5, strata=True)
    assert render_result(r) == render_result(r)


def test_every_result_carries_its_provenance():
    out = render_result(result("demo", 0.5))
    for field in ("2026-08-25", "test", "make bench", "Seed: 0"):
        assert field in out


def test_scores_are_rounded_so_the_output_is_platform_stable():
    """Unrounded floats differ in the last bit across platforms, which would
    make the committed file unstable."""
    r = result("demo", 1.0 / 3.0)
    out = render_result(r)
    assert "0.3333" in out
    assert "0.33333333" not in out


def test_strata_are_rendered_when_present():
    out = render_result(result("demo", 0.5, strata=True))
    assert "By stratum" in out and "| easy |" in out and "| hard |" in out


def test_no_strata_section_when_there_is_only_one_stratum():
    assert "By stratum" not in render_result(result("demo", 0.5))


# --- the reporting rule ------------------------------------------------


def test_a_reportable_delta_is_stated_plainly():
    out = render_comparison(result("base", 0.5), result("cand", 0.7),
                            0.20, (0.15, 0.25))
    assert "+0.2000" in out
    assert "excludes zero" in out


def test_a_delta_straddling_zero_is_labelled_not_reportable():
    """The failure this whole package exists to prevent: a point estimate
    presented as an improvement when the interval includes zero."""
    out = render_comparison(result("base", 0.5), result("cand", 0.52),
                            0.02, (-0.05, 0.09))
    assert "not reportable" in out
    assert "includes zero" in out
    assert "indistinguishable" in out


# --- calibration -------------------------------------------------------


def test_weak_agreement_is_called_out_explicitly():
    out = render_calibration(JudgeAgreement(kappa=0.2, n=100), judge_name="j")
    assert "not measuring what the reference measures" in out
    assert "should not gate a decision" in out


def test_moderate_agreement_warns_about_small_differences():
    out = render_calibration(JudgeAgreement(kappa=0.5, n=100), judge_name="j")
    assert "too" in out and "noisy" in out


def test_strong_agreement_still_refuses_to_call_the_judge_ground_truth():
    out = render_calibration(JudgeAgreement(kappa=0.85, n=100), judge_name="j")
    assert "not ground truth" in out


def test_calibration_reports_the_sample_size():
    assert "n = 250" in render_calibration(JudgeAgreement(kappa=0.7, n=250),
                                           judge_name="j")
