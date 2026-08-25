"""Tests for evaluation and judge calibration.

The behaviours pinned here are the ones downstream repositories depend on:
that every result carries a confidence interval, that per-stratum means are
always computed, and that calibration reports agreement rather than a verdict.
"""

from __future__ import annotations

import pytest

from llm_eval_harness.judge import ExactMatchJudge, HumanLabelJudge, JudgeAgreement
from llm_eval_harness.runner import calibrate, evaluate
from llm_eval_harness.types import RunMeta, Sample

META = RunMeta.now(hardware="test", model="exact-match", seed=0)


def dataset(n: int = 10) -> list[Sample]:
    return [
        Sample(
            id=str(i),
            inputs={"q": f"item {i}"},
            reference="yes" if i % 2 == 0 else "no",
            metadata={"stratum": "easy" if i < n // 2 else "hard"},
        )
        for i in range(n)
    ]


async def test_a_perfect_candidate_set_scores_one():
    ds = dataset()
    cands = {s.id: s.reference for s in ds}
    r = await evaluate(ds, cands, ExactMatchJudge(), name="t", meta=META)
    assert r.aggregate["mean"] == pytest.approx(1.0)
    assert len(r) == 10


async def test_every_result_carries_a_confidence_interval():
    """No aggregate ships without one. A mean with no interval invites a
    comparison it cannot support, which is the failure this package exists
    to prevent."""
    ds = dataset()
    cands = {s.id: ("yes" if int(s.id) % 3 else "no") for s in ds}
    r = await evaluate(ds, cands, ExactMatchJudge(), name="t", meta=META)
    assert "ci_lo" in r.aggregate and "ci_hi" in r.aggregate
    assert r.aggregate["ci_lo"] <= r.aggregate["mean"] <= r.aggregate["ci_hi"]


async def test_per_stratum_means_are_always_reported():
    """A headline number can hide a regressed slice, so slicing is not
    optional."""
    ds = dataset()
    cands = {s.id: (s.reference if s.stratum == "easy" else "wrong") for s in ds}
    r = await evaluate(ds, cands, ExactMatchJudge(), name="t", meta=META)
    assert r.aggregate["mean__easy"] == pytest.approx(1.0)
    assert r.aggregate["mean__hard"] == pytest.approx(0.0)
    # And the headline average hides exactly that split.
    assert r.aggregate["mean"] == pytest.approx(0.5)


async def test_provenance_travels_with_the_result():
    ds = dataset(4)
    r = await evaluate(ds, {s.id: "x" for s in ds}, ExactMatchJudge(),
                       name="t", meta=META)
    assert r.meta.hardware == "test"
    assert "Hardware: test" in r.meta.provenance_block()


async def test_concurrency_does_not_reorder_or_drop_judgements():
    ds = dataset(30)
    cands = {s.id: s.reference for s in ds}
    r = await evaluate(ds, cands, ExactMatchJudge(), name="t", meta=META, concurrency=8)
    assert len(r) == 30
    assert {j.sample_id for j in r.judgements} == {s.id for s in ds}


# --- calibration -------------------------------------------------------


async def test_calibration_reports_perfect_agreement_when_judges_match():
    """Both judges must actually vary, or kappa is 0 by definition.

    Half the candidates are correct and half wrong, and the human labels
    match. Feeding a dataset where every sample scores 1.0 would exercise the
    degenerate single-category branch instead -- see
    test_kappa_is_zero_when_both_raters_use_one_category.
    """
    ds = dataset()
    cands = {s.id: (s.reference if int(s.id) % 2 == 0 else "wrong") for s in ds}
    human = HumanLabelJudge(
        labels={s.id: (1.0 if int(s.id) % 2 == 0 else 0.0) for s in ds}
    )
    ag = await calibrate(ds, cands, ExactMatchJudge(), human)
    assert ag.kappa == pytest.approx(1.0)
    assert ag.n == 10


async def test_calibration_on_a_single_category_reports_zero_not_one():
    """The trap this catches: if every sample scores the same, two judges
    agree completely while conveying nothing. Reporting 1.0 there would
    certify an instrument that has not been tested."""
    ds = dataset()
    cands = {s.id: s.reference for s in ds}          # all correct
    human = HumanLabelJudge(labels={s.id: 1.0 for s in ds})   # all 1.0
    ag = await calibrate(ds, cands, ExactMatchJudge(), human)
    assert ag.kappa == pytest.approx(0.0)


async def test_calibration_reports_disagreement_honestly():
    """The important case: a judge that systematically disagrees with the
    human should score at or below chance, not be quietly rounded up."""
    ds = dataset()
    cands = {s.id: s.reference for s in ds}          # exact-match says 1.0
    human = HumanLabelJudge(labels={s.id: 0.0 for s in ds})   # human says 0.0
    ag = await calibrate(ds, cands, ExactMatchJudge(), human)
    assert ag.kappa <= 0.0


async def test_calibration_returns_agreement_not_a_verdict():
    """What counts as an acceptable kappa depends on what the eval gates, so
    that judgement stays with the caller."""
    ds = dataset(4)
    cands = {s.id: s.reference for s in ds}
    human = HumanLabelJudge(labels={s.id: 1.0 for s in ds})
    ag = await calibrate(ds, cands, ExactMatchJudge(), human)
    assert isinstance(ag, JudgeAgreement)
    assert not hasattr(ag, "passed")


def test_interpretation_bands_are_stated_not_left_to_the_reader():
    assert JudgeAgreement(kappa=-0.1, n=1).interpretation == "worse than chance"
    assert JudgeAgreement(kappa=0.30, n=1).interpretation == "fair"
    assert JudgeAgreement(kappa=0.65, n=1).interpretation == "substantial"


# --- judges ------------------------------------------------------------


async def test_exact_match_normalises_whitespace_and_case_by_default():
    s = Sample(id="1", inputs={}, reference="Yes")
    assert (await ExactMatchJudge().score(s, "  yes ")).score == 1.0
    assert (await ExactMatchJudge(normalise=False).score(s, "  yes ")).score == 0.0


async def test_exact_match_refuses_a_sample_with_no_reference():
    s = Sample(id="1", inputs={})
    with pytest.raises(ValueError, match="no reference"):
        await ExactMatchJudge().score(s, "anything")


async def test_human_judge_refuses_an_unlabelled_sample_unless_given_a_default():
    s = Sample(id="missing", inputs={})
    with pytest.raises(KeyError, match="no human label"):
        await HumanLabelJudge(labels={}).score(s, "x")
    assert (await HumanLabelJudge(labels={}, default=0.5).score(s, "x")).score == 0.5
