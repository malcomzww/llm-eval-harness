"""Running an evaluation, and calibrating a judge against a reference.

Public API frozen at v0.1.0.

`calibrate` is the function this package exists for. It runs two judges over
the same samples and reports their agreement, which is the step that turns an
LLM judge from an assumption into an instrument with a known error rate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from .judge import Judge, JudgeAgreement
from .stats import bootstrap_ci, cohen_kappa, mean
from .types import EvalResult, Judgement, RunMeta, Sample


async def evaluate(
    dataset: Sequence[Sample],
    candidates: Mapping[str, str],
    judge: Judge,
    *,
    name: str,
    meta: RunMeta,
    concurrency: int = 8,
) -> EvalResult:
    """Score every sample and aggregate.

    Args:
        dataset: the golden set.
        candidates: sample id -> the answer being scored.
        judge: the instrument.
        name: run label, used in the report.
        meta: provenance. Required, not optional -- a result without it cannot
            be committed under this portfolio's conventions.
        concurrency: bounded, so a large dataset does not open one connection
            per sample. The default is deliberately modest; several
            evaluations frequently run at once.

    Aggregates include a bootstrap CI on the mean, because a mean score with
    no interval invites comparison it cannot support.
    """
    sem = asyncio.Semaphore(concurrency)
    judgements: list[Judgement | None] = [None] * len(dataset)

    async def one(i: int, s: Sample) -> None:
        async with sem:
            judgements[i] = await judge.score(s, candidates[s.id])

    await asyncio.gather(*(one(i, s) for i, s in enumerate(dataset)))
    done = [j for j in judgements if j is not None]

    scores = [j.score for j in done]
    lo, hi = bootstrap_ci(scores, seed=meta.seed) if scores else (0.0, 0.0)
    aggregate = {
        "n": float(len(done)),
        "mean": mean(scores) if scores else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
    }

    # Per-stratum means, so a headline number cannot hide a regressed slice.
    for stratum in {s.stratum for s in dataset}:
        ids = {s.id for s in dataset if s.stratum == stratum}
        sub = [j.score for j in done if j.sample_id in ids]
        if sub:
            aggregate[f"mean__{stratum}"] = mean(sub)

    return EvalResult(name=name, judgements=done, aggregate=aggregate, meta=meta)


async def calibrate(
    dataset: Sequence[Sample],
    candidates: Mapping[str, str],
    judge: Judge,
    reference: Judge,
    *,
    bins: int = 2,
    concurrency: int = 8,
    meta: RunMeta | None = None,
) -> JudgeAgreement:
    """Score the same samples with two judges and report their agreement.

    The reference is normally a `HumanLabelJudge`. The returned kappa is the
    number that belongs next to every result the judge under test produces:
    without it, a reported score is an instrument reading of unknown accuracy.

    Deliberately returns agreement rather than a pass/fail. What counts as an
    acceptable kappa depends on what the eval gates, and that judgement stays
    with the caller.
    """
    m = meta or RunMeta.now(hardware="unspecified", model="unspecified")
    a = await evaluate(dataset, candidates, judge, name="judge", meta=m,
                       concurrency=concurrency)
    b = await evaluate(dataset, candidates, reference, name="reference", meta=m,
                       concurrency=concurrency)

    by_id_a = {j.sample_id: j.score for j in a.judgements}
    by_id_b = {j.sample_id: j.score for j in b.judgements}
    shared = [i for i in by_id_a if i in by_id_b]
    if not shared:
        raise ValueError("judges scored no samples in common; cannot compute agreement")

    xs = [by_id_a[i] for i in shared]
    ys = [by_id_b[i] for i in shared]
    return JudgeAgreement(
        kappa=cohen_kappa(xs, ys, bins=bins),
        n=len(shared),
        judge_scores=tuple(xs),
        reference_scores=tuple(ys),
    )
