"""Rendering results to markdown.

The output of this module is what gets committed, so it is built around one
constraint: **a committed result must regenerate byte-identically on a
different machine**, or the CI drift gate fails for no useful reason.

That rules out putting wall-clock timings, absolute memory figures, or
anything else machine-dependent in the committed file. Those belong in a
gitignored raw file. What gets committed is the claim: the score, its
interval, whether the delta is reportable, and the provenance needed to check
it.
"""

from __future__ import annotations

from .judge import JudgeAgreement
from .stats import is_reportable
from .types import EvalResult


def render_result(result: EvalResult) -> str:
    """Markdown for one evaluation run.

    Scores are rounded to 4 decimals. Not cosmetic: unrounded floats differ in
    the last bit across platforms, which would make the committed file
    unstable and defeat the drift gate.
    """
    a = result.aggregate
    lines = [
        f"# {result.name}",
        "",
        result.meta.provenance_block(),
        "",
        "## Result",
        "",
        f"- n = {int(a['n'])}",
        f"- mean = {a['mean']:.4f}",
        f"- 95% CI = [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]",
        "",
    ]

    strata = {k: v for k, v in a.items() if k.startswith("mean__")}
    if len(strata) > 1:
        lines += [
            "## By stratum",
            "",
            "A single headline number hides a regressed slice, so every run",
            "reports its strata.",
            "",
            "| stratum | mean |",
            "|---|---|",
        ]
        for k in sorted(strata):
            lines.append(f"| {k.removeprefix('mean__')} | {strata[k]:.4f} |")
        lines.append("")

    return "\n".join(lines)


def render_comparison(
    baseline: EvalResult,
    candidate: EvalResult,
    delta: float,
    ci: tuple[float, float],
) -> str:
    """Markdown for a two-system comparison.

    Leads with whether the difference is reportable rather than with the point
    estimate, because the point estimate is the part people quote and the
    interval is the part that decides whether quoting it is honest.
    """
    lo, hi = ci
    reportable = is_reportable(delta, ci)
    verdict = (
        f"**{delta:+.4f}** (95% CI [{lo:+.4f}, {hi:+.4f}])"
        if reportable
        else f"**not reportable** — point estimate {delta:+.4f}, "
        f"95% CI [{lo:+.4f}, {hi:+.4f}] includes zero"
    )
    return "\n".join([
        f"# {candidate.name} vs {baseline.name}",
        "",
        candidate.meta.provenance_block(),
        "",
        "## Difference",
        "",
        verdict,
        "",
        (
            "The interval excludes zero, so the direction of this difference "
            "is supported by the data."
            if reportable
            else "The interval includes zero. On this sample the two systems "
            "are indistinguishable, and reporting the point estimate as an "
            "improvement would imply a precision the measurement does not have."
        ),
        "",
        "| system | n | mean |",
        "|---|---|---|",
        f"| {baseline.name} | {int(baseline.aggregate['n'])} | "
        f"{baseline.aggregate['mean']:.4f} |",
        f"| {candidate.name} | {int(candidate.aggregate['n'])} | "
        f"{candidate.aggregate['mean']:.4f} |",
        "",
    ])


def render_calibration(agreement: JudgeAgreement, *, judge_name: str,
                       reference_name: str = "human labels") -> str:
    """Markdown for a judge-calibration run.

    This is the block that belongs next to any number an LLM judge produced.
    It states the kappa, names its band, and — when agreement is weak — says
    plainly what that means for downstream numbers, rather than leaving a bare
    figure to be read optimistically.
    """
    k = agreement.kappa
    lines = [
        f"# Judge calibration: {judge_name} vs {reference_name}",
        "",
        f"- Cohen's kappa = **{k:.3f}** ({agreement.interpretation})",
        f"- n = {agreement.n} samples scored by both",
        "",
    ]
    if k < 0.4:
        lines += [
            "**This judge is not measuring what the reference measures.**",
            "Agreement is at or near chance, so any score it produces carries",
            "an unknown error and should not gate a decision until the rubric",
            "is revised and re-calibrated.",
            "",
        ]
    elif k < 0.6:
        lines += [
            "Moderate agreement. Usable for tracking large movements, but too",
            "noisy to adjudicate small differences — pair any delta with the",
            "confidence interval before acting on it.",
            "",
        ]
    else:
        lines += [
            "Agreement is strong enough to use this judge for the decisions",
            "this eval gates. It remains an instrument with a known error",
            "rate, not ground truth.",
            "",
        ]
    return "\n".join(lines)
