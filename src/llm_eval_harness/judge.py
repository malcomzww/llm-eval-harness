"""Judges: things that score a candidate answer against a sample.

Public API frozen at v0.1.0. Eleven repositories build against this, so the
shape is fixed before any of them starts; implementations land behind it.

The design position this module takes: an LLM judge is an instrument, and an
uncalibrated instrument produces numbers with no meaning. `HumanLabelJudge`
exists so that every LLM judge can be scored against human labels on the same
samples, and the resulting kappa reported alongside any number the judge
produces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import Judgement, Sample


@dataclass(frozen=True)
class Rubric:
    """Scoring instructions given to an LLM judge.

    `criteria` are stated separately rather than embedded in one prose blob so
    that a judge can be asked to score each independently -- a single holistic
    score conflates dimensions that often move in opposite directions.
    """

    name: str
    instructions: str
    criteria: tuple[str, ...] = ()
    scale: tuple[float, float] = (0.0, 1.0)
    examples: tuple[tuple[str, float], ...] = ()

    def render(self, sample: Sample, candidate: str) -> str:
        """Build the prompt text for one judgement."""
        raise NotImplementedError("Rubric.render lands in v0.2.0")


@runtime_checkable
class Judge(Protocol):
    """Anything that can score a candidate for a sample.

    A Protocol rather than a base class: downstream repos frequently need a
    domain-specific judge (a SQL-execution checker, a field-level extraction
    comparator) and should not have to inherit from this package to be usable
    by `evaluate`.
    """

    async def score(self, sample: Sample, candidate: str) -> Judgement: ...


class LLMJudge:
    """Scores with a model, over an OpenAI-shape endpoint.

    Bias controls are constructor arguments rather than optional extras
    because they are not optional in practice: position bias alone can move a
    pairwise win rate by double digits.

    Args:
        client: an ``llm_client_kit.LLMClient``.
        model: model id to judge with. Recorded in ``RunMeta`` -- a kappa is
            only meaningful against a named judge.
        rubric: scoring instructions.
        seed: passed to the endpoint where supported.
        swap_positions: score both orderings and average, cancelling the
            tendency to favour whichever candidate is shown first.
        length_control: report a length-normalised score alongside the raw
            one, so verbosity cannot be mistaken for quality.
        cassette: record/replay path. Recorded once, replayed in CI, so tests
            are free and deterministic.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        rubric: Rubric,
        seed: int = 0,
        swap_positions: bool = True,
        length_control: bool = True,
        cassette: Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.rubric = rubric
        self.seed = seed
        self.swap_positions = swap_positions
        self.length_control = length_control
        self.cassette = cassette

    async def score(self, sample: Sample, candidate: str) -> Judgement:
        raise NotImplementedError("LLMJudge.score lands in v0.2.0")


@dataclass
class HumanLabelJudge:
    """Replays human labels as judgements.

    The reference instrument. Running this and an ``LLMJudge`` over the same
    samples and comparing with ``stats.cohen_kappa`` is how this harness
    establishes whether the LLM judge measures what a human measures.
    """

    labels: Mapping[str, float]
    default: float | None = None

    async def score(self, sample: Sample, candidate: str) -> Judgement:
        if sample.id in self.labels:
            return Judgement(sample_id=sample.id, score=float(self.labels[sample.id]),
                             label="human")
        if self.default is None:
            raise KeyError(
                f"no human label for sample {sample.id!r}; pass default= to "
                "score unlabelled samples instead of failing"
            )
        return Judgement(sample_id=sample.id, score=self.default, label="human-default")


@dataclass
class ExactMatchJudge:
    """Deterministic string comparison. No model, no cost, no variance.

    Included because the honest first question about any LLM judge is whether
    the task needed one. If exact match answers it, the cheaper instrument
    wins and the kappa question never arises.
    """

    normalise: bool = True

    async def score(self, sample: Sample, candidate: str) -> Judgement:
        ref = sample.reference
        if ref is None:
            raise ValueError(f"sample {sample.id!r} has no reference to match against")
        if self.normalise:
            a, b = ref.strip().lower(), candidate.strip().lower()
        else:
            a, b = ref, candidate
        return Judgement(sample_id=sample.id, score=1.0 if a == b else 0.0,
                         label="exact" if a == b else "mismatch")


@dataclass(frozen=True)
class JudgeAgreement:
    """The output of calibrating one judge against another."""

    kappa: float
    n: int
    judge_scores: Sequence[float] = field(default_factory=tuple)
    reference_scores: Sequence[float] = field(default_factory=tuple)

    @property
    def interpretation(self) -> str:
        """Landis and Koch bands, stated so a bare number is not left to be
        read optimistically."""
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"
