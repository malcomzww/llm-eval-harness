"""Core data types.

Frozen at v0.1.0 -- eleven repositories build against these, so the shape of
what crosses the boundary is fixed before any of them starts. Additive changes
(new optional fields) are fine; renaming or removing anything is not.

Everything here is a frozen dataclass. Evaluation results get passed around,
cached, and written to disk; making them immutable removes a whole class of
"something mutated my results between measuring and reporting" bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class RunMeta:
    """Provenance for a single evaluation run.

    Every committed number in this portfolio carries these fields. Without
    them a result is an assertion; with them it is a measurement someone else
    can check. `command` in particular must be literally runnable.
    """

    date: str                      # ISO date, e.g. "2026-08-25"
    hardware: str                  # "24-core CPU, 32GB, no GPU"
    model: str                     # repo id, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None    # exact model commit SHA, when pinned
    seed: int = 0
    command: str = ""              # the command that reproduces this
    artifact: str = ""             # path to the raw output this summarises

    @classmethod
    def now(cls, *, hardware: str, model: str, **kw: Any) -> RunMeta:
        return cls(date=date.today().isoformat(), hardware=hardware, model=model, **kw)

    def provenance_block(self) -> str:
        """The markdown block that must accompany any reported number."""
        lines = [
            f"- Date: {self.date}",
            f"- Hardware: {self.hardware}",
            f"- Model: `{self.model}`" + (f" @ `{self.revision}`" if self.revision else ""),
            f"- Seed: {self.seed}",
        ]
        if self.command:
            lines.append(f"- Reproduce: `{self.command}`")
        if self.artifact:
            lines.append(f"- Raw: `{self.artifact}`")
        return "\n".join(lines)


@dataclass(frozen=True)
class Sample:
    """One item in a golden dataset."""

    id: str
    inputs: dict[str, Any]
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stratum(self) -> str:
        """Slice label, used for per-stratum error analysis.

        A single aggregate score hides which segment regressed, so the harness
        treats stratification as a first-class property rather than an
        afterthought.
        """
        return str(self.metadata.get("stratum", "default"))


@dataclass(frozen=True)
class Judgement:
    """One judge's verdict on one candidate."""

    sample_id: str
    score: float                   # normalised to 0..1
    label: str | None = None
    rationale: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"score must be normalised to 0..1, got {self.score}. "
                "Judges with other ranges must normalise before returning."
            )


@dataclass(frozen=True)
class EvalResult:
    """The output of one evaluation run."""

    name: str
    judgements: list[Judgement]
    aggregate: dict[str, float]
    meta: RunMeta

    @property
    def scores(self) -> list[float]:
        return [j.score for j in self.judgements]

    def __len__(self) -> int:
        return len(self.judgements)
