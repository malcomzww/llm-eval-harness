# llm-eval-harness

An LLM-as-judge harness built around one question:

> **How far can an LLM judge be trusted, measured against human labels?**

The answer is a number — Cohen's kappa — and this library's position is that
you should not report anything a judge produces without it. An uncalibrated
judge is a random number generator with good manners.

```
$ python -m pytest -q
48 passed

$ python scripts/generate_results.py
  null delta   -0.0100 [-0.1000, +0.0850] not reportable
  real delta   +0.1700 [+0.0850, +0.2600] reportable
  kappa good=0.710 coin=0.008
```

## The rule this enforces

Two systems drawing from the *same* distribution — genuinely identical, true
difference exactly zero:

| | |
|---|---|
| observed delta | **−0.0100** |
| 95% CI | [−0.1000, +0.0850] |
| reportable | **no** — interval includes zero |

The point estimate is not zero and never will be. Quoting it as an improvement
is the most common way eval results mislead, so the harness refuses to.

Two systems that genuinely differ by 10 points, same sample size:

| | |
|---|---|
| observed delta | **+0.1700** |
| 95% CI | [+0.0850, +0.2600] |
| reportable | **yes** — interval excludes zero |

Full generated output: [`results/reporting-rule.md`](results/reporting-rule.md).

## Why kappa rather than accuracy

| judge | agreement with human | Cohen's kappa |
|---|---|---|
| 85%-agreeing | 86% | **0.710** |
| coin flip | 50% | **0.008** |

Raw agreement makes the coin flip look like it is doing something. If 90% of
your samples are "good", a judge that always says "good" agrees 90% of the
time while carrying no information at all. Kappa subtracts the agreement
expected by chance, and that judge scores zero.

So `calibrate()` is the function this package exists for: run an LLM judge and
a human-label judge over the same samples, and report the agreement. That is
the step that turns a judge from an assumption into an instrument with a known
error rate.

## How big does a golden set need to be?

Samples per arm to detect a given difference at 80% power, α=0.05, sd=0.5:

| difference to detect | samples per arm |
|---|---|
| 1% | 39,245 |
| 2% | 9,812 |
| 5% | 1,570 |
| 10% | 393 |
| 20% | 99 |

Worth computing before building the set rather than after. A 100-item golden
set cannot resolve a 2-point difference, and no amount of careful judging
changes that.

## Quickstart

```python
from llm_eval_harness.runner import evaluate, calibrate
from llm_eval_harness.judge import ExactMatchJudge, HumanLabelJudge
from llm_eval_harness.types import Sample, RunMeta

meta = RunMeta.now(hardware="24-core CPU, no GPU", model="gpt-4o-mini", seed=0)
result = await evaluate(dataset, candidates, judge, name="v2", meta=meta)

# The number that belongs next to every score the judge produced
agreement = await calibrate(dataset, candidates, judge, HumanLabelJudge(labels))
print(agreement.kappa, agreement.interpretation)   # 0.71 substantial
```

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/generate_results.py
```

## Design decisions

**`stats` is pure and synchronous.** No I/O, no async, no LLM machinery. Most
repos that depend on this harness need exactly `bootstrap_ci`,
`paired_delta_ci` and `cohen_kappa` and nothing else, so those work offline, in
CI, and in a unit test without a single mock. The inverse-normal CDF is
hand-rolled for the same reason — pulling in scipy for one quantile would make
every downstream repo carry it.

**`evaluate()` always returns a confidence interval and per-stratum means.**
Not optional flags. A mean with no interval invites a comparison it cannot
support, and a headline average hides the slice that regressed.

**`calibrate()` returns agreement, not pass/fail.** What counts as an
acceptable kappa depends on what the eval gates, and that judgement stays with
the caller.

**`Judge` is a Protocol, not a base class.** Downstream repos need
domain-specific judges — a SQL-execution checker, a field-level extraction
comparator — and should not have to inherit from this package to be usable by
`evaluate`.

**Committed output is rounded to 4 decimals.** Not cosmetic. Unrounded floats
differ in the last bit across platforms, which makes a committed results file
unstable and defeats the CI drift gate. This repo's own test suite failed on
exactly that — passing on Windows, failing on the Linux runner over
`0.626` vs `0.6260000000000001`.

## Limitations

- **The demonstration uses synthetic Bernoulli data**, stated in the output.
  The claim under test is that the harness refuses to report what it cannot
  support; real model noise would obscure that rather than demonstrate it.
  It does **not** establish behaviour on real skewed or bimodal score
  distributions.
- **`LLMJudge.score()` is not implemented.** Its API is frozen so eleven
  downstream repos can build against it, but there is no API key configured
  here and a judge that has never called a model is not one I will claim
  works. `ExactMatchJudge` and `HumanLabelJudge` are complete and tested.
- Kappa here is **binary**. Multi-category and weighted kappa behave
  differently and are not established.
- Sample sizing uses a normal approximation — loose at very small n or
  extreme proportions.

## Status

| module | state |
|---|---|
| `stats.py` — kappa, bootstrap, paired deltas, power | **done** |
| `types.py` — Sample/Judgement/EvalResult/RunMeta | **done** |
| `runner.py` — evaluate, calibrate | **done** |
| `report.py` — markdown rendering | **done** |
| `judge.py` — ExactMatch, HumanLabel | **done** |
| `judge.py` — LLMJudge, Rubric.render | API frozen, not implemented |
| golden-set curation CLI | not started |

## License

MIT
