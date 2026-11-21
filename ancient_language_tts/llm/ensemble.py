"""The model ensemble shared by `to_ipa` and `respell`, at five levels of effort.

`effort` decides which models write a line, whether each line is improved,
and who scores the lines that stand:

| effort   | writes                 | improves | scores                      |
|----------|------------------------|----------|-----------------------------|
| "ultra"  | every model of the tier | yes      | every model, cross-scored   |
| "high"   | every model of the tier | yes      | the first model only        |
| "medium" | every model of the tier | no       | the first model only        |
| "low"    | the first model         | yes      | nobody (one line)           |
| "basic"  | the first model         | no       | nobody (one line)           |

Improving a line: the author scores its own line, rewrites it from that
assessment, then scores the rewrite. The rewrite stands only when that
score is higher than the score of the first line.

A call is tried three times. The failure rule depends on the call. A model
that still cannot write the first line is dropped; at "low" and "basic"
the next model of the tier writes instead. A model that still cannot score
that line, or still cannot rewrite it, keeps the first line. At "ultra" a
model that still cannot score a standing line is left out of the mean. At
"high" and "medium" the next model of the tier scores instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from .. import config
from ..config import EFFORTS, Effort, Tier
from ..errors import TTSError

log = logging.getLogger("ancient_language_tts")

T = TypeVar("T")
# A line: a respelling (str) or an IPA token list (list[str]).
L = TypeVar("L")

# One call is tried this many times. The failure rule for that call then applies.
ATTEMPTS = 3


@dataclass(frozen=True)
class Plan:
    every_model: bool  # every model of the tier writes a line, or only the first
    improve: bool  # each author scores, rewrites, and rescores its line
    cross: bool  # every model scores every line, or only the first model


PLANS: dict[str, Plan] = {
    "ultra": Plan(every_model=True, improve=True, cross=True),
    "high": Plan(every_model=True, improve=True, cross=False),
    "medium": Plan(every_model=True, improve=False, cross=False),
    "low": Plan(every_model=False, improve=True, cross=False),
    "basic": Plan(every_model=False, improve=False, cross=False),
}


def check(effort: str) -> Effort:
    if effort not in PLANS:
        raise ValueError(f"unknown effort {effort!r}. Efforts: {', '.join(EFFORTS)}")
    return effort  # type: ignore[return-value]


def models(tier: Tier) -> list[str]:
    """The models asked: `models` when set, else this tier alone."""
    settings = config.current()
    if settings.models:
        names = list(settings.models)
    else:
        names = list(settings.tiers.get(tier, ()))
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique


def try_model(model: str, call: Callable[[], T], step: str) -> T | None:
    """Run `call` with `models` pinned to `model`. Give up on this call after `ATTEMPTS` failures."""
    last: BaseException | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            pinned = replace(config.current(), models=(model,))
            with config.use(pinned):
                return call()
        except TTSError as exc:
            last = exc
            log.info("%s %s attempt %s/%s failed: %s", model, step, attempt, ATTEMPTS, exc)
    log.info("%s %s failed after %s attempts (%s)", model, step, ATTEMPTS, last)
    return None


def show_mean(value: float | None) -> str:
    if value is None:
        return "unscored"
    text = f"{value:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def run(
    names: list[str],
    effort: Effort,
    *,
    draft: Callable[[str], L],
    judge: Callable[[str, L], dict[str, Any]],
    revise: Callable[[str, L, dict[str, Any]], L],
    label: str,
    noun: str,
    show: Callable[[L], str] = str,
) -> tuple[str, L, float | None]:
    """Return the winning model, its line, and that line's mean score.

    `draft(model)` writes a line. `judge(model, line)` returns a dict with
    `score` and `reasoning`. `revise(model, line, assessment)` rewrites a
    line from that assessment. `label`, `noun`, and `show` (a line as
    text) only shape the log. The
    mean is None when a single line stood and nobody scored it.
    """
    plan = PLANS[check(effort)]
    if not names:
        raise TTSError("No LLM models configured.")

    # `standing` is the line that stands. `own_score` is this model's score of
    # that line, when the call already succeeded. `pending` is a changed
    # rewrite waiting for the author's second score: the first line, and the
    # score that line received.
    standing: dict[str, L] = {}
    own_score: dict[str, int] = {}
    pending: dict[str, tuple[L, int]] = {}
    for name in names:
        first = try_model(name, lambda name=name: draft(name), noun)
        if first is None:
            log.info("dropping %s: the %s failed", name, noun)
            continue
        standing[name] = first
        if plan.improve:
            _improve(name, first, judge, revise, standing, own_score, pending, noun)
        if not plan.every_model:
            break

    # A model whose score of its own rewrite failed is left out of every mean.
    excluded: set[str] = set()
    for name, (first, initial_score) in pending.items():
        rewrite = standing[name]
        judged = try_model(name, lambda name=name, rewrite=rewrite: judge(name, rewrite), "self-score")
        if judged is None:
            log.info("%s score of the rewrite failed; its scores are left out of the mean", name)
            excluded.add(name)
            continue
        keep = "the rewrite" if judged["score"] > initial_score else f"the initial {noun}"
        log.info(
            "%s rewrite scored %s against the initial %s; keeping %s",
            name,
            judged["score"],
            initial_score,
            keep,
        )
        if judged["score"] <= initial_score:
            standing[name] = first
            own_score[name] = initial_score
        else:
            own_score[name] = judged["score"]

    if not standing:
        raise TTSError(f"Every model failed ({', '.join(names)}).")

    if not plan.cross and len(standing) == 1:
        winner = next(iter(standing))
        mean = own_score.get(winner)
        return winner, standing[winner], None if mean is None else float(mean)

    if plan.cross:
        judges, scores = _cross_scores(standing, own_score, excluded, judge)
    else:
        pool = [name for name in names if name not in excluded]
        judges, scores = _single_scores(standing, own_score, pool, judge)

    if not judges:
        raise TTSError(f"Every model failed ({', '.join(names)}).")

    means = {
        name: sum(scores[name][j] for j in judges) / len(judges)
        for name in standing
    }
    for name in standing:
        detail = ", ".join(f"{j} {scores[name][j]}" for j in judges)
        log.info(
            "%s candidate %s mean %s (%s) %s",
            label,
            name,
            show_mean(means[name]),
            detail,
            show(standing[name]),
        )
    winner = max(standing, key=lambda name: means[name])
    return winner, standing[winner], means[winner]


def _improve(
    name: str,
    first: L,
    judge: Callable[[str, L], dict[str, Any]],
    revise: Callable[[str, L, dict[str, Any]], L],
    standing: dict[str, L],
    own_score: dict[str, int],
    pending: dict[str, tuple[L, int]],
    noun: str,
) -> None:
    assessed = try_model(name, lambda: judge(name, first), "self-score")
    if assessed is None:
        log.info("%s keeps its initial %s: the assessment failed", name, noun)
        return
    revised = try_model(name, lambda: revise(name, first, assessed), "improve")
    if revised is None or revised == first:
        reason = "the rewrite failed" if revised is None else "the rewrite did not change it"
        log.info("%s keeps its initial %s: %s", name, noun, reason)
        own_score[name] = assessed["score"]
        return
    pending[name] = (first, assessed["score"])
    standing[name] = revised


def _cross_scores(
    standing: dict[str, L],
    own_score: dict[str, int],
    excluded: set[str],
    judge: Callable[[str, L], dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Every standing model scores every line. A judge that fails is left out of every mean."""
    scores: dict[str, dict[str, int]] = {name: {} for name in standing}
    judges = [name for name in standing if name not in excluded]
    for name, line in standing.items():
        for j in list(judges):
            if j == name and j in own_score:
                scores[name][j] = own_score[j]
                continue
            scored = try_model(j, lambda line=line, j=j: judge(j, line), "score")
            if scored is None:
                log.info("leaving %s out of the mean: scoring failed", j)
                judges.remove(j)
                for row in scores.values():
                    row.pop(j, None)
                continue
            scores[name][j] = scored["score"]
    return judges, scores


def _single_scores(
    standing: dict[str, L],
    own_score: dict[str, int],
    pool: list[str],
    judge: Callable[[str, L], dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """The first model of `pool` scores every line. If it fails, the next one does."""
    for j in pool:
        row: dict[str, int] = {}
        for name, line in standing.items():
            if j == name and j in own_score:
                row[name] = own_score[j]
                continue
            scored = try_model(j, lambda line=line, j=j: judge(j, line), "score")
            if scored is None:
                break
            row[name] = scored["score"]
        else:
            return [j], {name: {j: row[name]} for name in standing}
        log.info("%s could not score every line; the next model scores", j)
    return [], {}
