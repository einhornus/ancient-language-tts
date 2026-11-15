"""Step 1. Ancient-language text to IPA, and a score for a transcription.

`to_ipa` asks every model in the tier. Each model transcribes the passage,
scores its own line, and rewrites it from that assessment. The same model
then scores the rewrite. The rewrite stands only when that score is higher
than the score of the first line; otherwise the first line stands. Each
standing line is scored by every model still in the tier. The score of a
line is the arithmetic mean of those scores, and `to_ipa` returns the line
with the highest mean.

A call is tried three times. The failure rule depends on the call. A model
that still cannot produce the first transcription is dropped. A model that
still cannot score that line, or still cannot rewrite it, keeps the first
line. A model that still cannot score a standing line is left out of the
mean, and the mean is taken from the scores that remain.

The transcription system prompt is the phonetic rules of that language,
then worked examples of the same transformation. The model is told to learn
the decisions those examples make. It reasons in a fixed order, then writes
the line. A comma in that line is a short pause. An exclamation mark and a
question mark stay as written. A period is a long pause and the end of a
sentence.

`score_ipa` judges a transcription that already exists. The system prompt
is those same rules and examples, then four criteria in a fixed order.
The model reasons about each criterion from the one before it, then
scores them. The call returns that reasoning and a score from 0 to 1000.

`improve_ipa` scores a transcription, then rewrites it. The second call
sees the passage, the original IPA, the scoring reasoning, and the
phonetic rules. It reasons from the mistakes to the corrected sounds,
then writes the line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from . import chunks, config, languages, llm, rules
from .config import Tier
from .errors import TTSError
from .languages import AncientLanguage
from .schemas import IpaRevision, IpaScore, IpaTranscription

log = logging.getLogger("ancient_language_tts")

T = TypeVar("T")

# Values are the sounds. Syllables and stress are how those sounds are
# grouped and marked. The line is the writing. `score` is this weighted
# mean, scaled from 0-100 to 0-1000.
WEIGHTS = {"values": 5.0, "syllables": 1.0, "stress": 2.0, "line": 1.0}

# One call is tried this many times. The failure rule for that call then applies.
ATTEMPTS = 3


def system_prompt(source: str) -> str:
    language = languages.ancient(source)
    return (
        f"You transcribe {language.name} into the International Phonetic Alphabet.\n"
        "You do not translate or modernize the passage. The ipa field is the "
        "transcription and nothing else.\n"
        "Where a rule below gives a value, use it even when a more familiar "
        "reading of the same letters disagrees.\n\n"
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"{rules.phonology(language.code)}\n\n"
        "Examples\n"
        "Each line is a passage, then →, then the IPA these rules require. "
        "Learn from them. They show the rules applied: which reading a letter "
        "takes, which length is restored, where a syllable breaks, where the "
        "stress falls, which pause mark is written, and how the line is "
        "written. When the passage presents the same choice, follow the "
        "decision the examples make.\n"
        f"{rules.examples(language.code)}\n\n"
        "First, reason carefully about these things, in this order. Each "
        "step uses the result of the step before it.\n"
        "1. Values. Give each word's consonants and vowels from the rules. "
        "Where a letter has two readings, apply the rule that picks one, "
        "and apply the choice the examples make for that letter.\n"
        "2. Syllables. Split those values into syllables by the language's "
        "syllable rules, the way the examples split them.\n"
        "3. Stress. Place ˈ on those syllables by the language's stress or "
        "accent rule, and name the fact about the syllables that puts it there.\n"
        "Then write the IPA line from the stressed syllables of step 3. "
        "The pause marks are a comma, an exclamation mark, a question mark, "
        "and a period. A comma is a short pause. An exclamation mark stays "
        "an exclamation mark. A question mark stays a question mark. A "
        "period is a long pause and the end of a sentence. Where the passage "
        "has a semicolon or a colon, write a period."
    )


def user_prompt(text: str, language: AncientLanguage) -> str:
    return f"Language: {language.name} ({language.code}), {language.era}\n\nText:\n{text}"


def to_ipa(text: str, source: str, *, tier: Tier = "high", **settings: Any) -> str:
    """Transcribe `text` into IPA and return the line with the highest mean score.

    Every model of `tier` transcribes the passage, scores its own line, and
    rewrites it from that assessment. The same model then scores the rewrite.
    The rewrite stands only when that score is higher than the score of the
    first line; otherwise the first line stands. Each standing line is scored
    by every model still in the tier. The score of a line is the arithmetic
    mean of those scores. On a tie, the earlier model in the tier wins.

    A call is tried three times. A model that still cannot produce the first
    transcription is dropped. One that still cannot score that line, or still
    cannot rewrite it, keeps the first line. One that still cannot score a
    standing line is left out of the mean. When `models` is set, that list is
    the tier. Settings are the keyword arguments of `speak`.

    A passage longer than `max_chars` is split on sentence boundaries, and
    each piece is transcribed on its own. The lines are joined in order.
    `speak` respells each of those pieces before joining them.
    """
    with config.scope(**settings):
        parts = chunks.split(text)
        if len(parts) <= 1:
            return _ensemble(parts[0] if parts else text, source, tier)
        log.info(
            "splitting %s characters into %s pieces (IPA limit %s)",
            len(text.strip()),
            len(parts),
            config.current().max_chars,
        )
        lines: list[str] = []
        for index, part in enumerate(parts, start=1):
            log.info("piece %s/%s, %s characters", index, len(parts), len(part))
            lines.append(_ensemble(part, source, tier))
        return " ".join(lines)


def _models(tier: Tier) -> list[str]:
    """The models `to_ipa` asks: `models` when set, else this tier alone."""
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


def _try_model(model: str, call: Callable[[], T], step: str) -> T | None:
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


def _show_mean(value: float) -> str:
    text = f"{value:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def _ensemble(text: str, source: str, tier: Tier) -> str:
    language = languages.ancient(source)
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text is empty")
    llm.require_key()
    names = _models(tier)
    if not names:
        raise TTSError("No LLM models configured.")

    # `standing` is the line that stands. `own_score` is this model's score of
    # that line, when the call already succeeded. `pending` is a changed
    # rewrite waiting for the author's second score: the first line, and the
    # score that line received.
    standing: dict[str, str] = {}
    own_score: dict[str, int] = {}
    pending: dict[str, tuple[str, int]] = {}
    for name in names:
        draft = _try_model(name, lambda name=name: _draft(cleaned, language, name), "transcription")
        if draft is None:
            log.info("dropping %s: the transcription failed", name)
            continue
        assessed = _try_model(
            name,
            lambda name=name, draft=draft: _judge(cleaned, draft, language.code, name),
            "self-score",
        )
        if assessed is None:
            log.info("%s keeps its initial IPA: the assessment failed", name)
            standing[name] = draft
            continue
        revised = _try_model(
            name,
            lambda name=name, draft=draft, assessed=assessed: _revise(
                cleaned,
                draft,
                language.code,
                name,
                assessed["reasoning"],
                assessed["score"],
            ),
            "improve",
        )
        if revised is None or revised["ipa"] == draft:
            reason = "the rewrite failed" if revised is None else "the rewrite did not change it"
            log.info("%s keeps its initial IPA: %s", name, reason)
            standing[name] = draft
            own_score[name] = assessed["score"]
            continue
        pending[name] = (draft, assessed["score"])
        standing[name] = revised["ipa"]

    # A model whose score of a standing line failed is left out of every mean.
    excluded: set[str] = set()
    for name, (draft, initial_score) in pending.items():
        rewrite = standing[name]
        judged = _try_model(
            name,
            lambda name=name, rewrite=rewrite: _judge(cleaned, rewrite, language.code, name),
            "self-score",
        )
        if judged is None:
            log.info("%s score of the rewrite failed; its scores are left out of the mean", name)
            excluded.add(name)
            continue
        if judged["score"] <= initial_score:
            log.info(
                "%s rewrite scored %s against the initial %s; keeping the initial IPA",
                name,
                judged["score"],
                initial_score,
            )
            standing[name] = draft
            own_score[name] = initial_score
            continue
        log.info(
            "%s rewrite scored %s against the initial %s; keeping the rewrite",
            name,
            judged["score"],
            initial_score,
        )
        own_score[name] = judged["score"]

    if not standing:
        raise TTSError(f"Every model failed ({', '.join(names)}).")

    scores: dict[str, dict[str, int]] = {name: {} for name in standing}
    judges = [name for name in standing if name not in excluded]
    for name in standing:
        line = standing[name]
        for judge in list(judges):
            if judge == name and judge in own_score:
                scores[name][judge] = own_score[judge]
                continue
            scored = _try_model(
                judge,
                lambda line=line, judge=judge: _judge(cleaned, line, language.code, judge),
                "score",
            )
            if scored is None:
                log.info("leaving %s out of the mean: scoring failed", judge)
                judges.remove(judge)
                for row in scores.values():
                    row.pop(judge, None)
                continue
            scores[name][judge] = scored["score"]

    if not judges:
        raise TTSError(f"Every model failed ({', '.join(names)}).")

    means = {
        name: sum(scores[name][judge] for judge in judges) / len(judges)
        for name in standing
    }
    for name in standing:
        detail = ", ".join(f"{judge} {scores[name][judge]}" for judge in judges)
        log.info(
            "%s candidate %s mean %s (%s) %s",
            language.code,
            name,
            _show_mean(means[name]),
            detail,
            standing[name],
        )
    winner = max(standing, key=lambda name: means[name])
    log.info(
        "%s IPA: %s (%s, mean %s)",
        language.code,
        standing[winner],
        winner,
        _show_mean(means[winner]),
    )
    return standing[winner]


def _draft(text: str, language: AncientLanguage, model: str) -> str:
    result = llm.ask(IpaTranscription, system_prompt(language.code), user_prompt(text, language))
    ipa = _clean(result.ipa)
    if not ipa:
        raise TTSError("The model returned an empty transcription.")
    log.info("%s IPA draft: %s (%s)", language.code, ipa, model)
    return ipa


def score_system_prompt(source: str) -> str:
    language = languages.ancient(source)
    return (
        f"Your task is to judge one IPA transcription of {language.name}, "
        "exactly as written.\n"
        "The phonetic rules and the worked examples are what a correct "
        "transcription does. Where a rule or an example gives a value, "
        "that value is the right one, even when a more familiar reading of "
        "the same letters disagrees.\n"
        "You are scoring this transcription. A new transcription is not "
        "your answer.\n\n"
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"{rules.phonology(language.code)}\n\n"
        "Examples\n"
        "Each line is a passage, then →, then the IPA these rules require. "
        "They show the rules applied: which reading a letter takes, which "
        "length is restored, where a syllable breaks, where the stress "
        "falls, which pause mark is written, and how the line is written. "
        "When the passage presents the same choice, the examples are what "
        "a correct transcription does.\n"
        f"{rules.examples(language.code)}\n\n"
        "Score four criteria, 0-100 each, reasoning about each in this "
        "order. Each step uses the result of the step before it.\n\n"
        "values — are the consonants and vowels the ones the rules assign? "
        "Reason: each word's consonants and vowels in this transcription; "
        "where a letter has two readings, whether it takes the reading the "
        "rule picks and the choice the examples make. A segment the rules "
        "forbid drops it below 30.\n\n"
        "syllables — do the breaks follow the syllable rules, the way the "
        "examples split them? Reason: from the values already judged, where "
        "the dots fall, and whether that split is the one the rules "
        "require. A break the rules forbid drops it below 30.\n\n"
        "stress — does ˈ fall where the stress or accent rule puts it? "
        "Reason: from those syllables, the syllable this transcription "
        "marks and the fact about the syllables that should put ˈ there, "
        "and ˌ where the rules write one. Stress on the wrong syllable "
        "drops it below 30.\n\n"
        "line — is the line written the way the output rules and the "
        "examples write one? Reason: spaces, dots, stress marks, lengths, "
        "and pause marks; whether the words match the text; whether "
        "anything other than the transcription is present. A comma is a "
        "short pause. An exclamation mark and a question mark stay as they "
        "are. A period is a long pause and the end of a sentence. A "
        "semicolon or a colon in the passage is a period in the line. The "
        "wrong pause mark, or a mark other than these four, drops it below "
        "30. A missing word or an added word drops it below 30.\n\n"
        "First, reason carefully about each criterion. Then score them and "
        "give a one-sentence verdict. Be strict: reserve scores above 80 "
        "for a transcription you would accept without hesitation."
    )


def score_user_prompt(text: str, ipa: str, language: AncientLanguage) -> str:
    return (
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"Text:\n{text}\n\n"
        f"IPA:\n{ipa}\n\n"
        "After your analysis, score this transcription."
    )


def score_ipa(text: str, ipa: str, lang: str, model: str, **settings: Any) -> dict[str, Any]:
    """Score how well `ipa` renders `text` in `lang`.

    `model` is one OpenRouter id, and it is the only model asked. Settings
    are the keyword arguments of `speak`. The dict has `reasoning` and
    `score`. `score` runs from 0 to 1000.
    """
    cleaned_text = text.strip()
    cleaned_ipa = ipa.strip()
    name = model.strip()
    if not cleaned_text:
        raise ValueError("text is empty")
    if not cleaned_ipa:
        raise ValueError("ipa is empty")
    if not name:
        raise ValueError("model is empty")
    settings = dict(settings)
    settings["models"] = name
    with config.scope(**settings):
        return _judge(cleaned_text, cleaned_ipa, lang, name)


def _total(scores: dict[str, int]) -> int:
    """Weighted mean of the 0-100 criteria, scaled to 0-1000."""
    weight = sum(WEIGHTS.values())
    mean = sum(scores[name] * WEIGHTS[name] for name in WEIGHTS) / weight
    return round(mean * 10)


def _reasoning(result: IpaScore) -> str:
    steps = (
        ("Values", result.values),
        ("Syllables", result.syllables),
        ("Stress", result.stress),
        ("Line", result.line),
        ("Verdict", result.verdict),
    )
    return "\n\n".join(f"{title}\n{body.strip()}" for title, body in steps)


def _judge(text: str, ipa: str, lang: str, model: str) -> dict[str, Any]:
    language = languages.ancient(lang)
    result = llm.ask(
        IpaScore,
        score_system_prompt(language.code),
        score_user_prompt(text, ipa, language),
    )
    scores = {name: getattr(result, f"{name}_score") for name in WEIGHTS}
    score = _total(scores)
    log.info("%s IPA score: %s (%s)", language.code, score, model)
    return {"reasoning": _reasoning(result), "score": score}


def improve_system_prompt(source: str) -> str:
    language = languages.ancient(source)
    return (
        f"You correct an IPA transcription of {language.name}.\n"
        "You are given the passage, a transcription of it, and an assessment "
        "of that transcription. Rewrite the transcription so it follows the "
        "phonetic rules. Keep every part the assessment finds correct. "
        "Change a part only to repair a mistake the rules confirm.\n"
        "Where a rule below gives a value, use it even when a more familiar "
        "reading of the same letters disagrees.\n"
        "You do not translate or modernize the passage. The ipa field is the "
        "corrected transcription and nothing else.\n\n"
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"{rules.phonology(language.code)}\n\n"
        "Examples\n"
        "Each line is a passage, then →, then the IPA these rules require. "
        "Learn from them. They show the rules applied: which reading a letter "
        "takes, which length is restored, where a syllable breaks, where the "
        "stress falls, which pause mark is written, and how the line is "
        "written. When the passage presents the same choice, follow the "
        "decision the examples make.\n"
        f"{rules.examples(language.code)}\n\n"
        "First, reason carefully about these things, in this order. Each "
        "step uses the result of the step before it.\n"
        "1. Faults. Take each mistake the assessment names and check it "
        "against the rules. Keep the parts the assessment leaves standing.\n"
        "2. Values. From those faults, give each word's consonants and vowels "
        "from the rules. Where a letter has two readings, apply the rule that "
        "picks one, and apply the choice the examples make for that letter.\n"
        "3. Syllables. Split those values into syllables by the language's "
        "syllable rules, the way the examples split them.\n"
        "4. Stress. Place ˈ on those syllables by the language's stress or "
        "accent rule, and name the fact about the syllables that puts it there.\n"
        "Then write the IPA line from the stressed syllables of step 4. "
        "The pause marks are a comma, an exclamation mark, a question mark, "
        "and a period. A comma is a short pause. An exclamation mark stays "
        "an exclamation mark. A question mark stays a question mark. A "
        "period is a long pause and the end of a sentence. Where the passage "
        "has a semicolon or a colon, write a period."
    )


def improve_user_prompt(text: str, ipa: str, language: AncientLanguage, assessment: str, score: int) -> str:
    return (
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"Text:\n{text}\n\n"
        f"IPA:\n{ipa}\n\n"
        f"Assessment ({score} out of 1000):\n{assessment}\n\n"
        "After your analysis, write the corrected IPA."
    )


def improve_ipa(text: str, ipa: str, lang: str, model: str, **settings: Any) -> dict[str, Any]:
    """Rewrite `ipa` after scoring it.

    The call asks `score_ipa` first. The rewrite sees that reasoning, the
    passage, the original IPA, and the language's phonetic rules. `model`
    is the only OpenRouter id asked. The dict has `ipa`, `reasoning`,
    `score`, and `assessment`.
    """
    scored = score_ipa(text, ipa, lang, model, **settings)
    name = model.strip()
    passed = dict(settings)
    passed["models"] = name
    with config.scope(**passed):
        revised = _revise(text.strip(), ipa.strip(), lang, name, scored["reasoning"], scored["score"])
    revised["score"] = scored["score"]
    revised["assessment"] = scored["reasoning"]
    return revised


def _revision_reasoning(result: IpaRevision) -> str:
    steps = (
        ("Faults", result.faults),
        ("Values", result.values),
        ("Syllables", result.syllables),
        ("Stress", result.stress),
    )
    return "\n\n".join(f"{title}\n{body.strip()}" for title, body in steps)


def _revise(text: str, ipa: str, lang: str, model: str, assessment: str, score: int) -> dict[str, Any]:
    language = languages.ancient(lang)
    result = llm.ask(
        IpaRevision,
        improve_system_prompt(language.code),
        improve_user_prompt(text, ipa, language, assessment, score),
    )
    corrected = _clean(result.ipa)
    if not corrected:
        raise TTSError("The model returned an empty correction.")
    log.info("%s IPA revised: %s (%s)", language.code, corrected, model)
    return {"ipa": corrected, "reasoning": _revision_reasoning(result)}


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "/[]":
        inner = text[1:-1].strip()
        if inner:
            text = inner
    text = text.lstrip("*").strip()
    return " ".join(text.split())
