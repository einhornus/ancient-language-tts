"""Step 2. IPA to a spelling a modern voice will read, and a score for one.

`respell` asks every model in the tier. Each model respells the IPA,
scores its own line, and rewrites it from that assessment. The same model
then scores the rewrite. The rewrite stands only when that score is higher
than the score of the first line; otherwise the first line stands. Each
standing line is scored by every model still in the tier. The score of a
line is the arithmetic mean of those scores, and `respell` returns the line
with the highest mean.

A call is tried three times. The failure rule depends on the call. A model
that still cannot produce the first respelling is dropped. A model that
still cannot score that line, or still cannot rewrite it, keeps the first
line. A model that still cannot score a standing line is left out of the
mean, and the mean is taken from the scores that remain.

The respelling system prompt is how the modern spelling is read, which of
those spellings stands in for each sound of the source, and worked
examples of that mapping. Each example is the original text, then the IPA,
then the spelling. The model applies those rules to the IPA, works the
hardest words, then writes the line in the stated shape.

`score_respelling` judges a respelling that already exists. The system
prompt is those same rules and examples, then three criteria in a fixed
order: how close the sounds a modern voice would make are to the IPA, how
accurate the stress is, and how closely the line follows the stated shape.
The model reasons about each criterion from the one before it, then scores
them. The call returns that reasoning and a score from 0 to 1000.

`improve_respelling` scores a respelling, then rewrites it. The second
call sees the passage, the IPA, the original respelling, the scoring
reasoning, and the conversion rules. It reasons through those same three
criteria, from the mistakes to the corrected line, then writes the line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from .. import config, languages, rules
from ..config import Tier
from ..errors import TTSError
from ..languages import AncientLanguage, ModernLanguage
from . import client as llm
from .schemas import RespelledText, RespellRevision, RespellScore

log = logging.getLogger("ancient_language_tts")

T = TypeVar("T")

# Phonetics is how close the voice's sounds are to the IPA. Stress is which
# syllable the voice marks. Format is the shape in LINE_FORMAT. `score` is
# this weighted mean, scaled from 0-100 to 0-1000.
WEIGHTS = {"phonetics": 5.0, "stress": 3.0, "format": 5.0}

# One call is tried this many times. The failure rule for that call then applies.
ATTEMPTS = 3

# Shown to the model. "Respelling" makes it write a pronunciation key:
# hyphens between syllables, the stressed syllable in capitals.
LINE_FORMAT = (
    "How the line is written\n"
    "The line is ordinary text in the target spelling. It should look like "
    "a normal sentence of that language.\n"
    "A word is one unbroken string of letters. A space separates words. "
    "Do not split a word into syllables, and do not put a hyphen inside a word.\n"
    "Show stress with the accent the spelling rules place on the stressed "
    "vowel. Do not capitalize a syllable to show stress, and do not write "
    "the stressed syllable in capitals.\n"
    "Write lowercase letters. Capitalize only the first letter of a sentence. "
    "The accent stays on the stressed vowel: Gàllia èst òmnis, not "
    "GÀL-lia ÈST ÒM-nis.\n"
    "Keep a period, a comma, a question mark, or an exclamation mark that "
    "the IPA uses as a pause. No other punctuation. No IPA symbols, no "
    "notes, and no markdown.\n"
    "This shape wins over any later instruction to hyphenate a syllable or "
    "to capitalize a stressed syllable. Copy from the examples the spelling "
    "of each sound and the place of each accent. Do not copy a hyphen or a "
    "capital syllable from an example."
)

# A respelling that still contains these was not finished. The voice would
# read the symbols as their letter names, or skip them. We keep the text
# and warn: the caller can see it and decide.
_IPA_MARKS = set("ˈˌəɪʊæɑɔɛɨɵøœʌθðʃʒŋɣχʔɸβ̥̩̯͡ʰʷʲːʱẽõ")


def _examples_block(source: str, target: str) -> str:
    examples = rules.pair_examples(source, target)
    if not examples:
        return ""
    return (
        "\n\nExamples\n"
        "Each line is the original text, then →, then its IPA, then →, then "
        "the spelling these rules require. Learn the spelling from the part "
        "after the second arrow. The original shows which words the IPA "
        "transcribes. Do not copy that spelling. They show the rules "
        "applied: which spelling a sound takes, which accent a stressed "
        "vowel takes, and how the whole word is written. When the IPA "
        "presents the same choice, follow the decision the examples make. "
        "Write the line in the shape above, even where an example still "
        "shows a hyphen or a capital syllable.\n"
        f"{examples}"
    )


def _examples_score_block(source: str, target: str) -> str:
    examples = rules.pair_examples(source, target)
    if not examples:
        return ""
    return (
        "\n\nExamples\n"
        "Each line is the original text, then →, then its IPA, then →, then "
        "the spelling these rules require. Judge against the spelling after "
        "the second arrow. The original shows which words the IPA "
        "transcribes. They show the rules applied: which spelling a sound "
        "takes, which accent a stressed vowel takes, and how the whole word "
        "is written. When the IPA presents the same choice, the examples are "
        "what a correct line does. A hyphen or a capital syllable in an "
        "example is not part of that shape.\n"
        f"{examples}"
    )


def system_prompt(lang: str, phonetics_lang: str) -> str:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    return (
        f"You turn an IPA transcription of {ancient.name} into spelling a "
        f"{modern.name} text-to-speech voice can read aloud.\n"
        "You do not translate. You do not restore the original spelling. "
        f"The voice reads your line with ordinary {modern.name} letter-to-sound "
        "rules and will not see the IPA.\n"
        "If a pair rule and the general spelling rules disagree, follow the pair rule.\n\n"
        f"How {modern.name} spelling is read\n"
        f"{rules.orthography(modern.code)}\n\n"
        f"How {ancient.name} maps onto that spelling\n"
        f"{rules.pair(ancient.code, modern.code)}"
        f"{_examples_block(ancient.code, modern.code)}\n\n"
        "Reason in this order. Each step uses the result of the step before it.\n"
        "1. Rules. Apply these rules to this IPA: which spelling each sound "
        "takes, where stress is marked, and which letters the rules allow.\n"
        "2. Hard words. Find the words those rules are hardest to apply to, "
        "and for each one work out the spelling the rules require.\n"
        "Then write the line in this shape.\n\n"
        f"{LINE_FORMAT}"
    )


def user_prompt(text: str, ipa: str, ancient: AncientLanguage, modern: ModernLanguage) -> str:
    source_text = text.strip() if text.strip() else "(not given)"
    return (
        f"Source: {ancient.name} ({ancient.code})\n"
        f"Target voice: {modern.name} ({modern.code})\n\n"
        "Original text, for word boundaries only. Do not copy this spelling:\n"
        f"{source_text}\n\n"
        f"IPA:\n{ipa}"
    )


def respell(
    text: str,
    ipa: str,
    source: str,
    target: str = "en",
    *,
    tier: Tier = "high",
    **settings: Any,
) -> str:
    """Respell `ipa` into `target` and return the line with the highest mean score.

    `text` is the source passage, used for word boundaries only. `source`
    is the ancient-language code the IPA came from.

    Every model of `tier` respells the IPA, scores its own line, and
    rewrites it from that assessment. The same model then scores the rewrite.
    The rewrite stands only when that score is higher than the score of the
    first line; otherwise the first line stands. Each standing line is scored
    by every model still in the tier. The score of a line is the arithmetic
    mean of those scores. On a tie, the earlier model in the tier wins.

    A call is tried three times. A model that still cannot produce the first
    respelling is dropped. One that still cannot score that line, or still
    cannot rewrite it, keeps the first line. One that still cannot score a
    standing line is left out of the mean. When `models` is set, that list is
    the tier. Settings are the keyword arguments of `speak`.
    """
    with config.scope(**settings):
        return _ensemble(text, ipa, source, target, tier)


def _models(tier: Tier) -> list[str]:
    """The models `respell` asks: `models` when set, else this tier alone."""
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


def _ensemble(text: str, ipa: str, lang: str, phonetics_lang: str, tier: Tier) -> str:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    cleaned_text = text.strip()
    cleaned_ipa = _clean(ipa)
    if not cleaned_ipa:
        raise ValueError("ipa is empty")
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
        draft = _try_model(
            name,
            lambda name=name: _draft(cleaned_text, cleaned_ipa, ancient, modern, name),
            "respelling",
        )
        if draft is None:
            log.info("dropping %s: the respelling failed", name)
            continue
        assessed = _try_model(
            name,
            lambda name=name, draft=draft: _judge(
                cleaned_text, cleaned_ipa, draft, ancient.code, modern.code, name
            ),
            "self-score",
        )
        if assessed is None:
            log.info("%s keeps its initial respelling: the assessment failed", name)
            standing[name] = draft
            continue
        revised = _try_model(
            name,
            lambda name=name, draft=draft, assessed=assessed: _revise(
                cleaned_text,
                cleaned_ipa,
                draft,
                ancient.code,
                modern.code,
                name,
                assessed["reasoning"],
                assessed["score"],
            ),
            "improve",
        )
        if revised is None or revised["text"] == draft:
            reason = "the rewrite failed" if revised is None else "the rewrite did not change it"
            log.info("%s keeps its initial respelling: %s", name, reason)
            standing[name] = draft
            own_score[name] = assessed["score"]
            continue
        pending[name] = (draft, assessed["score"])
        standing[name] = revised["text"]

    # A model whose score of a standing line failed is left out of every mean.
    excluded: set[str] = set()
    for name, (draft, initial_score) in pending.items():
        rewrite = standing[name]
        judged = _try_model(
            name,
            lambda name=name, rewrite=rewrite: _judge(
                cleaned_text, cleaned_ipa, rewrite, ancient.code, modern.code, name
            ),
            "self-score",
        )
        if judged is None:
            log.info("%s score of the rewrite failed; its scores are left out of the mean", name)
            excluded.add(name)
            continue
        if judged["score"] <= initial_score:
            log.info(
                "%s rewrite scored %s against the initial %s; keeping the initial respelling",
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
                lambda line=line, judge=judge: _judge(
                    cleaned_text, cleaned_ipa, line, ancient.code, modern.code, judge
                ),
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
            "%s → %s candidate %s mean %s (%s) %s",
            ancient.code,
            modern.code,
            name,
            _show_mean(means[name]),
            detail,
            standing[name],
        )
    winner = max(standing, key=lambda name: means[name])
    text = standing[winner]
    leftover = sorted({ch for ch in text if ch in _IPA_MARKS})
    if leftover:
        log.warning("respelling still has IPA symbols (%s): %s", "".join(leftover), text)
    log.info(
        "%s → %s: %s (%s, mean %s)",
        ancient.code,
        modern.code,
        text,
        winner,
        _show_mean(means[winner]),
    )
    return text


def _draft(
    text: str,
    ipa: str,
    ancient: AncientLanguage,
    modern: ModernLanguage,
    model: str,
) -> str:
    result = llm.ask(
        RespelledText,
        system_prompt(ancient.code, modern.code),
        user_prompt(text, ipa, ancient, modern),
    )
    spelled = _clean(result.text)
    if not spelled:
        raise TTSError("The model returned an empty respelling.")
    log.info("%s → %s draft: %s (%s)", ancient.code, modern.code, spelled, model)
    return spelled


def score_system_prompt(lang: str, phonetics_lang: str) -> str:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    return (
        f"Your task is to judge one spelling of {ancient.name} IPA for a "
        f"{modern.name} text-to-speech voice, exactly as written.\n"
        "The pair rules, the spelling rules, and the worked examples are "
        "what a correct line does. Where a rule or an example gives "
        "a spelling, that spelling is the right one, even when a more "
        "familiar reading of the same sounds disagrees.\n"
        "You are scoring this line. A new line is not your answer.\n\n"
        f"{LINE_FORMAT}\n\n"
        f"Source: {ancient.name} ({ancient.code})\n"
        f"Target voice: {modern.name} ({modern.code})\n\n"
        f"How {modern.name} spelling is read\n"
        f"{rules.orthography(modern.code)}\n\n"
        f"How {ancient.name} maps onto that spelling\n"
        f"{rules.pair(ancient.code, modern.code)}"
        f"{_examples_score_block(ancient.code, modern.code)}\n\n"
        "Score three criteria, 0-100 each, reasoning about each in this "
        "order. Each step uses the result of the step before it.\n\n"
        f"phonetics — how close are the sounds a {modern.name} voice would "
        "make from this line to the IPA? Read the line with the spelling "
        "rules above. The voice will not see the IPA. Reason: for each "
        "sound, the sound the voice would make and the sound the IPA has, "
        "including the substitute the pair rules name when the target "
        "language has no letter for a sound. That substitute is the closest "
        "the voice can come, and it counts as accurate. A sound the voice "
        "would make that the IPA does not have, or a sound in the IPA the "
        "voice would miss, drops it below 30.\n\n"
        "stress — how accurate is the stress? Reason: from the sounds "
        "already judged, for each word, the syllable this line marks and "
        "the syllable the IPA stress mark selects. The accent the spelling "
        "rules place on the stressed vowel is the mark. Stress on the wrong "
        "syllable drops it below 30.\n\n"
        "format — how closely does the line follow the shape stated above? "
        "Reason: from that stress, each word unbroken, spaces between "
        "words, an accent on the stressed vowel, lowercase letters, a "
        "capital only at the start of a sentence, and pause marks; whether "
        "the words match the IPA; whether anything other than the line is "
        "present. A hyphen inside a word, a capital used to mark stress, "
        "leftover IPA, a missing word, or an added word drops it below 30.\n\n"
        "First, reason carefully about each criterion. Then score them and "
        "give a one-sentence verdict. Be strict: reserve scores above 80 "
        "for a line you would accept without hesitation."
    )


def score_user_prompt(
    text: str,
    ipa: str,
    respelling: str,
    ancient: AncientLanguage,
    modern: ModernLanguage,
) -> str:
    source_text = text.strip() if text.strip() else "(not given)"
    return (
        f"Source: {ancient.name} ({ancient.code})\n"
        f"Target voice: {modern.name} ({modern.code})\n\n"
        "Original text, for word boundaries only:\n"
        f"{source_text}\n\n"
        f"IPA:\n{ipa}\n\n"
        f"Line:\n{respelling}\n\n"
        "After your analysis, score this line."
    )


def score_respelling(
    text: str,
    ipa: str,
    respelling: str,
    lang: str,
    phonetics_lang: str,
    model: str,
    **settings: Any,
) -> dict[str, Any]:
    """Score how well `respelling` renders `ipa` for `phonetics_lang`.

    `model` is one OpenRouter id, and it is the only model asked. Settings
    are the keyword arguments of `speak`. The dict has `reasoning` and
    `score`. `score` runs from 0 to 1000.
    """
    cleaned_text = text.strip()
    cleaned_ipa = ipa.strip()
    cleaned = respelling.strip()
    name = model.strip()
    if not cleaned_ipa:
        raise ValueError("ipa is empty")
    if not cleaned:
        raise ValueError("respelling is empty")
    if not name:
        raise ValueError("model is empty")
    settings = dict(settings)
    settings["models"] = name
    with config.scope(**settings):
        return _judge(cleaned_text, cleaned_ipa, cleaned, lang, phonetics_lang, name)


def _total(scores: dict[str, int]) -> int:
    """Weighted mean of the 0-100 criteria, scaled to 0-1000."""
    weight = sum(WEIGHTS.values())
    mean = sum(scores[name] * WEIGHTS[name] for name in WEIGHTS) / weight
    return round(mean * 10)


def _reasoning(result: RespellScore) -> str:
    steps = (
        ("Phonetics", result.phonetics),
        ("Stress", result.stress),
        ("Format", result.format),
        ("Verdict", result.verdict),
    )
    return "\n\n".join(f"{title}\n{body.strip()}" for title, body in steps)


def _judge(
    text: str,
    ipa: str,
    respelling: str,
    lang: str,
    phonetics_lang: str,
    model: str,
) -> dict[str, Any]:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    result = llm.ask(
        RespellScore,
        score_system_prompt(ancient.code, modern.code),
        score_user_prompt(text, ipa, respelling, ancient, modern),
    )
    scores = {name: getattr(result, f"{name}_score") for name in WEIGHTS}
    score = _total(scores)
    log.info("%s → %s score: %s (%s)", ancient.code, modern.code, score, model)
    return {"reasoning": _reasoning(result), "score": score}


def improve_system_prompt(lang: str, phonetics_lang: str) -> str:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    return (
        f"You correct a spelling of {ancient.name} IPA for a {modern.name} "
        "text-to-speech voice.\n"
        "You are given the passage, the IPA, a spelling of that IPA, and "
        "an assessment of that spelling. Rewrite the line so the sounds the "
        "voice would make come closer to the IPA, the stress matches the "
        "IPA stress mark, and the line follows the shape stated below. Keep "
        "every part the assessment finds correct. Change a part only to "
        "repair a mistake the rules confirm.\n"
        "If a pair rule and the general spelling rules disagree, follow the "
        "pair rule. The shape of the line, stated below, wins when either of "
        "them tells you to hyphenate a syllable or to capitalize a stressed "
        "syllable.\n"
        "You do not translate. You do not restore the original spelling. "
        f"The voice reads your line with ordinary {modern.name} "
        "letter-to-sound rules and will not see the IPA. The text field is "
        "the corrected line and nothing else.\n\n"
        f"{LINE_FORMAT}\n\n"
        f"Source: {ancient.name} ({ancient.code})\n"
        f"Target voice: {modern.name} ({modern.code})\n\n"
        f"How {modern.name} spelling is read\n"
        f"{rules.orthography(modern.code)}\n\n"
        f"How {ancient.name} maps onto that spelling\n"
        f"{rules.pair(ancient.code, modern.code)}"
        f"{_examples_block(ancient.code, modern.code)}\n\n"
        "First, reason carefully about these things, in this order. Each "
        "step uses the result of the step before it.\n"
        f"1. Phonetics. Take each phonetic mistake the assessment names and "
        f"check it against the spelling rules. For each sound, name the "
        f"sound a {modern.name} voice would make from this line and the "
        "sound the IPA has. Where they differ, give the spelling that makes "
        "the voice produce the IPA, including the substitute the pair rules "
        "name when the target language has no letter for a sound. Where a "
        "vowel's length or quality depends on stress, use the syllable the "
        "IPA mark selects. Keep every sound the assessment leaves standing.\n"
        "2. Stress. From those spellings, for each word, name the syllable "
        "the IPA stress mark selects and the accent the spelling rules place "
        "on its vowel. Correct a stress the assessment finds on the wrong "
        "syllable. Keep a stress the assessment leaves standing.\n"
        "3. Format. From those syllables, put the line in the shape stated "
        "above: each word unbroken, spaces between words, the accent on the "
        "stressed vowel, lowercase letters, a capital only at the start of "
        "a sentence, and the pause marks the IPA uses. Repair each format "
        "mistake the assessment names.\n"
        "Then write the line from the format of step 3."
    )


def improve_user_prompt(
    text: str,
    ipa: str,
    respelling: str,
    ancient: AncientLanguage,
    modern: ModernLanguage,
    assessment: str,
    score: int,
) -> str:
    source_text = text.strip() if text.strip() else "(not given)"
    return (
        f"Source: {ancient.name} ({ancient.code})\n"
        f"Target voice: {modern.name} ({modern.code})\n\n"
        "Original text, for word boundaries only. Do not copy this spelling:\n"
        f"{source_text}\n\n"
        f"IPA:\n{ipa}\n\n"
        f"Line:\n{respelling}\n\n"
        f"Assessment ({score} out of 1000):\n{assessment}\n\n"
        "After your analysis, write the corrected line."
    )


def improve_respelling(
    text: str,
    ipa: str,
    respelling: str,
    lang: str,
    phonetics_lang: str,
    model: str,
    **settings: Any,
) -> dict[str, Any]:
    """Rewrite `respelling` after scoring it.

    The call asks `score_respelling` first. The rewrite sees that reasoning,
    the passage, the IPA, the original respelling, and the conversion rules.
    `model` is the only OpenRouter id asked. The dict has `text`,
    `reasoning`, `score`, and `assessment`.
    """
    scored = score_respelling(text, ipa, respelling, lang, phonetics_lang, model, **settings)
    name = model.strip()
    passed = dict(settings)
    passed["models"] = name
    with config.scope(**passed):
        revised = _revise(
            text.strip(),
            ipa.strip(),
            respelling.strip(),
            lang,
            phonetics_lang,
            name,
            scored["reasoning"],
            scored["score"],
        )
    revised["score"] = scored["score"]
    revised["assessment"] = scored["reasoning"]
    return revised


def _revision_reasoning(result: RespellRevision) -> str:
    steps = (
        ("Phonetics", result.phonetics),
        ("Stress", result.stress),
        ("Format", result.format),
    )
    return "\n\n".join(f"{title}\n{body.strip()}" for title, body in steps)


def _revise(
    text: str,
    ipa: str,
    respelling: str,
    lang: str,
    phonetics_lang: str,
    model: str,
    assessment: str,
    score: int,
) -> dict[str, Any]:
    ancient = languages.ancient(lang)
    modern = languages.modern(phonetics_lang)
    result = llm.ask(
        RespellRevision,
        improve_system_prompt(ancient.code, modern.code),
        improve_user_prompt(text, ipa, respelling, ancient, modern, assessment, score),
    )
    corrected = _clean(result.text)
    if not corrected:
        raise TTSError("The model returned an empty correction.")
    log.info("%s → %s revised: %s (%s)", ancient.code, modern.code, corrected, model)
    return {"text": corrected, "reasoning": _revision_reasoning(result)}


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1].strip()
        if inner:
            text = inner
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "/[]":
        inner = text[1:-1].strip()
        if inner:
            text = inner
    return " ".join(text.split())
