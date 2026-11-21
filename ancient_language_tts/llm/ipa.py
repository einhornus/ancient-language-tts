"""Step 1. Ancient-language text to IPA, and a score for a transcription.

`effort` decides how hard `to_ipa` works the tier (see `llm.ensemble`).
What follows is "ultra", the default. "high" scores the standing lines
with the first model only. "medium" also skips the rewrite. "low" asks
only the first model and rewrites its line. "basic" asks only the first
model, once.

At "ultra", `to_ipa` asks every model in the tier. Each model transcribes the passage,
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

IPA is a list of tokens (see `ipa_tokens`). The model writes a line of
text; `ipa_tokens.parse_with_penalty` cuts it into the phonemes of
`phonemes.json` and the special tokens, and keeps each symbol that is
neither as a token of its own. A line that is empty or holds no allowed
phoneme fails that call, which is then retried like any other failure.
A line with unknown symbols is scored, and its score is cut by
`PENALTY_FACTOR` times the share of its symbols that are unknown. The
line `to_ipa` returns has its unknown symbols dropped.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import chunks, config, ipa_tokens, languages, rules
from ..config import Effort, Tier
from ..errors import TTSError
from ..languages import AncientLanguage
from . import client as llm
from . import ensemble
from .ensemble import ATTEMPTS  # noqa: F401  (kept importable from here)
from .schemas import IpaRevision, IpaScore, IpaTranscription

log = logging.getLogger("ancient_language_tts")

# How hard a score is cut for symbols outside the inventory. A line whose
# share of unknown symbols is p keeps max(0, 1 - PENALTY_FACTOR * p) of
# its score: at 2.0, one unknown symbol in ten costs 20% of the score.
PENALTY_FACTOR = 2.0

# Values are the sounds. Syllables and stress are how those sounds are
# grouped and marked. The line is the writing. `score` is this weighted
# mean, scaled from 0-100 to 0-1000.
WEIGHTS = {"values": 5.0, "syllables": 1.0, "stress": 2.0, "line": 1.0}


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
        f"{_inventory(language.code)}"
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


def _inventory(code: str) -> str:
    """The prompt section that names the only symbols the line may use."""
    found = ipa_tokens.phonemes(code)
    return (
        "Allowed symbols\n"
        "Write each sound as exactly one of these phonemes, and use no other "
        "letter, length mark, or diacritic: "
        f"{' '.join(found)}\n"
        "Besides those, the line holds only ˈ, ˌ, the syllable dot, spaces "
        "between words, and the pause marks.\n\n"
    )


def user_prompt(text: str, language: AncientLanguage) -> str:
    return f"Language: {language.name} ({language.code}), {language.era}\n\nText:\n{text}"


def to_ipa(
    text: str,
    source: str,
    *,
    tier: Tier = "high",
    effort: Effort = "ultra",
    **settings: Any,
) -> list[chunks.Chunk]:
    """Transcribe `text` into IPA, piece by piece, keeping the best line of each piece.

    Returns the pieces as chunks, `{"text": piece, "ipa": tokens}`, the
    tokens being allowed IPA tokens (see `ipa_tokens`). `chunks.join_ipa`
    joins them into one line. `respell` takes the chunks as they are.

    `effort` decides how the models of `tier` are used (see `llm.ensemble`).
    At "ultra", every model of `tier` transcribes the passage, scores its own
    line, and rewrites it from that assessment. The same model then scores
    the rewrite. The rewrite stands only when that score is higher than the
    score of the first line; otherwise the first line stands. Each standing
    line is scored by every model still in the tier. The score of a line is
    the arithmetic mean of those scores. On a tie, the earlier model in the
    tier wins. "high" improves the same way but only the first model scores.
    "medium" skips the improvement; the first model scores. "low" asks the
    first model alone and improves its line. "basic" asks the first model
    alone, once.

    A call is tried three times. A model that still cannot produce the first
    transcription is dropped. One that still cannot score that line, or still
    cannot rewrite it, keeps the first line. One that still cannot score a
    standing line is left out of the mean. When `models` is set, that list is
    the tier. Settings are the settings of `config`.

    A passage longer than `max_chars` is split on sentence boundaries, and
    each piece is transcribed on its own: one chunk per piece.
    `api.respell` respells each piece before it transcribes the next.
    """
    ensemble.check(effort)
    with config.scope(**settings):
        parts = chunks.split(text)
        if not parts:
            raise ValueError("text is empty")
        if len(parts) > 1:
            log.info(
                "splitting %s characters into %s pieces (IPA limit %s)",
                len(text.strip()),
                len(parts),
                config.current().max_chars,
            )
        pieces: list[chunks.Chunk] = []
        for index, part in enumerate(parts, start=1):
            if len(parts) > 1:
                log.info("piece %s/%s, %s characters", index, len(parts), len(part))
            pieces.append({"text": part, "ipa": _ensemble(part, source, tier, effort)})
        return pieces


def _ensemble(text: str, source: str, tier: Tier, effort: Effort = "ultra") -> list[str]:
    language = languages.ancient(source)
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text is empty")
    ensemble.check(effort)
    llm.require_key()
    winner, line, mean = ensemble.run(
        ensemble.models(tier),
        effort,
        draft=lambda name: _draft(cleaned, language, name),
        judge=lambda name, ipa: _judge(cleaned, ipa, language.code, name),
        revise=lambda name, ipa, assessed: _revise(
            cleaned, ipa, language.code, name, assessed["reasoning"], assessed["score"]
        )["ipa"],
        label=language.code,
        noun="transcription",
        show=ipa_tokens.render,
    )
    log.info(
        "%s IPA: %s (%s, effort %s, mean %s)",
        language.code,
        ipa_tokens.render(line),
        winner,
        effort,
        ensemble.show_mean(mean),
    )
    bad = ipa_tokens.unknown(line, language.code)
    if bad:
        log.warning("%s IPA: dropping symbols outside the inventory: %s", language.code, bad)
        line = ipa_tokens.drop_unknown(line, language.code)
    return line


def _draft(text: str, language: AncientLanguage, model: str) -> list[str]:
    result = llm.ask(IpaTranscription, system_prompt(language.code), user_prompt(text, language))
    if not result.ipa.strip():
        raise TTSError("The model returned an empty transcription.")
    ipa, penalty = ipa_tokens.parse_with_penalty(result.ipa, language.code)
    log.info("%s IPA draft: %s (%s, unknown %.0f%%)", language.code, ipa_tokens.render(ipa), model, penalty * 100)
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


def score_user_prompt(text: str, ipa: list[str], language: AncientLanguage) -> str:
    return (
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"Text:\n{text}\n\n"
        f"IPA:\n{ipa_tokens.render(ipa)}\n\n"
        "After your analysis, score this transcription."
    )


def score_ipa(text: str, ipa: list[str], lang: str, model: str, **settings: Any) -> dict[str, Any]:
    """Score how well `ipa` (a token list) renders `text` in `lang`.

    `model` is one OpenRouter id, and it is the only model asked. Settings
    are the settings of `config`. The dict has `reasoning`, `score`, and
    `penalty`. `score` runs from 0 to 1000 and is already cut by
    `PENALTY_FACTOR` times `penalty`, the share of symbols in `ipa` that are
    outside the inventory.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("text is empty")
    cleaned_ipa = ipa_tokens.ensure(ipa, lang, allow_unknown=True)
    name = model.strip()
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


def _judge(text: str, ipa: list[str], lang: str, model: str) -> dict[str, Any]:
    language = languages.ancient(lang)
    result = llm.ask(
        IpaScore,
        score_system_prompt(language.code),
        score_user_prompt(text, ipa, language),
    )
    scores = {name: getattr(result, f"{name}_score") for name in WEIGHTS}
    raw = _total(scores)
    reasoning = _reasoning(result)
    bad = ipa_tokens.unknown(ipa, language.code)
    penalty = ipa_tokens.penalty(ipa, language.code)
    score = _penalize(raw, penalty)
    if bad:
        reasoning += (
            "\n\nUnknown symbols\n"
            f"These symbols are outside the allowed inventory: {' '.join(dict.fromkeys(bad))}. "
            f"They are {penalty:.0%} of the symbols, which cuts the score from {raw} to {score}."
        )
        log.info("%s IPA score: %s, %s before the unknown-symbol penalty (%s)", language.code, score, raw, model)
    else:
        log.info("%s IPA score: %s (%s)", language.code, score, model)
    return {"reasoning": reasoning, "score": score, "penalty": penalty}


def _penalize(score: int, penalty: float) -> int:
    """`score` cut by `PENALTY_FACTOR` times `penalty`, never below 0."""
    return round(score * max(0.0, 1.0 - PENALTY_FACTOR * penalty))


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
        f"{_inventory(language.code)}"
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


def improve_user_prompt(
    text: str, ipa: list[str], language: AncientLanguage, assessment: str, score: int
) -> str:
    return (
        f"Language: {language.name} ({language.code}), {language.era}\n\n"
        f"Text:\n{text}\n\n"
        f"IPA:\n{ipa_tokens.render(ipa)}\n\n"
        f"Assessment ({score} out of 1000):\n{assessment}\n\n"
        "After your analysis, write the corrected IPA."
    )


def improve_ipa(text: str, ipa: list[str], lang: str, model: str, **settings: Any) -> dict[str, Any]:
    """Rewrite `ipa` (a token list) after scoring it.

    The call asks `score_ipa` first. The rewrite sees that reasoning, the
    passage, the original IPA, and the language's phonetic rules. `model`
    is the only OpenRouter id asked. The dict has `ipa` (a token list),
    `reasoning`, `score`, and `assessment`.
    """
    scored = score_ipa(text, ipa, lang, model, **settings)
    name = model.strip()
    passed = dict(settings)
    passed["models"] = name
    with config.scope(**passed):
        revised = _revise(
            text.strip(),
            ipa_tokens.ensure(ipa, lang, allow_unknown=True),
            lang,
            name,
            scored["reasoning"],
            scored["score"],
        )
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


def _revise(text: str, ipa: list[str], lang: str, model: str, assessment: str, score: int) -> dict[str, Any]:
    language = languages.ancient(lang)
    result = llm.ask(
        IpaRevision,
        improve_system_prompt(language.code),
        improve_user_prompt(text, ipa, language, assessment, score),
    )
    if not result.ipa.strip():
        raise TTSError("The model returned an empty correction.")
    corrected, penalty = ipa_tokens.parse_with_penalty(result.ipa, language.code)
    log.info(
        "%s IPA revised: %s (%s, unknown %.0f%%)", language.code, ipa_tokens.render(corrected), model, penalty * 100
    )
    return {"ipa": corrected, "reasoning": _revision_reasoning(result)}

