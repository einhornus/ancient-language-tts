"""The public pipeline, one function per step.

    ipa = to_ipa(text, "grc", method="llm", tier="low")
    line = respell_ipa(ipa, "grc", "el", method="llm", text=text)
    # or, from the spelling in one go:
    line = respell(text, "grc", "el", method="rules")
    audio = speak(line, "el", output="out/grc.mp3")

`method` is `"llm"` or `"rules"`.

- `"llm"` is the same pipeline for every language: text → IPA → respelling.
  Each step asks every model of `tier` (see `llm.ipa` / `llm.respelling`).
- `"rules"` uses the language's deterministic engine (`rule_based`). It
  makes no OpenRouter call and ignores `tier`. Only `lat-ecc` → `it` has
  real rules today; the other engines are mocks (`rule_based.is_mock`).

Every keyword in `**settings` is a setting of `config` (keys, models,
cache, ...). One left out is read from the environment or `.env`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from . import chunks, config, languages, rule_based, rules, tts
from . import llm as llm_steps
from .config import Tier

log = logging.getLogger("ancient_language_tts")

Method = Literal["llm", "rules"]
METHODS: tuple[str, ...] = ("llm", "rules")


def to_ipa(
    text: str,
    source: str,
    method: Method = "llm",
    *,
    tier: Tier = "high",
    **settings: Any,
) -> str:
    """Transcribe `text`, written in the ancient language `source`, into IPA.

    Does not depend on a modern language. With `method="llm"` the models of
    `tier` transcribe it; a long passage is split on sentence boundaries
    (`max_chars`) and the pieces are joined.
    """
    language = languages.ancient(source)
    if _method(method) == "rules":
        return rule_based.to_ipa(text.strip(), language.code)
    return llm_steps.to_ipa(text.strip(), language.code, tier=tier, **settings)


def respell(
    text: str,
    source: str,
    modern_lang: str = "en",
    method: Method = "llm",
    *,
    tier: Tier = "high",
    **settings: Any,
) -> str:
    """Respell `text` (ancient spelling) into the spelling of `modern_lang`.

    With `method="rules"` the engine's direct text → respelling rules run,
    with no IPA step. With `method="llm"` the text is transcribed to IPA,
    then that IPA is respelled: the same LLM pipeline for every language.
    A long passage is handled piece by piece and the pieces are joined.
    """
    language = languages.ancient(source)
    modern = languages.modern(modern_lang)
    cleaned = text.strip()
    if _method(method) == "rules":
        return rule_based.respell(cleaned, language.code, modern.code)
    rules.pair(language.code, modern.code)  # fail before paying for the IPA
    with config.scope(**settings):
        lines: list[str] = []
        for part in _pieces(cleaned):
            ipa = llm_steps.to_ipa(part, language.code, tier=tier)
            lines.append(llm_steps.respell(part, ipa, language.code, modern.code, tier=tier))
        return " ".join(lines)


def respell_ipa(
    ipa: str,
    source: str,
    modern_lang: str = "en",
    method: Method = "llm",
    *,
    text: str | None = None,
    tier: Tier = "high",
    **settings: Any,
) -> str:
    """Respell an IPA line of the ancient language `source` into `modern_lang`.

    `text`, when given, is the original passage. The LLM uses it for word
    boundaries only; the rule engine ignores it.
    """
    language = languages.ancient(source)
    modern = languages.modern(modern_lang)
    cleaned = ipa.strip()
    if _method(method) == "rules":
        return rule_based.respell_ipa(cleaned, language.code, modern.code)
    return llm_steps.respell(
        (text or "").strip(),
        cleaned,
        language.code,
        modern.code,
        tier=tier,
        **settings,
    )


def speak(
    respelled_text: str,
    modern_lang: str = "en",
    voice: str | None = None,
    *,
    output: str | Path | None = None,
    seed: int | None = None,
    **settings: Any,
) -> bytes:
    """Read `respelled_text` aloud with an ElevenLabs voice of `modern_lang`.

    `voice` is an ElevenLabs voice id; it defaults to `ELEVENLABS_VOICE_ID`,
    then the language's bundled voice. Returns the audio bytes and writes
    them to `output` when set. `seed` asks for a repeatable take.
    """
    modern = languages.modern(modern_lang)
    return tts.synthesize(
        respelled_text.strip(),
        modern.code,
        output,
        voice_id=voice,
        seed=seed,
        **settings,
    )


def _method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}. Methods: {', '.join(METHODS)}")
    return method


def _pieces(text: str) -> list[str]:
    parts = chunks.split(text) or [text]
    if len(parts) > 1:
        log.info(
            "splitting %s characters into %s pieces (IPA limit %s)",
            len(text),
            len(parts),
            config.current().max_chars,
        )
    return parts
