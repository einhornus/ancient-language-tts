"""The public pipeline, one function per step.

    pieces = to_ipa(text, "grc", method="llm", tier="low")
    # [{"text": "φίλος.", "ipa": ["ˈ", "pʰ", "i", ".", ...]}, ...]
    line = respell_ipa(pieces, "grc", "el", method="llm")
    # or, from the spelling in one go:
    line = respell(text, "grc", "el", method="rules")
    audio = speak(line, "el", output="out/grc.mp3")

`method` is `"llm"` or `"rules"`.

- `"llm"` is the same pipeline for every language: text → IPA → respelling.
  Each step uses the models of `tier` as `effort` says (see `llm.ensemble`):
  "ultra" (default), "high", "medium", "low", or "basic".
- `"rules"` uses the language's deterministic engine (`rule_based`). It
  makes no OpenRouter call and ignores `tier` and `effort`. Every ancient
  language has real rules for each of its bundled targets.

IPA is always a list of tokens: the phonemes of `data/<code>/phonemes.json`
and the special tokens of `ipa_tokens` (stress, syllable dot, space, line
break, punctuation). `ipa_tokens.render` writes it as one line;
`ipa_tokens.parse` reads a written line back.

`to_ipa` returns the passage as chunks (`chunks.Chunk`): one
`{"text": piece, "ipa": tokens}` per piece the passage was cut into.
`respell_ipa` takes those chunks, so each piece of IPA is respelled with
the text it came from. `chunks.join_ipa` joins the chunks into one line.

Every keyword in `**settings` is a setting of `config` (keys, models,
cache, ...). One left out is read from the environment or `.env`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal

from . import chunks, config, ipa_tokens, languages, rule_based, rules, tts
from . import llm as llm_steps
from .config import Effort, Tier

log = logging.getLogger("ancient_language_tts")

Method = Literal["llm", "rules"]
METHODS: tuple[str, ...] = ("llm", "rules")


def to_ipa(
    text: str,
    source: str,
    method: Method = "llm",
    *,
    tier: Tier = "high",
    effort: Effort = "ultra",
    **settings: Any,
) -> list[chunks.Chunk]:
    """Transcribe `text`, written in the ancient language `source`, into IPA.

    Returns the passage as chunks: `[{"text": piece, "ipa": tokens}, ...]`.
    A long passage is split on sentence boundaries (`max_chars`), one chunk
    per piece; a short one is a single chunk. Does not depend on a modern
    language. With `method="llm"` the models of `tier` transcribe each piece.
    """
    language = languages.ancient(source)
    if _method(method) == "rules":
        with config.scope(**settings):
            parts = _pieces(text.strip())
        return [{"text": part, "ipa": rule_based.to_ipa(part, language.code)} for part in parts]
    return llm_steps.to_ipa(text.strip(), language.code, tier=tier, effort=effort, **settings)


def respell(
    text: str,
    source: str,
    modern_lang: str = "en",
    method: Method = "llm",
    *,
    tier: Tier = "high",
    effort: Effort = "ultra",
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
            pieces = llm_steps.to_ipa(part, language.code, tier=tier, effort=effort)
            lines.append(llm_steps.respell(
                pieces, language.code, modern.code, tier=tier, effort=effort
            ))
        return " ".join(lines)


def respell_ipa(
    ipa: Sequence[chunks.Chunk] | Sequence[str],
    source: str,
    modern_lang: str = "en",
    method: Method = "llm",
    *,
    tier: Tier = "high",
    effort: Effort = "ultra",
    **settings: Any,
) -> str:
    """Respell the IPA of the ancient language `source` into `modern_lang`.

    `ipa` is the chunk list `to_ipa` returns. Each chunk is respelled with
    its own text, which the LLM uses for word boundaries only; the rule
    engine ignores the text. A bare token list is also taken: it is cut on
    sentence boundaries and respelled without text. A written line goes
    through `ipa_tokens.parse` first.
    """
    language = languages.ancient(source)
    modern = languages.modern(modern_lang)
    with config.scope(**settings):
        pieces = _chunks(ipa)
    for piece in pieces:
        piece["ipa"] = ipa_tokens.ensure(piece["ipa"], language.code)
    if _method(method) == "rules":
        return rule_based.respell_ipa(chunks.join_ipa(pieces), language.code, modern.code)
    return llm_steps.respell(
        pieces,
        language.code,
        modern.code,
        tier=tier,
        effort=effort,
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


def _chunks(ipa: Any) -> list[chunks.Chunk]:
    """The chunks of `ipa`: a chunk list as it is, a bare token list cut into chunks."""
    if isinstance(ipa, str):
        raise TypeError("IPA is a list of tokens, not a string. Use ipa_tokens.parse() for a written line.")
    items = list(ipa)
    if items and all(isinstance(item, str) for item in items):
        pieces, _ = chunks.from_ipa(items)
        return pieces or [{"text": "", "ipa": items}]
    return chunks.ensure(items)


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
