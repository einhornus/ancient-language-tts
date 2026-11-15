"""One passage, spoken.

A passage longer than the IPA limit is split on sentence boundaries first
(`chunks`). Each piece is transcribed and respelled on its own, and the
pieces are joined before the voice reads them. A short passage is one piece.

With `use_rules=False` (default), the LLM pipeline runs:

    1. Every model in the tier transcribes the ancient-language text into
       IPA, scores its own line, and rewrites it. The rewrite stands when
       that model scores it higher than the first line. Each standing line
       is scored by every model still in the tier. The call returns the
       line with the highest mean score.
    2. Every model in the tier respells that IPA into a modern language,
       scores its own line, and rewrites it the same way. The system
       prompt is that modern spelling's letter-to-sound rules, plus the
       pair rules for these two languages. The call returns the line with
       the highest mean score.
    3. ElevenLabs speaks the respelling with a voice of the modern language.

With `use_rules=True`, steps 1–2 use the deterministic engine in
`rule_based` when one exists for the language pair (currently
Ecclesiastical Latin → Italian). ElevenLabs still speaks the result.

`through_ipa=False` skips both of those and respells the Latin spelling
straight into Italian (`rule_based.direct`). The returned `ipa` is empty.

There is no fourth step that "fixes" the audio. If a sound has no letter
in the modern language, the pair rules already name the substitute, and
the voice says that.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import chunks, config, languages, rule_based, tts
from . import llm as llm_steps
from .config import Tier
from .errors import TTSError

ipa_step = llm_steps.ipa
respell_step = llm_steps.respelling

log = logging.getLogger("ancient_language_tts")


@dataclass
class Speech:
    """One spoken passage. `summary()` drops `audio` so the rest is JSON."""

    source: str
    source_name: str
    target: str
    target_name: str
    text: str
    ipa: str
    respelling: str
    voice_id: str
    voice_name: str | None
    model: str
    tier: str
    path: str | None
    audio: bytes

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("audio")
        data["ok"] = True
        return data


def speak(
    text: str,
    source: str,
    target: str = "en",
    output: str | Path | None = None,
    *,
    voice_id: str | None = None,
    openrouter_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    tier: Tier = "high",
    models: str | Sequence[str] | None = None,
    tier_models: Mapping[str, str | Sequence[str]] | None = None,
    cache_dir: str | Path | None = None,
    cache: bool | None = None,
    max_tokens: int | None = None,
    max_chars: int | None = None,
    elevenlabs_model: str | None = None,
    output_format: str | None = None,
    stability: float | None = None,
    similarity_boost: float | None = None,
    seed: int | None = None,
    use_rules: bool = False,
    through_ipa: bool = True,
) -> Speech:
    """Speak `text` in `source` with a `target` voice.

    `source` is an ancient-language code (`lat`, `lat-ecc`, `grc`, `eme`,
    `enm`, `ang`, `gem-pro`, `ine-pro`, `sla-pro`). `target` is a modern-language
    code (`en` or `it`). English pair files are bundled for every ancient
    language except the two Latins, which are bundled with Italian. `output`
    is the audio file to write; leave it empty to only get the bytes on the
    returned `Speech`.

    `use_rules` asks for the deterministic rule engine when one exists for
    this language pair (no OpenRouter calls for IPA or respelling). When
    the pair has no rule engine, a `ValueError` is raised. The default is
    the LLM pipeline.

    `through_ipa` chooses the shape of that rule engine. True (default)
    transcribes to IPA, then respells. False respells the Latin spelling
    directly and does not call an LLM. A pair with no direct rules raises
    `ValueError`.

    `openrouter_api_key` and `elevenlabs_api_key` are your keys. `tier`
    (`high`, `medium`, `low`) picks the models. The IPA step and the
    respelling step each ask every model of that tier. `tier_models`
    replaces the lists of the tiers it names.
    `models` replaces the whole list with your own OpenRouter ids.
    `voice_id` replaces the target language's voice for this call.
    `elevenlabs_model` and `output_format` select the speech model and the
    encoding. `stability` and `similarity_boost` are the ElevenLabs voice
    settings, each from 0 to 1. They default to 1 and 0.75: the fixed
    reading, and the clarity setting short of the distortion their docs
    warn about. `seed` asks ElevenLabs for a repeatable
    take. `cache_dir`, `cache`, and `max_tokens` control the caches and
    the LLM output budget. `max_chars` is how many source characters one
    IPA call may see. A longer passage is split on sentence boundaries,
    and each piece is transcribed and respelled on its own.

    Every one of these left as None is read from the environment or `.env`
    (see `config`). A rerun with the same text, rules, and voice reads the
    LLM answers and the audio from the cache.
    """
    with config.scope(
        openrouter_api_key=openrouter_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        models=models,
        tier_models=tier_models,
        cache_dir=cache_dir,
        cache=cache,
        max_tokens=max_tokens,
        max_chars=max_chars,
        elevenlabs_model=elevenlabs_model,
        voice_id=voice_id,
        output_format=output_format,
        stability=stability,
        similarity_boost=similarity_boost,
    ):
        return _speak(
            text,
            source,
            target,
            output=output,
            voice_id=voice_id,
            tier=tier,
            seed=seed,
            use_rules=use_rules,
            through_ipa=through_ipa,
        )


def _speak(
    text: str,
    source: str,
    target: str,
    *,
    output: str | Path | None,
    voice_id: str | None,
    tier: Tier,
    seed: int | None,
    use_rules: bool,
    through_ipa: bool,
) -> Speech:
    settings = config.current()
    language = languages.ancient(source)
    modern = languages.modern(target)
    cleaned = text.strip()
    if (use_rules or not through_ipa) and not rule_based.has(language.code, modern.code):
        have = ", ".join(f"{s}→{t}" for s, t in rule_based.available()) or "(none)"
        raise ValueError(
            f"no rule-based solution for {language.code} → {modern.code}. "
            f"Available: {have}."
        )
    parts = chunks.split(cleaned) or [cleaned]
    if len(parts) > 1:
        log.info(
            "splitting %s characters into %s pieces (IPA limit %s)",
            len(cleaned),
            len(parts),
            settings.max_chars,
        )
    ipa_lines: list[str] = []
    spelling_lines: list[str] = []
    for index, part in enumerate(parts, start=1):
        if len(parts) > 1:
            log.info("piece %s/%s, %s characters", index, len(parts), len(part))
        if not through_ipa:
            spelling_lines.append(rule_based.direct(part, language.code, modern.code))
            ipa_lines.append("")
            continue
        if use_rules:
            ipa_line = rule_based.to_ipa(part, language.code, modern.code)
            spelling_lines.append(rule_based.respell(ipa_line, language.code, modern.code))
        else:
            ipa_line = ipa_step.to_ipa(part, language.code, tier=tier)
            spelling_lines.append(respell_step.respell(
                part,
                ipa_line,
                language.code,
                modern.code,
                tier=tier,
            ))
        ipa_lines.append(ipa_line)
    ipa = " ".join(line for line in ipa_lines if line)
    respelling = " ".join(spelling_lines)
    try:
        audio = tts.synthesize(
            respelling,
            modern.code,
            output=output,
            voice_id=voice_id,
            seed=seed,
        )
    except TTSError as exc:
        raise TTSError(f"{exc} Respelling: {respelling}") from exc
    chosen = (voice_id or settings.voice_id or modern.voice_id).strip()
    language_voice = modern.voice_name if chosen == modern.voice_id else None
    return Speech(
        source=language.code,
        source_name=language.name,
        target=modern.code,
        target_name=modern.name,
        text=cleaned,
        ipa=ipa,
        respelling=respelling,
        voice_id=chosen,
        voice_name=language_voice,
        model=settings.elevenlabs_model,
        tier=tier,
        path=str(Path(output)) if output is not None else None,
        audio=audio,
    )
