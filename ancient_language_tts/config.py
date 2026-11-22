"""Settings for one run: the two keys, the LLM chain, and the caches.

Bring your own keys. `OPENROUTER_API_KEY` pays for the LLM calls (through
Pydantic AI). `ELEVENLABS_API_KEY` pays for the speech.

Every setting passed to a step function (`to_ipa`, `respell`,
`respell_ipa`, `speak`) wins. A missing one is read from the environment,
then from the nearest `.env` file at or above the current folder, then
the built-in default:

    OPENROUTER_API_KEY          the LLM key
    ELEVENLABS_API_KEY          the speech key
    ELEVENLABS_VOICE_ID         a voice for every modern language, instead
                                of that language's built-in voice
    ANCIENT_TTS_ELEVENLABS_MODEL  ElevenLabs model (default eleven_v4)
    ANCIENT_TTS_OUTPUT_FORMAT   audio encoding (default mp3_44100_128)
    ANCIENT_TTS_STABILITY       voice stability, 0 to 1 (default 1). Higher
                                holds the reading to one baseline
    ANCIENT_TTS_SIMILARITY      similarity boost, 0 to 1 (default 0.75).
                                Higher is clearer; very high values distort
    ANCIENT_TTS_SPEED           speaking rate, 0.7 to 1.2 (default 0.7).
                                Only models before v3 accept it
    ANCIENT_TTS_LLM_MODELS      comma-separated OpenRouter ids, tried in
                                order, instead of the tier lists
    ANCIENT_TTS_MODELS_HIGH     comma-separated OpenRouter ids replacing one
    ANCIENT_TTS_MODELS_MEDIUM   built-in tier list
    ANCIENT_TTS_MODELS_LOW
    ANCIENT_TTS_CACHE_DIR       where answers and audio are cached
                                (default `.cache/ancient-language-tts`)
    ANCIENT_TTS_CACHE           `0` turns both caches off
    ANCIENT_TTS_MAX_TOKENS      output budget of one LLM call (default 16384)
    ANCIENT_TTS_MAX_CHARS       source characters one IPA call may see
                                (default 1000). A longer passage is split
                                on sentence boundaries; each piece is
                                transcribed and respelled on its own.

A step function resolves these once and installs them for the run with `use`, so
nested calls read them with `current` instead of passing them around.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from dotenv import find_dotenv, load_dotenv

from .chunks import MAX_CHARS as DEFAULT_MAX_CHARS

Tier = Literal["high", "medium", "low"]
TIERS: tuple[Tier, ...] = get_args(Tier)

# How hard `to_ipa` and `respell` work the tier (see `llm.ensemble`).
# "ultra": every model writes and improves, every model scores every line.
# "high": every model writes and improves, the first model scores.
# "medium": every model writes, the first model scores.
# "low": the first model writes and improves. "basic": the first model writes.
Effort = Literal["ultra", "high", "medium", "low", "basic"]
EFFORTS: tuple[Effort, ...] = get_args(Effort)

# OpenRouter models for each cost tier, first choice first. `to_ipa` and
# `respell` each ask every model of the tier you name. `ask` tries the next
# model in the tier, then the next cheaper tier.
TIER_MODELS: dict[Tier, tuple[str, ...]] = {
    "high": (
        "anthropic/claude-opus-4.7",
        "google/gemini-4-carbon",
    ),
    "medium": (
        "anthropic/claude-sonnet-5.5",
        "openai/gpt-5.6-sol",
        "x-ai/grok-4.7",
        "google/gemini-4-carbon",
    ),
    "low": (
        "google/gemini-2.5-flash-lite",
        "openai/gpt-5.6-luna",
        "anthropic/claude-haiku-5.5",
    ),
}
_NEXT_TIER: dict[Tier, Tier | None] = {"high": "medium", "medium": "low", "low": None}

# Thinking models spend this budget on reasoning before the structured
# answer. The provider default is often too small.
DEFAULT_MAX_TOKENS = 16384
DEFAULT_CACHE_DIR = Path(".cache/ancient-language-tts")
DEFAULT_ELEVENLABS_MODEL = "eleven_v4"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
# ElevenLabs: higher stability keeps the delivery on a fixed baseline, and
# low values make the reading random and rushed. 1 is that fixed reading.
# Similarity is their clarity control. The docs say very high values add
# distortion and copy artifacts out of the recording, which covers the
# consonants, so this stays at their clarity default rather than at 1.
DEFAULT_STABILITY = 1.0
DEFAULT_SIMILARITY = 0.75
# Speaking rate, from ElevenLabs' range of 0.7 to 1.2 (1 is the voice's
# own pace). The slowest end gives every sound of a respelling its time.
DEFAULT_SPEED = 0.7
SPEED_RANGE = (0.7, 1.2)

APP_TITLE = "Ancient Language TTS"
APP_URL = "https://github.com/einhornus/ancient-language-tts"

_SETTING_KEYS = frozenset({
    "openrouter_api_key",
    "elevenlabs_api_key",
    "models",
    "tier_models",
    "cache_dir",
    "cache",
    "max_tokens",
    "max_chars",
    "elevenlabs_model",
    "voice_id",
    "output_format",
    "stability",
    "similarity_boost",
    "speed",
})


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    models: tuple[str, ...] = ()
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache: bool = True
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_chars: int = DEFAULT_MAX_CHARS
    elevenlabs_model: str = DEFAULT_ELEVENLABS_MODEL
    voice_id: str | None = None
    output_format: str = DEFAULT_OUTPUT_FORMAT
    stability: float = DEFAULT_STABILITY
    similarity_boost: float = DEFAULT_SIMILARITY
    speed: float = DEFAULT_SPEED
    tiers: Mapping[Tier, tuple[str, ...]] = field(default_factory=lambda: dict(TIER_MODELS))

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def audio_cache_dir(self) -> Path:
        return self.cache_dir / "audio"

    def llm_models(self, tier: Tier) -> list[str]:
        """The chain `ask` walks: `models` when set, else this tier and every cheaper one."""
        if self.models:
            return list(self.models)
        names: list[str] = []
        current: Tier | None = tier
        while current is not None:
            for name in self.tiers.get(current, TIER_MODELS[current]):
                if name not in names:
                    names.append(name)
            current = _NEXT_TIER[current]
        return names


def resolve(
    *,
    openrouter_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    models: str | Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    cache: bool | None = None,
    max_tokens: int | None = None,
    max_chars: int | None = None,
    tier_models: Mapping[str, str | Sequence[str]] | None = None,
    elevenlabs_model: str | None = None,
    voice_id: str | None = None,
    output_format: str | None = None,
    stability: float | None = None,
    similarity_boost: float | None = None,
    speed: float | None = None,
) -> Settings:
    """Settings from the arguments, then the environment and `.env`, then the defaults.

    `tier_models` replaces the model list of each tier it names, e.g.
    `{"low": ["google/gemini-3-flash-preview"]}`. A tier it leaves out comes
    from `ANCIENT_TTS_MODELS_<TIER>`, else the built-in list.
    """
    load_env()
    openrouter = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY") or "").strip() or None
    elevenlabs = (elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY") or "").strip() or None
    if models is None:
        models = os.getenv("ANCIENT_TTS_LLM_MODELS", "")
    chain = _chain(models)
    given = dict(tier_models or {})
    unknown = sorted(set(given) - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tier in tier_models: {', '.join(unknown)} (use {', '.join(TIERS)})")
    tiers: dict[Tier, tuple[str, ...]] = {}
    for tier in TIERS:
        names = _chain(given[tier]) if tier in given else _chain(os.getenv(f"ANCIENT_TTS_MODELS_{tier.upper()}", ""))
        tiers[tier] = names or TIER_MODELS[tier]
    if cache_dir is None:
        cache_dir = os.getenv("ANCIENT_TTS_CACHE_DIR") or DEFAULT_CACHE_DIR
    if cache is None:
        cache = os.getenv("ANCIENT_TTS_CACHE", "1").strip().lower() not in ("0", "false", "no")
    if max_tokens is None:
        raw = os.getenv("ANCIENT_TTS_MAX_TOKENS", "").strip()
        max_tokens = int(raw) if raw else DEFAULT_MAX_TOKENS
    max_chars = _count(
        "max_chars",
        max_chars if max_chars is not None else os.getenv("ANCIENT_TTS_MAX_CHARS", ""),
        DEFAULT_MAX_CHARS,
    )
    if elevenlabs_model is None:
        elevenlabs_model = os.getenv("ANCIENT_TTS_ELEVENLABS_MODEL", "").strip() or DEFAULT_ELEVENLABS_MODEL
    if voice_id is None:
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip() or None
    else:
        voice_id = voice_id.strip() or None
    if output_format is None:
        output_format = os.getenv("ANCIENT_TTS_OUTPUT_FORMAT", "").strip() or DEFAULT_OUTPUT_FORMAT
    stability = _unit(
        "stability",
        stability if stability is not None else os.getenv("ANCIENT_TTS_STABILITY", ""),
        DEFAULT_STABILITY,
    )
    similarity_boost = _unit(
        "similarity_boost",
        similarity_boost if similarity_boost is not None else os.getenv("ANCIENT_TTS_SIMILARITY", ""),
        DEFAULT_SIMILARITY,
    )
    speed = _speed(speed if speed is not None else os.getenv("ANCIENT_TTS_SPEED", ""))
    return Settings(
        openrouter_api_key=openrouter,
        elevenlabs_api_key=elevenlabs,
        models=chain,
        cache_dir=Path(cache_dir).expanduser(),
        cache=cache,
        max_tokens=max_tokens,
        max_chars=max_chars,
        elevenlabs_model=elevenlabs_model.strip(),
        voice_id=voice_id,
        output_format=output_format.strip(),
        stability=stability,
        similarity_boost=similarity_boost,
        speed=speed,
        tiers=tiers,
    )


def _count(name: str, raw: int | str, default: int) -> int:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number of at least 1") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _unit(name: str, raw: float | str, default: float) -> float:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number between 0 and 1") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _speed(raw: float | str) -> float:
    low, high = SPEED_RANGE
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return DEFAULT_SPEED
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"speed must be a number between {low} and {high}") from None
    if not low <= value <= high:
        raise ValueError(f"speed must be between {low} and {high}")
    return value


_env_loaded = False
_env_lock = threading.Lock()


def load_env() -> None:
    """Read the nearest `.env` at or above the current folder, once.

    Variables already set in the environment are kept.
    """
    global _env_loaded
    if _env_loaded:
        return
    # Threads (the test harness transcribes in parallel) must wait for the
    # first load to finish, not see the flag early and read an empty env.
    with _env_lock:
        if _env_loaded:
            return
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)
        _env_loaded = True


def _chain(models: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(models, str):
        models = models.split(",")
    return tuple(_bare(model) for model in models if model and str(model).strip())


def _bare(model: str) -> str:
    """`openrouter:google/x` and `google/x` both mean the OpenRouter id `google/x`."""
    name = model.strip()
    if name.startswith("openrouter:"):
        name = name[len("openrouter:"):]
    return name


_current: ContextVar[Settings | None] = ContextVar("ancient_tts_settings", default=None)


def current() -> Settings:
    """The settings installed by `use`, or ones resolved from the environment."""
    settings = _current.get()
    return settings if settings is not None else resolve()


@contextmanager
def use(settings: Settings) -> Iterator[Settings]:
    token = _current.set(settings)
    try:
        yield settings
    finally:
        _current.reset(token)


@contextmanager
def scope(**kwargs: object) -> Iterator[Settings]:
    """Install `resolve(**kwargs)` for the block.

    A keyword left as None is read from the environment. When every keyword
    is None and a settings object is already installed, that object stays:
    an outer call can call the inner steps and they keep its keys.
    """
    unknown = sorted(set(kwargs) - _SETTING_KEYS)
    if unknown:
        raise TypeError(f"unknown setting: {', '.join(unknown)}")
    passed = {key: value for key, value in kwargs.items() if value is not None}
    if passed or _current.get() is None:
        with use(resolve(**passed)):
            yield current()
    else:
        yield current()
