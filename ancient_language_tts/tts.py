"""Step 3. Modern-language spelling to speech, through ElevenLabs.

The text is a phonetic respelling, so nothing is allowed to rewrite it.
Text normalization and the language-specific normalizer are off. The
request also refuses the instant-clone substitute of a professional
voice, which ElevenLabs describes as the more expressive path. No audio
effects are sent, and neither is a latency optimization: those trade
clarity for color or speed, and the fastest latency mode can mispronounce.

`language_code` is sent when the model accepts it, so the voice uses
that language's letter-to-sound rules. Short respellings are the
ambiguous case their docs say this field is for. `eleven_multilingual_v2`
ignores it and decides from the text, so it is left out for that model.

Stability defaults to 1, the top of their scale: higher values hold the
delivery to a fixed baseline, and low values make the reading random and
rushed. Similarity stays at 0.75. Higher is clearer, but they say very
high values distort and copy artifacts from the recording. On models
that still accept the other knobs, style is 0 (they say a non-zero style
mispronounces), speaker boost is on (clarity), and speed is 1 (the
extremes hurt quality). Eleven v4 accepts only stability and similarity,
and Eleven v3 accepts stability and style, so the others are not sent.

A call is cached under `<cache_dir>/audio/` from the text, the voice, the
model, and these settings. A rerun of an unchanged passage does not pay
for the audio again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from . import config, languages
from .errors import TTSError
from .languages import ModernLanguage

log = logging.getLogger("ancient_language_tts")

URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TIMEOUT = 180.0
_SEED_MAX = 4_294_967_295


def synthesize(
    text: str,
    target: str = "en",
    output: str | Path | None = None,
    *,
    voice_id: str | None = None,
    seed: int | None = None,
    **settings: Any,
) -> bytes:
    """Synthesize `text`, already spelled in `target`. Returns the audio bytes.

    When `output` is set, the bytes are also written there. `voice_id`
    overrides the language's voice and `ELEVENLABS_VOICE_ID` for this call.
    """
    with config.scope(**settings):
        return _synthesize(text, target, output=output, voice_id=voice_id, seed=seed)


def _synthesize(
    text: str,
    target: str,
    *,
    output: str | Path | None,
    voice_id: str | None,
    seed: int | None,
) -> bytes:
    settings = config.current()
    language = languages.modern(target)
    spoken = " ".join(text.strip().split())
    if not spoken:
        raise ValueError("text is empty")
    if not settings.elevenlabs_api_key:
        raise TTSError("No ElevenLabs key. Pass elevenlabs_api_key= or set ELEVENLABS_API_KEY.")
    chosen = (voice_id or settings.voice_id or language.voice_id).strip()
    if not chosen:
        raise TTSError(f"no voice for {language.code}. Pass voice_id= or set ELEVENLABS_VOICE_ID.")
    model = settings.elevenlabs_model
    output_format = settings.output_format
    if seed is not None and not 0 <= seed <= _SEED_MAX:
        raise ValueError(f"seed must be from 0 to {_SEED_MAX}")
    audio = _cached_audio(spoken, language, chosen, model, output_format, seed)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        log.info("wrote %s", path)
    return audio


def _cached_audio(
    text: str,
    language: ModernLanguage,
    voice_id: str,
    model: str,
    output_format: str,
    seed: int | None,
) -> bytes:
    settings = config.current()
    key = _cache_key(text, language, voice_id, model, output_format, seed)
    suffix = _suffix(output_format)
    if settings.cache:
        cached = _cache_load(settings.audio_cache_dir, key, suffix)
        if cached is not None:
            log.info("audio cache hit")
            return cached
    audio = _request(text, language, voice_id, model, output_format, seed)
    if settings.cache and audio:
        _cache_save(settings.audio_cache_dir, key, suffix, audio)
    return audio


def _cache_key(
    text: str,
    language: ModernLanguage,
    voice_id: str,
    model: str,
    output_format: str,
    seed: int | None,
) -> str:
    request = {
        "voice_id": voice_id,
        "output_format": output_format,
        "body": _request_body(text, language, model, seed),
    }
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sends_language(model: str) -> bool:
    """`eleven_multilingual_v2` rejects, in practice ignores, language_code."""
    return "multilingual_v2" not in model


def _request_body(
    text: str,
    language: ModernLanguage,
    model: str,
    seed: int | None,
) -> dict[str, Any]:
    """The body of one speech request, at the robust end of the docs.

    Normalization rewrites numbers, symbols, and abbreviations into other
    words, so both normalizers are off. `use_pvc_as_ivc` would swap a
    professional clone for its more expressive instant clone. A seed is
    sent only when the caller asked for one: it pins a take, it does not
    make that take more accurate, and their docs say it is not guaranteed.
    """
    body: dict[str, Any] = {
        "text": text,
        "model_id": model,
        "apply_text_normalization": "off",
        "apply_language_text_normalization": False,
        "use_pvc_as_ivc": False,
        "voice_settings": _voice_settings(model),
    }
    if _sends_language(model):
        body["language_code"] = language.elevenlabs_code
    if seed is not None:
        body["seed"] = seed
    return body


def _voice_settings(model: str) -> dict[str, Any]:
    """The knobs this model accepts.

    Eleven v4 takes stability and similarity only, plus a clone strength
    that applies only to a professional clone fine-tuned for v4. The
    bundled voices are premade, and they recommend 0 when the slider
    does appear, so it is left unset. Eleven v3 takes stability and
    style, which is pinned at 0. Older models also take speaker boost
    and speed: speaker boost on, speed 1.
    """
    settings = config.current()
    name = model.lower()
    if "v4" in name:
        return {
            "stability": settings.stability,
            "similarity_boost": settings.similarity_boost,
        }
    if "v3" in name:
        return {
            "stability": settings.stability,
            "style": 0.0,
        }
    return {
        "stability": settings.stability,
        "similarity_boost": settings.similarity_boost,
        "style": 0.0,
        "use_speaker_boost": True,
        "speed": 1.0,
    }


def _request(
    text: str,
    language: ModernLanguage,
    voice_id: str,
    model: str,
    output_format: str,
    seed: int | None,
) -> bytes:
    settings = config.current()
    body = _request_body(text, language, model, seed)
    headers = {
        "xi-api-key": settings.elevenlabs_api_key or "",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg" if output_format.startswith("mp3") else "application/octet-stream",
    }
    url = URL.format(voice_id=voice_id)
    log.info("ElevenLabs %s voice %s (%s)", model, voice_id, language.code)
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=body, params={"output_format": output_format})
    except httpx.HTTPError as exc:
        raise TTSError(f"ElevenLabs request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = (response.text or response.reason_phrase or "").strip()
        raise TTSError(f"ElevenLabs failed ({response.status_code}): {detail[:400]}")
    audio = response.content
    if not audio or audio[:1] in (b"{", b"["):
        detail = audio.decode("utf-8", errors="replace")[:400] if audio else "(empty)"
        raise TTSError(f"ElevenLabs returned no audio: {detail}")
    return audio


def _suffix(output_format: str) -> str:
    kind = output_format.split("_", 1)[0]
    return {
        "mp3": ".mp3",
        "pcm": ".pcm",
        "ulaw": ".ulaw",
        "opus": ".opus",
        "wav": ".wav",
    }.get(kind, ".audio")


def _cache_load(directory: Path, key: str, suffix: str) -> bytes | None:
    path = directory / f"{key}{suffix}"
    try:
        if path.is_file():
            data = path.read_bytes()
            if data:
                return data
    except OSError:
        return None
    return None


def _cache_save(directory: Path, key: str, suffix: str, data: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}{suffix}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
