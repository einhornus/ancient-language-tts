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
mispronounces), speaker boost is on (clarity), and speed is the `speed`
setting (default 0.7, the slow end of their 0.7 to 1.2 range). Eleven v4
accepts only stability and similarity, and Eleven v3 accepts stability
and style, so the others, speed included, are not sent to them.

A call is cached under `<cache_dir>/audio/` from the text, the voice, the
model, and these settings. A rerun of an unchanged passage does not pay
for the audio again.

ElevenLabs returns each take at its own level, so the audio is then
normalized: one gain brings the gated speech loudness (BS.1770 blocks,
without the K-weighting filter) to `TARGET_DBFS`, a 5 ms limiter keeps
the peaks under `PEAK_CEILING_DBFS`, and the take is re-encoded in the
format that was requested. The cache keeps the raw take, so changing the
target does not pay for the audio again.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile
from numpy.lib.stride_tricks import sliding_window_view

from . import config, languages
from .errors import TTSError
from .languages import ModernLanguage

log = logging.getLogger("ancient_language_tts")

URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TIMEOUT = 180.0
_SEED_MAX = 4_294_967_295

# Loudness every take is brought to: the mean power of the gated speech
# blocks, in dB below full scale. -23 is the EBU R128 broadcast level, and
# about 95% of ElevenLabs takes reach it with no limiting.
TARGET_DBFS = -23.0
# The limiter keeps every sample under this. It leaves room for the
# overshoot that lossy re-encoding adds.
PEAK_CEILING_DBFS = -1.0
_LIMIT_SECONDS = 0.005
_BLOCK_SECONDS = 0.4
_ABSOLUTE_GATE_DB = -70.0
_RELATIVE_GATE_DB = -10.0
_MIN_GAIN_DB = 0.05


def synthesize(
    text: str,
    target: str = "en",
    output: str | Path | None = None,
    *,
    voice_id: str | None = None,
    seed: int | None = None,
    normalize: bool = True,
    **settings: Any,
) -> bytes:
    """Synthesize `text`, already spelled in `target`. Returns the audio bytes.

    When `output` is set, the bytes are also written there. `voice_id`
    overrides the language's voice and `ELEVENLABS_VOICE_ID` for this call.
    `normalize=False` returns the take at the level ElevenLabs gave it.
    """
    with config.scope(**settings):
        return _synthesize(text, target, output=output, voice_id=voice_id, seed=seed, normalize=normalize)


def _synthesize(
    text: str,
    target: str,
    *,
    output: str | Path | None,
    voice_id: str | None,
    seed: int | None,
    normalize: bool,
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
    if normalize:
        audio = normalize_volume(audio, output_format)
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
    and speed: speaker boost on, speed from the `speed` setting.
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
        "speed": settings.speed,
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


def normalize_volume(audio: bytes, output_format: str) -> bytes:
    """`audio` re-encoded at `TARGET_DBFS` of speech loudness.

    A format libsndfile cannot decode, or a silent take, comes back as it was.
    """
    try:
        layout = _raw_layout(output_format)
        samples, rate = soundfile.read(io.BytesIO(audio), dtype="float64", always_2d=True, **layout)
        if not layout:
            info = soundfile.info(io.BytesIO(audio))
            layout = {"format": info.format, "subtype": info.subtype}
    except (soundfile.LibsndfileError, RuntimeError, TypeError, ValueError) as exc:
        log.warning("audio not normalized: cannot decode %s (%s)", output_format, exc)
        return audio
    level_db = loudness_db(samples, rate)
    if level_db is None:
        return audio
    gain_db = TARGET_DBFS - level_db
    ceiling = 10.0 ** (PEAK_CEILING_DBFS / 20.0)
    if abs(gain_db) < _MIN_GAIN_DB and np.abs(samples).max() <= ceiling:
        return audio
    samples = _limit(samples * 10.0 ** (gain_db / 20.0), rate)
    buffer = io.BytesIO()
    write_options = dict(layout)
    if write_options.get("format") == "MP3":
        write_options.update(_mp3_bitrate(output_format))
    for key in ("samplerate", "channels"):
        write_options.pop(key, None)
    soundfile.write(buffer, samples, rate, **write_options)
    log.info("normalized audio by %+.1f dB", gain_db)
    return buffer.getvalue()


def _limit(samples: np.ndarray, rate: int) -> np.ndarray:
    """Pull every peak under `PEAK_CEILING_DBFS` with a short smoothed gain dip.

    The gain each sample needs is taken as a minimum over ±`_LIMIT_SECONDS`,
    then averaged over half that span, so the dip starts before the peak
    and every sample still gets at least the reduction it needs.
    """
    ceiling = 10.0 ** (PEAK_CEILING_DBFS / 20.0)
    peaks = np.abs(samples).max(axis=1)
    if peaks.max() <= ceiling:
        return samples
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    reach = max(1, int(rate * _LIMIT_SECONDS))
    padded = np.pad(need, reach, constant_values=1.0)
    floor = sliding_window_view(padded, 2 * reach + 1).min(axis=1)
    half = reach // 2
    window = np.full(2 * half + 1, 1.0 / (2 * half + 1))
    gain = np.convolve(np.pad(floor, half, mode="edge"), window, mode="valid")
    log.info("limited peaks by up to %.1f dB", -20.0 * math.log10(float(gain.min())))
    return samples * gain[:, None]


def loudness_db(samples: np.ndarray, rate: int) -> float | None:
    """Gated speech loudness of `(frames, channels)` samples, in dBFS. None for silence.

    BS.1770's gating: mean power over 400 ms blocks with 75% overlap,
    keeping blocks above -70 dBFS and then those within 10 dB of that
    mean, so the pauses between words do not lower it.
    """
    mono = samples.mean(axis=1)
    if mono.size == 0:
        return None
    size = max(1, min(mono.size, int(rate * _BLOCK_SECONDS)))
    hop = max(1, size // 4)
    energy = np.concatenate(([0.0], np.cumsum(mono * mono)))
    starts = np.arange(0, mono.size - size + 1, hop)
    power = (energy[starts + size] - energy[starts]) / size
    power = power[power > 10.0 ** (_ABSOLUTE_GATE_DB / 10.0)]
    if power.size == 0:
        return None
    power = power[power > power.mean() * 10.0 ** (_RELATIVE_GATE_DB / 10.0)]
    return 10.0 * math.log10(float(power.mean()))


def _raw_layout(output_format: str) -> dict[str, Any]:
    """How to read a headerless ElevenLabs format, e.g. `pcm_44100` or `ulaw_8000`."""
    kind, _, rest = output_format.partition("_")
    subtype = {"pcm": "PCM_16", "ulaw": "ULAW", "alaw": "ALAW"}.get(kind)
    if subtype is None:
        return {}
    rate = int(rest.split("_", 1)[0])
    return {"format": "RAW", "subtype": subtype, "samplerate": rate, "channels": 1, "endian": "LITTLE"}


def _mp3_bitrate(output_format: str) -> dict[str, Any]:
    """Re-encode an `mp3_<rate>_<kbps>` take at its bitrate.

    libsndfile maps compression level 0..1 onto 320..8 kbps.
    """
    parts = output_format.split("_")
    try:
        kbps = int(parts[2])
    except (IndexError, ValueError):
        return {}
    level = min(1.0, max(0.0, (320 - kbps) / (320 - 8)))
    return {"bitrate_mode": "CONSTANT", "compression_level": level}


def _suffix(output_format: str) -> str:
    kind = output_format.split("_", 1)[0]
    return {
        "mp3": ".mp3",
        "pcm": ".pcm",
        "ulaw": ".ulaw",
        "alaw": ".alaw",
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
