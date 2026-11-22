"""Turn CTC logits into per-frame phoneme distributions, and those into phonemes.

Each 20 ms frame of the model output is a distribution over ZIPA's
vocabulary (`data/zipa_tokens.json`): single IPA characters plus a blank
("no sound") token. Per frame:

* softmax the logits and fold the special tokens (`<blk>`, `<sos/eos>`,
  `<unk>`, the word marker `▁`) into the blank,
* confidence = 1 - P(blank): how sure the model is that *some* sound is
  emitted here,
* the remaining mass, renormalized, ranks which character it is.

`recognize_phonemes` returns the frames whose confidence is above the
threshold, each with its own distribution (`Frame`). Nothing is merged.

`phonemes_from_frames` turns those frames into phonemes (`Phoneme`): runs
of *consecutive* kept frames with the same top character merge into one
run. A dropped frame (silence or CTC blank) always ends a run, so a
genuinely repeated sound ("ll") stays two phonemes.

ZIPA spells a phone as a base letter followed by modifier characters
(`pʰ` is `p` + `ʰ`, `aː` is `a` + `ː`, `t̪` is `t` + combining bridge). A run
whose top character is such a modifier is attached to the phoneme before
it, so `phoneme` is the whole phone. Affricates and diphthongs are two
phonemes (`t` `ʃ`, `a` `u`), as in the project's `phonemes.json` files.

Pass `source` (an ancient language code) to condition all of this on that
language (see `conditioning.py`): the model's tokens are mapped by
`data/<code>/zipa.json` and sounds the language lacks are never chosen. A
frame's distribution is over those mapped characters, and each merged
`phoneme` is one token of `data/<code>/phonemes.json`.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .. import ipa_tokens
from . import conditioning as lang
from .audio import AudioInput, load_audio
from .model import TICK_MS, get_phoneme_model

BLANK = lang.BLANK
DROP = lang.DROP
DEFAULT_THRESHOLD = 0.2
PAUSE_MS = 300  # `hear` writes a silence at least this long as a space

# ZIPA's tokens.txt (ipa_simplified), id = position. Used when no table is given.
ZIPA_TOKENS = [row["token"] for row in lang.model_tokens()]
SPECIAL_TOKENS = {
    row["token"] for row in lang.model_tokens() if row["type"] != "phone" and row["type"] != "modifier"
} | {"|", " "}

# Characters that modify the phone before them rather than being one.
MODIFIERS = frozenset(row["token"] for row in lang.model_tokens() if row["type"] == "modifier")


class Frame(TypedDict):
    start_ms: int
    end_ms: int
    confidence: float                # 1 - P(blank) of this frame
    probas: list[tuple[str, float]]  # character distribution, blank excluded, sums to ~1


class Phoneme(TypedDict):
    phoneme: str                  # IPA phone: top base character plus any modifiers
    start_ms: int
    end_ms: int
    confidence: float             # mean 1 - P(blank) over the phone's frames
    probas: list[tuple[str, float]]  # base-character distribution, blank excluded, sums to ~1


def recognize_phonemes(
    audio: AudioInput,
    sample_rate: int | None = None,
    *,
    source: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 10,
    model_id: str | None = None,
    model_file: str | None = None,
) -> list[Frame]:
    """The phoneme distribution of every confident frame of `audio`.

    Runs the local ZIPA CTC model. `audio` is a file path, encoded file
    bytes (e.g. what `speak` returns) or a numpy waveform (pass
    `sample_rate` if it is not 16 kHz). Returns, in time order, one `Frame`
    per 20 ms frame whose confidence is above `threshold`. Frames at or
    below it are left out, so a gap in the times is silence or blank. With
    `source` the distributions are conditioned on that ancient language;
    without it they are over the model's own IPA characters. `top_k` caps
    each frame's `probas`. `phonemes_from_frames` merges the frames into
    phonemes.
    """
    waveform = load_audio(audio, sample_rate)
    model = get_phoneme_model(model_id, model_file)
    logits = model.get_logits(waveform)
    return frames_from_logits(
        logits, glyph_table(model.tokens, source), source=source, threshold=threshold, top_k=top_k
    )


def recognize(audio: AudioInput, sample_rate: int | None = None, **kwargs) -> list[str]:
    """The phones heard in `audio`, e.g. ``["a", "r", "m", "aː"]``."""
    frames = recognize_phonemes(audio, sample_rate, **kwargs)
    return [p["phoneme"] for p in phonemes_from_frames(frames, source=kwargs.get("source"))]


def hear(
    audio: AudioInput,
    source: str,
    sample_rate: int | None = None,
    *,
    pause_ms: int = PAUSE_MS,
    **kwargs,
) -> list[str]:
    """The IPA token list heard in `audio`, in the phonology of `source`.

    Like the rest of the package's IPA: a list of tokens of
    `data/<source>/phonemes.json`, with a space token wherever the audio
    pauses for `pause_ms` or longer. The recognizer gives no stress,
    syllable dots or punctuation.

    Built from the frames the way `check.compare` sees them
    (`mismatch.prepare_frames`): repeated frames of one sound collapsed,
    modifier frames attached, then each frame's top glyph, cut into the
    language's tokens.
    """
    from .mismatch import heard_tokens, prepare_frames

    frames = recognize_phonemes(audio, sample_rate, source=source, **kwargs)
    conditioning = lang.language(source)
    tokens: list[str] = []
    for glyph in heard_tokens(prepare_frames(frames), pause_ms=pause_ms):
        tokens.extend([glyph] if glyph == ipa_tokens.SPACE else conditioning.tokenize(glyph))
    return tokens


def tokens_from_phonemes(phonemes: list[Phoneme], *, pause_ms: int = PAUSE_MS) -> list[str]:
    """The `hear` token list of phonemes already recognized."""
    tokens: list[str] = []
    previous_end = None
    for p in phonemes:
        if previous_end is not None and p["start_ms"] - previous_end >= pause_ms:
            tokens.append(ipa_tokens.SPACE)
        tokens.append(p["phoneme"])
        previous_end = p["end_ms"]
    return tokens


def glyph_table(tokens: dict[int, str] | list[str], source: str | None = None) -> dict[int, str]:
    """{id: glyph} from a token table: special tokens map to `BLANK`.

    With `source`, every token is renamed by that language's rules, and a
    token that is no sound of the language maps to `DROP`.
    """
    items = tokens.items() if isinstance(tokens, dict) else enumerate(tokens)
    conditioning = lang.language(source) if source else None
    table = {}
    for i, t in items:
        if t in SPECIAL_TOKENS:
            table[i] = BLANK
        elif conditioning is not None:
            table[i] = conditioning.glyph(t)
        else:
            table[i] = t
    return table


def frames_from_logits(
    logits: np.ndarray,
    glyphs: dict[int, str] | None = None,
    *,
    source: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 10,
    frame_ms: int = TICK_MS,
) -> list[Frame]:
    """The model-free half of `recognize_phonemes`: (time, vocab) logits -> frames."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    glyphs = glyph_table(ZIPA_TOKENS, source) if glyphs is None else glyphs
    names, fold = _fold_matrix(glyphs, logits.shape[-1])

    probs = _softmax(logits) @ fold  # (time, glyph)
    blank = names.index(BLANK)
    confidence = 1.0 - probs[:, blank]
    probs[:, blank] = 0.0
    if DROP in names:
        probs[:, names.index(DROP)] = 0.0  # still "a sound", but never the one chosen
    mass = probs.sum(axis=1)  # 0 when only dropped glyphs had mass

    frames: list[Frame] = []
    for i in np.flatnonzero((confidence > threshold) & (mass > 0)):
        dist = probs[i] / mass[i]
        order = np.argsort(-dist, kind="stable")[:top_k]
        frames.append({
            "start_ms": int(i) * frame_ms,
            "end_ms": (int(i) + 1) * frame_ms,
            "confidence": round(float(confidence[i]), 3),
            "probas": [(names[j], round(float(dist[j]), 3)) for j in order if dist[j] > 1e-4],
        })
    return frames


def phonemes_from_frames(
    frames: list[Frame],
    *,
    source: str | None = None,
    top_k: int = 5,
) -> list[Phoneme]:
    """Merge the frames of `recognize_phonemes` into phonemes.

    Pass the `source` the frames were recognized with, so that each phoneme
    is split into that language's tokens. `top_k` caps each phoneme's
    `probas`.
    """
    runs: list[list[Frame]] = []
    for frame in frames:
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and frame["start_ms"] == last["end_ms"]
            and frame["probas"][0][0] == last["probas"][0][0]
        ):
            runs[-1].append(frame)
        else:
            runs.append([frame])

    result: list[Phoneme] = []
    sizes: list[int] = []  # frames behind each phoneme, for its mean confidence
    for run in runs:
        glyph = run[0]["probas"][0][0]
        if lang.is_modifier(glyph):
            if result:  # a modifier with nothing before it is dropped
                _attach(result[-1], sizes, glyph, run)
            continue
        result.append(_merge_run(run, top_k))
        sizes.append(len(run))
    if source is None:
        return result

    conditioning = lang.language(source)
    split: list[Phoneme] = []
    for phoneme in result:
        for token in conditioning.tokenize(phoneme["phoneme"]):
            split.append({**phoneme, "phoneme": token})
    return split


def phonemes_from_logits(
    logits: np.ndarray,
    glyphs: dict[int, str] | None = None,
    *,
    source: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
    frame_ms: int = TICK_MS,
) -> list[Phoneme]:
    """(time, vocab) logits -> phonemes: `frames_from_logits`, then `phonemes_from_frames`."""
    frames = frames_from_logits(
        logits, glyphs, source=source, threshold=threshold, top_k=max(top_k, 10), frame_ms=frame_ms
    )
    return phonemes_from_frames(frames, source=source, top_k=top_k)


def _merge_run(run: list[Frame], top_k: int) -> Phoneme:
    # Weight each frame's distribution by its confidence.
    dist: dict[str, float] = {}
    for frame in run:
        for name, p in frame["probas"]:
            dist[name] = dist.get(name, 0.0) + p * frame["confidence"]
    total = sum(dist.values()) or 1.0
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])[:top_k]
    probas = [(name, round(p / total, 3)) for name, p in ranked if p / total > 1e-4]
    return {
        "phoneme": run[0]["probas"][0][0],
        "start_ms": run[0]["start_ms"],
        "end_ms": run[-1]["end_ms"],
        "confidence": round(sum(f["confidence"] for f in run) / len(run), 3),
        "probas": probas,
    }


def _attach(phoneme: Phoneme, sizes: list[int], modifier: str, run: list[Frame]) -> None:
    """Append a modifier run to `phoneme`, extending its span and confidence."""
    n_base = sizes[-1]
    total = phoneme["confidence"] * n_base + sum(f["confidence"] for f in run)
    phoneme["phoneme"] += modifier
    phoneme["end_ms"] = run[-1]["end_ms"]
    sizes[-1] = n_base + len(run)
    phoneme["confidence"] = round(total / sizes[-1], 3)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _fold_matrix(glyphs: dict[int, str], vocab: int) -> tuple[list[str], np.ndarray]:
    """A (vocab, n_glyphs) matrix summing vocabulary probabilities into glyphs."""
    names = [BLANK]
    index = {BLANK: 0}
    fold_entries = []
    for i in range(vocab):
        glyph = glyphs.get(i, BLANK)
        if glyph not in index:
            index[glyph] = len(names)
            names.append(glyph)
        fold_entries.append((i, index[glyph]))
    fold = np.zeros((vocab, len(names)))
    for i, j in fold_entries:
        fold[i, j] = 1.0
    return names, fold
