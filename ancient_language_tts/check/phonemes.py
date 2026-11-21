"""Turn CTC logits into a list of recognized phonemes.

Each 20 ms frame of the model output is a distribution over ZIPA's
vocabulary (`data/zipa_tokens.json`): single IPA characters plus a blank
("no sound") token. Per frame:

* softmax the logits and fold the special tokens (`<blk>`, `<sos/eos>`,
  `<unk>`, the word marker `▁`) into the blank,
* confidence = 1 - P(blank): how sure the model is that *some* sound is
  emitted here,
* the remaining mass, renormalized, ranks which character it is.

Frames with confidence above the threshold are kept, and runs of
*consecutive* kept frames with the same top character merge into one run.
A dropped frame (silence or CTC blank) always ends a run, so a genuinely
repeated sound ("ll") stays two phonemes.

ZIPA spells a phone as a base letter followed by modifier characters
(`pʰ` is `p` + `ʰ`, `aː` is `a` + `ː`, `t̪` is `t` + combining bridge). A run
whose top character is such a modifier is attached to the phoneme before
it, so `phoneme` is the whole phone. Affricates and diphthongs are two
phonemes (`t` `ʃ`, `a` `u`), as in the project's `phonemes.json` files.

Pass `source` (an ancient language code) to condition all of this on that
language (see `conditioning.py`): the model's tokens are mapped by
`data/<code>/zipa.json`, sounds the language lacks are never chosen, and
each `phoneme` is one token of `data/<code>/phonemes.json`.
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
DEFAULT_THRESHOLD = 0.5
PAUSE_MS = 300  # `hear` writes a silence at least this long as a space

# ZIPA's tokens.txt (ipa_simplified), id = position. Used when no table is given.
ZIPA_TOKENS = [row["token"] for row in lang.model_tokens()]
SPECIAL_TOKENS = {
    row["token"] for row in lang.model_tokens() if row["type"] != "phone" and row["type"] != "modifier"
} | {"|", " "}

# Characters that modify the phone before them rather than being one.
MODIFIERS = frozenset(row["token"] for row in lang.model_tokens() if row["type"] == "modifier")


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
) -> list[Phoneme]:
    """Recognize the phonemes in `audio` with the local ZIPA CTC model.

    `audio` is a file path, encoded file bytes (e.g. what `speak` returns) or
    a numpy waveform (pass `sample_rate` if it is not 16 kHz). With `source`
    the recognition is conditioned on that ancient language and every
    `phoneme` is one of its tokens; without it the model's own IPA comes
    back. Returns the phonemes in time order; see the module docstring for
    how frames are kept and merged. `top_k` caps each phoneme's `probas`.
    """
    waveform = load_audio(audio, sample_rate)
    model = get_phoneme_model(model_id, model_file)
    logits = model.get_logits(waveform)
    return phonemes_from_logits(
        logits, glyph_table(model.tokens, source), source=source, threshold=threshold, top_k=top_k
    )


def recognize(audio: AudioInput, sample_rate: int | None = None, **kwargs) -> list[str]:
    """Just the phones of `recognize_phonemes`, e.g. ``["a", "r", "m", "aː"]``."""
    return [p["phoneme"] for p in recognize_phonemes(audio, sample_rate, **kwargs)]


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
    """
    phonemes = recognize_phonemes(audio, sample_rate, source=source, **kwargs)
    return tokens_from_phonemes(phonemes, pause_ms=pause_ms)


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


def phonemes_from_logits(
    logits: np.ndarray,
    glyphs: dict[int, str] | None = None,
    *,
    source: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
    frame_ms: int = TICK_MS,
) -> list[Phoneme]:
    """The model-free half of `recognize_phonemes`: (time, vocab) logits -> phonemes."""
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
    top = probs.argmax(axis=1)
    audible = probs.max(axis=1) > 0  # False when only dropped glyphs had mass

    runs: list[tuple[int, int]] = []
    run_start = None
    for i in range(len(probs) + 1):
        kept = i < len(probs) and confidence[i] > threshold and audible[i]
        if run_start is not None and (not kept or top[i] != top[run_start]):
            runs.append((run_start, i))
            run_start = None
        if kept and run_start is None:
            run_start = i

    result: list[Phoneme] = []
    for start, end in runs:
        glyph = names[top[start]]
        if lang.is_modifier(glyph):
            if result:  # a modifier with nothing before it is dropped
                _attach(result[-1], glyph, confidence, start, end, frame_ms)
            continue
        result.append(_merge_run(probs, confidence, names, start, end, top_k, frame_ms))
    for phoneme in result:
        del phoneme["_frames"]
    if source is None:
        return result

    conditioning = lang.language(source)
    split: list[Phoneme] = []
    for phoneme in result:
        for token in conditioning.tokenize(phoneme["phoneme"]):
            split.append({**phoneme, "phoneme": token})
    return split


def _merge_run(probs, confidence, names, start, end, top_k, frame_ms) -> Phoneme:
    # Weight each frame by its confidence.
    weights = confidence[start:end]
    dist = (probs[start:end] * weights[:, None]).sum(axis=0)
    dist /= dist.sum() or 1.0
    order = np.argsort(-dist)[:top_k]
    probas = [(names[j], round(float(dist[j]), 3)) for j in order if dist[j] > 1e-4]
    return {
        "phoneme": probas[0][0],
        "start_ms": start * frame_ms,
        "end_ms": end * frame_ms,
        "confidence": round(float(weights.mean()), 3),
        "probas": probas,
        "_frames": end - start,  # removed once modifiers are attached
    }


def _attach(phoneme, modifier, confidence, start, end, frame_ms) -> None:
    """Append a modifier run to `phoneme`, extending its span and confidence."""
    n_base = phoneme["_frames"]
    n_mod = end - start
    total = phoneme["confidence"] * n_base + float(confidence[start:end].sum())
    phoneme["phoneme"] += modifier
    phoneme["end_ms"] = end * frame_ms
    phoneme["_frames"] = n_base + n_mod
    phoneme["confidence"] = round(total / (n_base + n_mod), 3)


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
