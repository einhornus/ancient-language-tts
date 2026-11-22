"""How far the frames heard in the audio are from the IPA that was meant.

    from ancient_language_tts import check

    frames = check.recognize_phonemes(audio, source="lat-ecc")
    result = check.compare(expected_tokens, frames)
    result.mismatch          # 0.12: penalty per unit of expected weight
    result.mistakes          # [Mistake(kind="substituted", expected="t", heard="d", ...), ...]
    result.breakdown()       # the mistakes grouped by (expected, heard), largest first

Inputs: the expected IPA (a token list, or a written line with `source`)
and the recognizer's frames, each a dict with `probas` (a distribution over
glyphs) and `confidence`. Frames from `recognize_phonemes` and merged
phonemes from `phonemes_from_frames` both work.

Distance. `data/phoneme_distances.json` (built by
`scripts/build_phoneme_distances.py`) gives a distance from 0 to 1 between
an expected phoneme and a glyph; a pair it lacks is 1. The distance from a
frame to a phoneme is the expected distance under the frame's
distribution: sum of P(glyph) * distance(glyph, phoneme).

Collapsing. The recognizer often emits one sound in two or more adjacent
frames, and at a low threshold those frames are often split by a dropped
one. Before aligning, a run of frames with the same top glyph that touch in
time, or are at most `REPEAT_GAP_MS` apart when not both confident (see
`collapse_frames`), becomes one frame: its
distribution is the confidence-weighted mean of theirs, its confidence the
highest of theirs. Pass `collapse=False` to skip this.

Modifiers. ZIPA writes a modifier (ː, ʰ, ʷ …) as a token of its own, so a
long vowel is a vowel frame followed by a ː frame. A frame whose top glyph
is a modifier is never a sound: it is folded into the frame before, which
then gives the modified glyph (aː) the probability the modifier was heard
with (`attach_modifiers`). This happens with or without collapsing.

Alignment. A dynamic program finds the order-keeping one-to-one matching
of frames to expected phonemes with the least total cost: a matched pair
costs its distance, and every phoneme or frame left unmatched costs
`UNMATCHED` (3).

Mistakes. Each expected phoneme has a weight: `WEIGHT_CONSONANT` (1),
`WEIGHT_VOWEL` (2) for an unstressed vowel, `WEIGHT_STRESSED` (4) for a
vowel of a syllable marked with ˈ. Then

* a matched pair costs its distance * weight,
* an expected phoneme left unmatched costs 3 * weight,
* a frame left unmatched costs 0.5 * sqrt(confidence).

The mismatch is the summed penalty divided by the summed weight of the
expected phonemes: 0 for a perfect match, 3 when nothing was heard.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import ipa_tokens

_DATA = Path(__file__).resolve().parent.parent / "data"
DISTANCES_FILE = _DATA / "phoneme_distances.json"

UNMATCHED = 3.0           # alignment cost of a phoneme or frame left unmatched
MISSING_PENALTY = 3.0     # times the weight of an expected phoneme left unmatched
EXTRA_PENALTY = 0.5       # times sqrt(confidence) of a frame left unmatched
UNKNOWN_DISTANCE = 1.0    # a pair the table lacks
REPEAT_GAP_MS = 60        # collapse_frames bridges a gap of dropped frames up to this long
REPEAT_CONFIDENCE = 0.5   # ... unless both sides are this sure (a real repeat)
PAUSE_MS = 300            # heard_tokens writes a gap at least this long as a space
MODIFIER_STEP = 0.15      # a modifier one side has and the other lacks, outside the table
WEIGHT_CONSONANT = 1.0
WEIGHT_VOWEL = 2.0
WEIGHT_STRESSED = 4.0

VOWEL_LETTERS = frozenset("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ")
_SAME = {"ɡ": "g"}  # script g and plain g are one sound
# What ends a stressed syllable, besides a consonant after its vowel.
_SYLLABLE_END = frozenset(ipa_tokens.SPECIAL) - {ipa_tokens.STRESS}

MATCHED, SUBSTITUTED, MISSING, EXTRA = "matched", "substituted", "missing", "extra"


@dataclass(frozen=True)
class Mistake:
    """One step of the alignment that costs something.

    `kind` is "substituted" (an expected phoneme matched to a frame whose
    top glyph is another one), "matched" (top glyph right, but the
    distribution still pays a little), "missing" (expected phoneme with no
    frame) or "extra" (frame with no expected phoneme). `heard` is the
    frame's top glyph; `expected` is None for an extra frame, `heard` None
    for a missing phoneme. `word` is the expected IPA word the step falls
    in (an extra frame: that of the phoneme before it).
    """

    kind: str
    expected: str | None
    heard: str | None
    penalty: float
    weight: float | None = None       # of the expected phoneme
    distance: float | None = None     # frame-to-phoneme distance of a pair
    confidence: float | None = None   # of the frame
    expected_index: int | None = None
    frame_index: int | None = None
    start_ms: int | None = None
    word: str | None = None


@dataclass(frozen=True)
class MistakeGroup:
    """The mistakes of one (expected, heard), and what they add to the mismatch.

    `penalty` is the mean penalty of one occurrence, `total` their sum, and
    `contribution` is `total / Comparison.weight`; the contributions of all
    groups sum to `Comparison.mismatch`. `words` are the expected IPA words
    the mistakes happen in, as (word, times), most frequent first.
    """

    expected: str | None
    heard: str | None
    penalty: float
    times: int
    total: float
    contribution: float
    words: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self, digits: int = 4) -> dict:
        return {
            "expected": self.expected,
            "heard": self.heard,
            "penalty": round(self.penalty, digits),
            "times": self.times,
            "total": round(self.total, digits),
            "contribution": round(self.contribution, digits),
            "words": [[word, times] for word, times in self.words],
        }


@dataclass(frozen=True)
class Comparison:
    expected: list[str]                 # expected phonemes, specials stripped
    weights: list[float]                # weight of each expected phoneme
    heard: list[str]                    # top glyph of each (collapsed) frame
    pairs: list[tuple[int | None, int | None]] = field(repr=False)  # (expected i, frame j)
    mistakes: list[Mistake] = field(repr=False)
    frames: list[dict] = field(default_factory=list, repr=False)  # as aligned (collapsed)

    @property
    def penalty(self) -> float:
        """Summed penalty of all mistakes."""
        return sum(m.penalty for m in self.mistakes)

    distance = penalty  # the name the test harness reads

    @property
    def weight(self) -> float:
        return sum(self.weights)

    @property
    def mismatch(self) -> float:
        """`penalty` per unit of expected weight (0.0 when nothing was expected nor heard)."""
        if not self.weights:
            return 0.0 if not self.heard else float("inf")
        return self.penalty / self.weight

    def heard_tokens(self, pause_ms: int = PAUSE_MS) -> list[str]:
        """The top glyph of each aligned frame, with a space at each pause (see `heard_tokens`)."""
        return heard_tokens(self.frames, pause_ms=pause_ms)

    def breakdown(self) -> list[MistakeGroup]:
        """The mistakes grouped by (expected, heard), largest contribution first."""
        groups: dict[tuple[str | None, str | None], list[Mistake]] = {}
        for m in self.mistakes:
            groups.setdefault((m.expected, m.heard), []).append(m)
        weight = self.weight
        result = []
        for (e, h), items in groups.items():
            total = sum(m.penalty for m in items)
            words = Counter(m.word for m in items if m.word)
            result.append(MistakeGroup(
                e, h, total / len(items), len(items), total,
                total / weight if weight else float("inf"),
                sorted(words.items(), key=lambda item: (-item[1], item[0])),
            ))
        return sorted(
            result, key=lambda g: (-g.contribution, -g.times, g.expected or "", g.heard or "")
        )


# ------------------------------------------------------------------ expected


def phonemes_only(ipa: Sequence[str] | str, source: str | None = None) -> list[str]:
    """The phoneme tokens of `ipa`: no stress, syllable dots, spaces or punctuation.

    `ipa` is a token list, or a written line, which needs `source` to be cut
    into tokens (characters that are no token of `source` are kept on their own).
    """
    return [p for p, _, _ in _expected(ipa, source)]


def is_vowel(phoneme: str) -> bool:
    return _canon(phoneme)[:1] in VOWEL_LETTERS


def phoneme_weights(ipa: Sequence[str] | str, source: str | None = None) -> list[float]:
    """The weight of each phoneme of `phonemes_only(ipa)` (see the module docstring)."""
    return [w for _, w, _ in _expected(ipa, source)]


def _expected(ipa: Sequence[str] | str, source: str | None) -> list[tuple[str, float, str]]:
    """Each phoneme of `ipa` with its weight and the word it is in."""
    if isinstance(ipa, str):
        if source is None:
            raise TypeError("a written IPA line needs `source` to be cut into tokens")
        tokens, _ = ipa_tokens.parse_with_penalty(ipa, source)
    else:
        tokens = list(ipa)
    words: list[list[tuple[str, float]]] = [[]]
    stressed = False      # inside a syllable marked ˈ
    nucleus = False       # that syllable's vowel has been seen
    for token in tokens:
        token = unicodedata.normalize("NFC", token)
        if token in (ipa_tokens.SPACE, ipa_tokens.LINE_BREAK) or token.isspace():
            if words[-1]:
                words.append([])
            stressed = False
            continue
        token = token.strip()
        if not token:
            continue
        if token == ipa_tokens.STRESS:
            stressed, nucleus = True, False
            continue
        if token in _SYLLABLE_END:
            stressed = False
            continue
        if is_vowel(token):
            weight = WEIGHT_STRESSED if stressed else WEIGHT_VOWEL
            nucleus = nucleus or stressed
        else:
            if stressed and nucleus:
                stressed = False  # a consonant after the stressed vowel: syllable over
            weight = WEIGHT_CONSONANT
        words[-1].append((token, weight))
    return [
        (phoneme, weight, "".join(p for p, _ in word))
        for word in words
        for phoneme, weight in word
    ]


# ----------------------------------------------------------------- distance


def distance(phoneme: str, glyph: str) -> float:
    """Distance from 0 to 1 between an expected phoneme and a heard glyph."""
    a, b = _canon(phoneme), _canon(glyph)
    if a == b:
        return 0.0
    table = _table()
    row = table.get(a)
    if row is not None and b in row:
        return row[b]
    row = table.get(b)
    if row is not None and a in row:
        return row[a]
    # A glyph built by attaching a modifier frame ("nː", "eːː") may be in no
    # pair: compare the letters without their modifiers, plus a step per
    # modifier one side has and the other lacks.
    core_a, mods_a = _strip_modifiers(a)
    core_b, mods_b = _strip_modifiers(b)
    if (core_a, mods_a) == (a, Counter()) and (core_b, mods_b) == (b, Counter()):
        return UNKNOWN_DISTANCE
    if not core_a or not core_b:
        return UNKNOWN_DISTANCE
    unmatched = sum(((mods_a - mods_b) + (mods_b - mods_a)).values())
    return min(1.0, distance(core_a, core_b) + MODIFIER_STEP * unmatched)


def _strip_modifiers(token: str) -> tuple[str, Counter]:
    """`token` without its modifier characters, and those characters."""
    core, mods = [], Counter()
    for ch in unicodedata.normalize("NFD", token):
        if _is_modifier(ch) and core:
            mods[ch] += 1
        else:
            core.append(ch)
    return unicodedata.normalize("NFC", "".join(core)), mods


def frame_distance(frame: Mapping, phoneme: str) -> float:
    """Expected distance from `phoneme` to the glyph drawn from `frame["probas"]`."""
    probas = list(frame["probas"])
    mass = sum(p for _, p in probas)
    if mass <= 0:
        return UNKNOWN_DISTANCE
    return sum(p * distance(phoneme, g) for g, p in probas) / mass


def _cost_matrix(expected: list[str], frames: Sequence[Mapping]) -> np.ndarray:
    """(expected, frame) matrix of frame-to-phoneme distances."""
    glyphs: dict[str, int] = {}
    rows = []
    for frame in frames:
        row: dict[int, float] = {}
        for g, p in frame["probas"]:
            k = glyphs.setdefault(_canon(g), len(glyphs))
            row[k] = row.get(k, 0.0) + float(p)
        rows.append(row)
    probs = np.zeros((len(frames), len(glyphs)))
    for j, row in enumerate(rows):
        for k, p in row.items():
            probs[j, k] = p
    mass = probs.sum(axis=1, keepdims=True)
    empty = mass[:, 0] <= 0
    probs = np.divide(probs, mass, out=np.zeros_like(probs), where=mass > 0)
    names = list(glyphs)
    dist = np.array([[distance(e, g) for g in names] for e in expected]).reshape(
        len(expected), len(names)
    )
    cost = dist @ probs.T
    cost[:, empty] = UNKNOWN_DISTANCE
    return cost


# ---------------------------------------------------------------- alignment


def align(cost: np.ndarray, gap: float = UNMATCHED) -> list[tuple[int | None, int | None]]:
    """Order-keeping one-to-one matching of rows to columns with the least total cost.

    `cost[i, j]` is the cost of matching row i with column j; every row or
    column left unmatched costs `gap`. Returns the steps in order as
    (i, j), (i, None) or (None, j).
    """
    n, m = cost.shape
    dp = np.empty((n + 1, m + 1))
    dp[0, :] = gap * np.arange(m + 1)
    dp[:, 0] = gap * np.arange(n + 1)
    for i in range(1, n + 1):
        prev, row, c = dp[i - 1], dp[i], cost[i - 1]
        diag = prev[:-1] + c          # match i-1 with each j-1
        up = prev[1:] + gap           # leave row i-1 unmatched
        best = np.minimum(diag, up)
        # leave column j-1 unmatched: row[j] = min(best[j-1], row[j-1] + gap), a running scan
        for j in range(1, m + 1):
            left = row[j - 1] + gap
            b = best[j - 1]
            row[j] = b if b <= left else left

    steps: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and _close(dp[i, j], dp[i - 1, j - 1] + cost[i - 1, j - 1]):
            steps.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and _close(dp[i, j], dp[i - 1, j] + gap):
            steps.append((i - 1, None))
            i -= 1
        else:
            steps.append((None, j - 1))
            j -= 1
    steps.reverse()
    return steps


def collapse_frames(frames: Sequence[Mapping]) -> list[dict]:
    """Merge each run of adjacent frames with the same top glyph into one frame.

    Frames are adjacent when one starts where the other ends, or, when
    either has no times, when they follow each other in the list. A gap of
    dropped frames up to `REPEAT_GAP_MS` is bridged too, unless the run so
    far and the next frame are both at least `REPEAT_CONFIDENCE` sure: two
    confident peaks with blank between is how CTC writes a real repeat
    ("ll"), while a weak frame beside a strong one is the same sound fading
    in or out. The merged frame spans the run, its `probas` are the
    confidence-weighted mean of the run's (sorted, summing to 1) and its
    `confidence` is the run's highest. `frames` (a count) says how many
    frames it stands for.
    """
    runs: list[list[Mapping]] = []
    for frame in frames:
        run = runs[-1] if runs else None
        last = run[-1] if run else None
        if last is not None and _top(frame) is not None and _top(frame) == _top(last):
            end, start = last.get("end_ms"), frame.get("start_ms")
            if end is None or start is None or start == end:
                run.append(frame)
                continue
            strong = (
                max(_confidence(f) for f in run) >= REPEAT_CONFIDENCE
                and _confidence(frame) >= REPEAT_CONFIDENCE
            )
            if 0 <= start - end <= REPEAT_GAP_MS and not strong:
                run.append(frame)
                continue
        runs.append([frame])
    merged = []
    for run in runs:
        if len(run) == 1:
            merged.append({**run[0], "frames": 1})
            continue
        dist: dict[str, float] = {}
        for frame in run:
            w = max(_confidence(frame), 1e-6)
            mass = sum(p for _, p in frame["probas"]) or 1.0
            for g, p in frame["probas"]:
                dist[g] = dist.get(g, 0.0) + w * p / mass
        total = sum(dist.values()) or 1.0
        merged.append({
            **run[0],
            "end_ms": run[-1].get("end_ms"),
            "confidence": max(_confidence(f) for f in run),
            "probas": sorted(((g, p / total) for g, p in dist.items()), key=lambda gp: -gp[1]),
            "frames": len(run),
        })
    return merged


def attach_modifiers(frames: Sequence[Mapping]) -> list[dict]:
    """Fold each frame whose top glyph is a modifier (ː, ʰ, ʷ …) into the frame before.

    ZIPA writes a modifier as a token of its own, so a long vowel comes out
    as a vowel frame and then a ː frame. A modifier frame says "the sound
    before has this modifier" with probability P(modifier) = confidence *
    its share of the frame's distribution; the frame before is split
    accordingly: each glyph g of probability p becomes g+modifier with
    p * P(modifier) and stays g with the rest. The modifier frame's other
    glyphs are left out. A modifier frame with no frame before is dropped,
    and so is the modifier share of a sound frame (no glyph of the result
    is a bare modifier).
    """
    out: list[dict] = []
    for frame in frames:
        frame = {"frames": 1, **frame}
        if not _is_modifier(_top(frame) or ""):
            sounds = [(g, p) for g, p in frame["probas"] if not _is_modifier(g)]
            mass = sum(p for _, p in sounds) or 1.0
            frame["probas"] = [(g, p / mass) for g, p in sounds]
            out.append(frame)
            continue
        if not out:
            continue
        prev = out[-1]
        mass = sum(p for _, p in frame["probas"]) or 1.0
        conf = min(max(_confidence(frame), 0.0), 1.0)
        mods = [(g, conf * p / mass) for g, p in frame["probas"] if _is_modifier(g)]
        keep = max(0.0, 1.0 - sum(q for _, q in mods))
        prev_mass = sum(p for _, p in prev["probas"]) or 1.0
        dist: dict[str, float] = {}
        for g, p in prev["probas"]:
            p /= prev_mass
            if _is_modifier(g):  # a stray modifier glyph in a sound frame
                dist[g] = dist.get(g, 0.0) + p
                continue
            dist[g] = dist.get(g, 0.0) + p * keep
            for mod, q in mods:
                dist[g + mod] = dist.get(g + mod, 0.0) + p * q
        prev["probas"] = sorted(dist.items(), key=lambda gp: -gp[1])
        prev["end_ms"] = frame.get("end_ms", prev.get("end_ms"))
        prev["frames"] = prev.get("frames", 1) + frame["frames"]
    return out


def prepare_frames(frames: Sequence[Mapping], *, collapse: bool = True) -> list[dict]:
    """The frames `compare` aligns: collapsed (unless `collapse=False`), modifiers attached."""
    return attach_modifiers(collapse_frames(frames) if collapse else frames)


def heard_tokens(frames: Sequence[Mapping], *, pause_ms: int = PAUSE_MS) -> list[str]:
    """The top glyph of each frame, with a space token where frames are `pause_ms` or more apart.

    Pass frames from `prepare_frames` (or `Comparison.frames`): raw frames
    repeat a sound once per frame.
    """
    tokens: list[str] = []
    previous_end = None
    for frame in frames:
        glyph = _top(frame)
        if glyph is None:
            continue
        start = frame.get("start_ms")
        if previous_end is not None and start is not None and start - previous_end >= pause_ms:
            tokens.append(ipa_tokens.SPACE)
        tokens.append(glyph)
        previous_end = frame.get("end_ms")
    return tokens


def compare(
    expected: Sequence[str] | str,
    frames: Sequence[Mapping],
    source: str | None = None,
    *,
    collapse: bool = True,
) -> Comparison:
    """Align `frames` to the phonemes of `expected` and list the mistakes."""
    items = _expected(expected, source)
    exp = [p for p, _, _ in items]
    weights = [w for _, w, _ in items]
    word_of = [word for _, _, word in items]
    frames = prepare_frames(frames, collapse=collapse)
    heard = [_top(f) for f in frames]
    cost = _cost_matrix(exp, frames)
    pairs = align(cost)

    mistakes: list[Mistake] = []
    last_word = word_of[0] if word_of else None
    for i, j in pairs:
        if i is not None:
            last_word = word_of[i]
        if i is not None and j is not None:
            d = float(cost[i, j])
            penalty = d * weights[i]
            if penalty <= 0:
                continue
            kind = MATCHED if _canon(heard[j] or "") == _canon(exp[i]) else SUBSTITUTED
            mistakes.append(Mistake(
                kind, exp[i], heard[j], penalty, weights[i], d,
                _confidence(frames[j]), i, j, frames[j].get("start_ms"), word_of[i],
            ))
        elif i is not None:
            mistakes.append(Mistake(
                MISSING, exp[i], None, MISSING_PENALTY * weights[i], weights[i],
                expected_index=i, word=word_of[i],
            ))
        else:
            confidence = _confidence(frames[j])
            mistakes.append(Mistake(
                EXTRA, None, heard[j], EXTRA_PENALTY * math.sqrt(max(confidence, 0.0)),
                confidence=confidence, frame_index=j,
                start_ms=frames[j].get("start_ms"), word=last_word,
            ))
    return Comparison(exp, weights, heard, pairs, mistakes, frames)


def mismatch(
    expected: Sequence[str] | str,
    frames: Sequence[Mapping],
    source: str | None = None,
    *,
    collapse: bool = True,
) -> float:
    """Summed penalty of the mistakes per unit of expected weight."""
    return compare(expected, frames, source, collapse=collapse).mismatch


# ------------------------------------------------------------------ helpers


def _top(frame: Mapping) -> str | None:
    probas = list(frame["probas"])
    return max(probas, key=lambda gp: gp[1])[0] if probas else None


def _confidence(frame: Mapping) -> float:
    return float(frame.get("confidence", 1.0))


def _is_modifier(text: str) -> bool:
    """True when `text` is only diacritics/modifier letters (ː, ʰ, ̃ …)."""
    return bool(text) and all(
        unicodedata.category(ch) in ("Mn", "Lm", "Sk") for ch in unicodedata.normalize("NFD", text)
    )


def _close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-9


def _canon(token: str) -> str:
    text = unicodedata.normalize("NFC", token)
    return "".join(_SAME.get(ch, ch) for ch in text)


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, float]]:
    raw = json.loads(DISTANCES_FILE.read_text(encoding="utf-8"))["distances"]
    table: dict[str, dict[str, float]] = {}
    for a, row in raw.items():
        out = table.setdefault(_canon(a), {})
        for b, d in row.items():
            d = float(d)
            if not 0.0 <= d <= 1.0:
                raise ValueError(f"phoneme_distances.json: {a!r}/{b!r} is outside 0..1")
            out[_canon(b)] = d
    return table
