"""How far the IPA heard in the audio is from the IPA that was meant.

    from ancient_language_tts import check

    check.mismatch(expected_tokens, heard_tokens)          # 0.12
    check.compare("ˈbɔ.re.as", "bɔreaz", source="lat-ecc").ops
    check.compare(expected_tokens, heard_tokens).breakdown()
    # [Mistake(expected="t", heard=None, penalty=1.0, times=4, contribution=0.009), ...]

Both sides are reduced to their phonemes: stress marks, syllable dots,
spaces, line breaks and punctuation are dropped (the recognizer hears none
of them). The two phoneme strings are then aligned by a weighted edit
distance:

* inserting or deleting a phoneme costs 1,
* substituting one phoneme for another costs 1, or less when the two are
  close sounds listed in `data/similar_phonemes.json` (see that file for
  how a penalty is composed from a base pair and diacritics),
* a labialized consonant on one side may match that consonant plus a
  [u]/[w] glide on the other, as one step: [kʷ] heard as `k u` is the same
  sound, since the recognizer writes the glide as a vowel. The step costs
  what the consonant and the glide cost on their own (0 for `k u`).

The mismatch is that distance divided by the number of expected phonemes:
0 for a perfect match, about 1 when nothing matches, and possibly above 1
when the audio has many extra sounds.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .. import ipa_tokens

_DATA = Path(__file__).resolve().parent.parent / "data"
SIMILAR_FILE = _DATA / "similar_phonemes.json"

INDEL = 1.0
_LABIAL = "ʷ"
_GLIDES = ("u", "w")  # what the [w] of a labialized consonant may be heard as
_SAME = {"ɡ": "g"}  # script g and plain g are one sound


@dataclass(frozen=True)
class Op:
    """One step of the alignment: `expected` → `heard` (None on an insertion/deletion)."""

    expected: str | None
    heard: str | None
    cost: float
    # The expected IPA word (phonemes between spaces, specials stripped) the
    # step falls in. An inserted phoneme belongs to the word of the expected
    # phoneme before it (the first word when it comes before everything).
    word: str | None = None


@dataclass(frozen=True)
class Mistake:
    """One kind of mistake: `expected` heard as `heard`, and what it adds to the mismatch.

    `expected` is None for an inserted phoneme, `heard` None for a missing one.
    `contribution` is `penalty * times / len(expected phonemes)`; the
    contributions of all mistakes sum to `Comparison.mismatch`. `words` are
    the expected IPA words the mistake happens in, as (word, times), most
    frequent first.
    """

    expected: str | None
    heard: str | None
    penalty: float      # cost of one occurrence
    times: int
    contribution: float
    words: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self, digits: int = 4) -> dict:
        return {
            "expected": self.expected,
            "heard": self.heard,
            "penalty": round(self.penalty, digits),
            "times": self.times,
            "contribution": round(self.contribution, digits),
            "words": [[word, times] for word, times in self.words],
        }


@dataclass(frozen=True)
class Comparison:
    expected: list[str]           # expected phonemes, specials stripped
    heard: list[str]              # heard phonemes, specials stripped
    distance: float               # weighted edit distance
    ops: list[Op] = field(repr=False)

    @property
    def mismatch(self) -> float:
        """`distance` per expected phoneme (0.0 when nothing was expected and nothing heard)."""
        if not self.expected:
            return 0.0 if not self.heard else float("inf")
        return self.distance / len(self.expected)

    def errors(self) -> list[Op]:
        """The steps that cost something."""
        return [op for op in self.ops if op.cost > 0]

    def breakdown(self) -> list[Mistake]:
        """The mistakes grouped by (expected, heard), largest contribution first."""
        groups: dict[tuple[str | None, str | None], list[Op]] = {}
        for op in self.errors():
            groups.setdefault((op.expected, op.heard), []).append(op)
        n = len(self.expected)
        mistakes = []
        for (e, h), ops in groups.items():
            cost = ops[0].cost
            words = Counter(op.word for op in ops if op.word)
            mistakes.append(Mistake(
                e, h, cost, len(ops),
                cost * len(ops) / n if n else float("inf"),
                sorted(words.items(), key=lambda item: (-item[1], item[0])),
            ))
        return sorted(
            mistakes, key=lambda m: (-m.contribution, -m.times, m.expected or "", m.heard or "")
        )


def phonemes_only(ipa: Sequence[str] | str, source: str | None = None) -> list[str]:
    """The phoneme tokens of `ipa`: no stress, syllable dots, spaces or punctuation.

    `ipa` is a token list, or a written line, which needs `source` to be cut
    into tokens (characters that are no token of `source` are kept on their own).
    """
    return [phoneme for phoneme, _ in _phonemes_with_words(ipa, source)]


def _phonemes_with_words(ipa: Sequence[str] | str, source: str | None) -> list[tuple[str, str]]:
    """Each phoneme of `ipa` with the word it is in (that word's phonemes, joined)."""
    if isinstance(ipa, str):
        if source is None:
            raise TypeError("a written IPA line needs `source` to be cut into tokens")
        tokens, _ = ipa_tokens.parse_with_penalty(ipa, source)
    else:
        tokens = list(ipa)
    words: list[list[str]] = [[]]
    for token in tokens:
        token = unicodedata.normalize("NFC", token)
        if token in (ipa_tokens.SPACE, ipa_tokens.LINE_BREAK) or token.isspace():
            if words[-1]:
                words.append([])
            continue
        token = token.strip()
        if token and token not in ipa_tokens.SPECIAL:
            words[-1].append(token)
    return [(phoneme, "".join(word)) for word in words for phoneme in word]


def substitution_cost(a: str, b: str) -> float:
    """What hearing `b` in place of `a` costs, from 0 to 1."""
    a, b = _canon(a), _canon(b)
    if a == b:
        return 0.0
    pairs, modifiers = _table()
    if (a, b) in pairs:
        return pairs[(a, b)]
    base_a, mods_a = _split(a)
    base_b, mods_b = _split(b)
    if base_a == base_b:
        base = 0.0
    else:
        base = pairs.get((base_a, base_b))
        if base is None:
            return 1.0
    unmatched = (mods_a - mods_b) + (mods_b - mods_a)
    extra = sum(modifiers.get(m, 1.0) * n for m, n in unmatched.items())
    return min(1.0, base + extra)


def compare(
    expected: Sequence[str] | str, heard: Sequence[str] | str, source: str | None = None
) -> Comparison:
    """Align the phonemes of `expected` and `heard` (see the module docstring)."""
    with_words = _phonemes_with_words(expected, source)
    exp = [phoneme for phoneme, _ in with_words]
    word_of = [word for _, word in with_words]
    got = phonemes_only(heard, source)
    n, m = len(exp), len(got)
    # dist[i][j]: cost of turning exp[:i] into got[:j]
    dist = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i * INDEL
    for j in range(1, m + 1):
        dist[0][j] = j * INDEL
    for i in range(1, n + 1):
        row, prev = dist[i], dist[i - 1]
        for j in range(1, m + 1):
            best = min(
                prev[j - 1] + substitution_cost(exp[i - 1], got[j - 1]),
                prev[j] + INDEL,
                row[j - 1] + INDEL,
            )
            if j > 1:
                best = min(best, prev[j - 2] + _split_cost(exp[i - 1], got[j - 2], got[j - 1]))
            if i > 1:
                best = min(best, dist[i - 2][j - 1] + _split_cost(got[j - 1], exp[i - 2], exp[i - 1]))
            row[j] = best

    ops: list[Op] = []
    i, j = n, m
    while i or j:
        if i and j > 1:
            cost = _split_cost(exp[i - 1], got[j - 2], got[j - 1])
            if _close(dist[i][j], dist[i - 1][j - 2] + cost):
                ops.append(Op(exp[i - 1], got[j - 2] + got[j - 1], cost, word_of[i - 1]))
                i, j = i - 1, j - 2
                continue
        if i > 1 and j:
            cost = _split_cost(got[j - 1], exp[i - 2], exp[i - 1])
            if _close(dist[i][j], dist[i - 2][j - 1] + cost):
                ops.append(Op(exp[i - 2] + exp[i - 1], got[j - 1], cost, word_of[i - 1]))
                i, j = i - 2, j - 1
                continue
        if i and j:
            cost = substitution_cost(exp[i - 1], got[j - 1])
            if _close(dist[i][j], dist[i - 1][j - 1] + cost):
                ops.append(Op(exp[i - 1], got[j - 1], cost, word_of[i - 1]))
                i, j = i - 1, j - 1
                continue
        if i and _close(dist[i][j], dist[i - 1][j] + INDEL):
            ops.append(Op(exp[i - 1], None, INDEL, word_of[i - 1]))
            i -= 1
        else:
            # inserted after exp[i - 1]: that phoneme's word (the first word when i == 0)
            word = word_of[i - 1] if i else (word_of[0] if word_of else None)
            ops.append(Op(None, got[j - 1], INDEL, word))
            j -= 1
    ops.reverse()
    return Comparison(exp, got, dist[n][m], ops)


def mismatch(
    expected: Sequence[str] | str, heard: Sequence[str] | str, source: str | None = None
) -> float:
    """Weighted edit distance between the phonemes, per expected phoneme."""
    return compare(expected, heard, source).mismatch


def _split_cost(labialized: str, consonant: str, glide: str) -> float:
    """What `consonant` + `glide` costs as the two halves of `labialized` (inf when it is none)."""
    whole = _canon(labialized)
    if not whole.endswith(_LABIAL) or len(whole) < 2:
        return float("inf")
    glide_cost = min(substitution_cost(g, glide) for g in _GLIDES)
    if glide_cost >= 1.0:
        return float("inf")
    return substitution_cost(whole[: -len(_LABIAL)], consonant) + glide_cost


def _close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-9


def _canon(token: str) -> str:
    text = unicodedata.normalize("NFC", token)
    return "".join(_SAME.get(ch, ch) for ch in text)


def _split(token: str) -> tuple[str, Counter]:
    """Base letter and the multiset of diacritics/modifier letters after it."""
    chars = unicodedata.normalize("NFD", token)
    base = unicodedata.normalize("NFC", chars[:1])
    return base, Counter(chars[1:])


@lru_cache(maxsize=1)
def _table() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    raw = json.loads(SIMILAR_FILE.read_text(encoding="utf-8"))
    pairs: dict[tuple[str, str], float] = {}
    for a, b, penalty in raw.get("pairs", []):
        penalty = float(penalty)
        if not 0.0 <= penalty <= 1.0:
            raise ValueError(f"similar_phonemes.json: penalty of {a!r}/{b!r} is outside 0..1")
        a, b = _canon(a), _canon(b)
        pairs[(a, b)] = pairs[(b, a)] = penalty
    modifiers = {
        unicodedata.normalize("NFD", m): float(p) for m, p in raw.get("modifiers", {}).items()
    }
    return pairs, modifiers
