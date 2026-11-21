"""Proto-Germanic spelling → phonological units, syllables, and stress.

This is the part of `data/gem-pro/phonology.txt` that reads the written
form: letters and diacritics, the stop or fricative value of b d g h,
[ŋ] before velars, morpheme boundaries, syllables, and stress. It does not
write IPA. `ipa.py` turns the result into IPA; the `direct_*` modules turn
it into a modern spelling with no IPA step.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace

MACRON = "̄"
CIRCUMFLEX = "̂"
OGONEK = "̨"
RING = "̥"
TILDE = "̃"

_VOWELS = frozenset("aeiou")

# Unstressed function words (prepositions, conjunctions, the negative).
FUNCTION_WORDS = frozenset({
    "in", "ana", "at", "bi", "mid", "fram", "uf", "ut", "und",
    "jah", "ak", "auk", "eþþau", "ne",
})

# Prefixes that are never stressed.
UNSTRESSED_PREFIXES = ("ga", "bi", "fra", "fur")
# Prefixes that are stressed in nouns and adjectives, unstressed in verbs.
VERB_PREFIXES = ("wiþra", "under", "ufar", "umbi", "ana", "uz", "us", "at", "in", "ut")

# Words that look like ga-/bi- + root but are not prefixed.
_NOT_PREFIXED = frozenset({"galaną", "galdrą", "gabaz"})

# Word-initial consonant clusters (legal onsets after a prefix).
_ONSETS = frozenset({
    "bl", "br", "dr", "dw", "fl", "fr", "gl", "gr", "gn", "hl", "hr", "hn",
    "kl", "kr", "kn", "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw",
    "tr", "tw", "þr", "þw", "wl", "wr", "skr", "spr", "str", "skw",
})

_PUNCT = ",.!?;:"
_DROP = "()[]{}\"“”„«»‘’‹›"
TOKEN_RE = re.compile(r"[,.!?;:]|[^\s,.!?;:()\[\]{}\"“”„«»‘’‹›]+")


@dataclass(frozen=True)
class Unit:
    """One sound of the written word.

    `base` is the normalized letter: a e i o u, a diphthong (ai au eu iu oi
    ou ei, the last three only long), or a consonant (p t k b d g f þ s z h
    hw kw gw m n l r j w). `length` is 0 short, 1 long, 2 overlong. `fric`
    is True for the fricative value of b d g gw and the velar value of h hw;
    for n it means [ŋ].
    """

    base: str
    kind: str  # "v" vowel, "d" diphthong, "c" consonant
    length: int = 0
    nasal: bool = False
    close: bool = False
    fric: bool = False

    @property
    def nucleus(self) -> bool:
        return self.kind in ("v", "d")


@dataclass
class Word:
    syllables: list[list[Unit]] = field(default_factory=list)
    stress: int | None = None


def tokens(text: str) -> list[str]:
    """Split a line into words and pause marks (parentheses and quotes go)."""
    text = unicodedata.normalize("NFC", text.strip())
    return TOKEN_RE.findall(text)


def pause(token: str) -> str | None:
    """The pause mark for a punctuation token, or None for a word."""
    if len(token) == 1 and token in _PUNCT:
        return "." if token in ";:" else token
    return None


# --- letters ---------------------------------------------------------------

@dataclass
class _Letter:
    ch: str
    length: int = 0
    nasal: bool = False
    close: bool = False
    ring: bool = False


def _normalize(part: str) -> str:
    s = unicodedata.normalize("NFD", part.lower())
    for old, new in (
        ("ƕ", "hw"), ("hʷ", "hw"), ("kʷ", "kw"), ("gʷ", "gw"), ("q", "kw"),
        ("ƀ", "b"), ("đ", "d"), ("ð", "d"), ("ǥ", "g"), ("ʒ", "g"),
        ("th", "þ"), ("c", "k"), ("v", "w"), ("y", "i"), ("x", "ks"),
    ):
        s = s.replace(old, new)
    return s


def _letters(part: str) -> list[_Letter]:
    out: list[_Letter] = []
    for ch in _normalize(part):
        if unicodedata.combining(ch) or ch in "₂²":
            if not out:
                continue
            last = out[-1]
            if ch == MACRON:
                last.length = max(last.length, 1)
            elif ch == CIRCUMFLEX:
                last.length = 2
            elif ch in (OGONEK, TILDE):
                last.nasal = True
            elif ch == RING:
                last.ring = True
            elif ch in "₂²":
                last.close = True
            continue
        if ch.isalpha() or ch == "þ":
            out.append(_Letter(ch))
    # Syllabic resonants: r̥ l̥ m̥ n̥ are ur ul um un.
    fixed: list[_Letter] = []
    for letter in out:
        if letter.ring:
            fixed.append(_Letter("u"))
            letter.ring = False
        fixed.append(letter)
    out = fixed
    # Older ei for ī.
    merged: list[_Letter] = []
    i = 0
    while i < len(out):
        a = out[i]
        b = out[i + 1] if i + 1 < len(out) else None
        if (
            b is not None and a.ch == "e" and b.ch == "i"
            and not (a.length or a.nasal or a.close or b.length or b.nasal)
        ):
            merged.append(_Letter("i", 1))
            i += 2
            continue
        merged.append(a)
        i += 1
    out = merged
    # an, in, un before h: long nasal vowel, no n.
    result: list[_Letter] = []
    i = 0
    while i < len(out):
        a = out[i]
        if (
            a.ch in "aiu" and not a.length and not a.nasal
            and i + 2 < len(out) and out[i + 1].ch == "n" and out[i + 2].ch == "h"
            and not (i == 0 and a.ch in "iu")  # the prefixes in-, un-
        ):
            result.append(_Letter(a.ch, 1, True))
            i += 2
            continue
        result.append(a)
        i += 1
    return result


def _units(letters: list[_Letter]) -> list[Unit]:
    units: list[Unit] = []
    i = 0
    n = len(letters)
    while i < n:
        a = letters[i]
        b = letters[i + 1] if i + 1 < n else None
        c = a.ch
        if c in _VOWELS:
            if (
                b is not None and b.ch in "iu" and not b.length and not b.nasal
                and not a.nasal and _diphthong(a, b.ch)
            ):
                first = "o" if c == "o" else c
                units.append(Unit(first + b.ch, "d", 1 if (a.length or c == "o") else 0,
                                  close=a.close))
                i += 2
                continue
            length = a.length
            if c == "o" and length == 0:
                length = 1  # o does not occur short
            units.append(Unit(c, "v", length, a.nasal, a.close))
            i += 1
            continue
        if c in "hkg" and b is not None and b.ch == "w":
            units.append(Unit(c + "w", "c"))
            i += 2
            continue
        if c == "f":
            units.append(Unit("f", "c"))
        elif c in "ptkbdgþszhmnlrjw":
            units.append(Unit(c, "c"))
        i += 1
    return units


def _diphthong(a: _Letter, second: str) -> bool:
    if a.length or a.ch == "o":
        return a.ch in "eoa" and True
    return (a.ch + second) in ("ai", "au", "eu", "iu")


def _plain(part: str) -> str:
    """Lowercase, no diacritics except þ; for prefix and word lookups."""
    s = _normalize(part)
    return "".join(ch for ch in s if not unicodedata.combining(ch) and ch not in "₂²")


def _nuclei(part: str) -> int:
    return sum(1 for u in _units(_letters(part)) if u.nucleus)


def _legal_onset(rest: str) -> bool:
    plain = _plain(rest)
    if not plain or plain[0] in _VOWELS:
        return False
    cluster = ""
    for ch in plain:
        if ch in _VOWELS:
            break
        cluster += ch
    if cluster in ("hw", "kw", "gw") or len(cluster) == 1:
        return True
    return cluster in _ONSETS or cluster[:-1] in ("hw", "kw") and cluster[-1] in "lr"


def _split_prefix(part: str) -> tuple[str, str, bool] | None:
    """Detect an unwritten prefix boundary: (prefix, rest, unstressed)."""
    nfc = unicodedata.normalize("NFC", part.lower())
    plain = _plain(part)
    if nfc in _NOT_PREFIXED:
        return None
    for prefix in UNSTRESSED_PREFIXES:
        if plain.startswith(prefix) and nfc.startswith(prefix):
            rest = nfc[len(prefix):]
            if _legal_onset(rest) and _nuclei(rest) >= 2:
                return prefix, rest, True
    if _is_verb(nfc):
        for prefix in VERB_PREFIXES:
            if nfc.startswith(prefix):
                rest = nfc[len(prefix):]
                if _legal_onset(rest) and _nuclei(rest) >= 2:
                    return prefix, rest, True
    # un-, in- before h: the h starts the root.
    for prefix in ("un", "in"):
        if nfc.startswith(prefix + "h") and _nuclei(nfc[2:]) >= 1:
            return prefix, nfc[2:], prefix == "in" and _is_verb(nfc)
    return None


def _is_verb(nfc: str) -> bool:
    """An infinitive: -aną, -janą, -ōną, -ēną."""
    return nfc.endswith(("aną", "ōną", "ēną"))


# --- word ------------------------------------------------------------------

def analyze(raw: str) -> Word:
    """Units, syllables, and stress for one written word."""
    word = unicodedata.normalize("NFC", raw).lstrip("*").lower()
    lead = word.startswith("-")
    trail = word.endswith("-") and len(word.strip("-")) > 0
    word = word.strip("-")
    parts = [p for p in word.split("-") if p]
    if not parts:
        return Word()

    # Morphemes: (text, unstressed_prefix)
    morphs: list[tuple[str, bool]] = []
    first = parts[0]
    if len(parts) > 1:
        unstressed = first in UNSTRESSED_PREFIXES or (
            first in VERB_PREFIXES and _is_verb(parts[-1])
        )
        morphs.append((first, unstressed))
        morphs.extend((p, False) for p in parts[1:])
    else:
        split = _split_prefix(first)
        if split:
            morphs.append((split[0], split[2]))
            morphs.append((split[1], False))
        else:
            morphs.append((first, False))

    units: list[Unit] = []
    starts: list[int] = []  # unit index where each morpheme starts
    for text, _ in morphs:
        starts.append(len(units))
        units.extend(_units(_letters(text)))
    units = _allophones(units, set(starts))
    syllables, morph_syll = _syllabify(units, starts)

    plain_word = _plain(word.replace("-", ""))
    stress: int | None
    if lead or not syllables:
        stress = None
    elif len(morphs) == 1 and plain_word in FUNCTION_WORDS:
        stress = None
    elif trail and len(parts) == 1 and parts[0] in UNSTRESSED_PREFIXES:
        stress = None
    else:
        stress = 0
        for (text, unstressed), syl in zip(morphs, morph_syll):
            if unstressed:
                continue
            stress = syl
            break
        if stress is not None and stress >= len(syllables):
            stress = 0
    return Word(syllables, stress)


def _allophones(units: list[Unit], starts: set[int]) -> list[Unit]:
    out: list[Unit] = []
    n = len(units)
    for i, u in enumerate(units):
        prev = units[i - 1] if i and i not in starts else None
        nxt = units[i + 1] if i + 1 < n and (i + 1) not in starts else None
        initial = i in starts
        b = u.base
        if b == "b":
            stop = initial or (prev and prev.base in ("m", "b")) or (nxt and nxt.base == "b")
            u = replace(u, fric=not stop)
        elif b == "d":
            stop = initial or (prev and prev.base in ("n", "l", "z", "d")) or (nxt and nxt.base == "d")
            u = replace(u, fric=not stop)
        elif b in ("g", "gw"):
            stop = (prev and prev.base == "n") or (prev and prev.base == "g") or (
                nxt and nxt.base == "g" and b == "g"
            )
            u = replace(u, fric=not stop)
        elif b == "h":
            onset = initial and nxt is not None and (nxt.nucleus or nxt.base in ("l", "r", "n", "w"))
            u = replace(u, fric=not onset)
        elif b == "hw":
            u = replace(u, fric=not initial)
        elif b == "n":
            velar = nxt is not None and nxt.base in ("k", "g", "kw", "gw")
            u = replace(u, fric=velar)
        out.append(u)
    return out


def _syllabify(units: list[Unit], starts: list[int]) -> tuple[list[list[Unit]], list[int]]:
    """Split into syllables; return them and the syllable index of each morpheme."""
    nuclei = [i for i, u in enumerate(units) if u.nucleus]
    if not nuclei:
        return ([units] if units else []), [0 for _ in starts]
    boundaries = set(starts[1:])
    cuts: list[int] = []  # start index of each syllable after the first
    for a, b in zip(nuclei, nuclei[1:]):
        inside = [k for k in boundaries if a < k <= b]
        if inside:
            cuts.append(inside[0])
            continue
        between = b - a - 1
        cuts.append(b if between == 0 else b - 1)
    edges = [0] + cuts + [len(units)]
    syllables = [units[edges[k]:edges[k + 1]] for k in range(len(edges) - 1)]
    morph_syll: list[int] = []
    for start in starts:
        index = 0
        for k, edge in enumerate(edges[:-1]):
            if edge <= start:
                index = k
        # a morpheme that starts mid-syllable belongs with the next nucleus
        morph_syll.append(_syllable_of_next_nucleus(start, nuclei, edges, index))
    return syllables, morph_syll


def _syllable_of_next_nucleus(start: int, nuclei: list[int], edges: list[int], default: int) -> int:
    for nucleus in nuclei:
        if nucleus >= start:
            for k in range(len(edges) - 1):
                if edges[k] <= nucleus < edges[k + 1]:
                    return k
    return default
