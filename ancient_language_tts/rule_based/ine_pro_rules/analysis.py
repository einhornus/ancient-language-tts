"""Proto-Indo-European spelling → phonemes, syllables, and stress.

Shared by `ipa` (phonemes → IPA) and the `direct_*` respellers (phonemes →
a modern spelling, with no IPA string in between). Implements the parts of
`data/ine-pro/phonology.txt` that do not depend on the output alphabet:
reading the notations, the resonant and laryngeal decisions, laryngeal
coloring, s-voicing, n before velars, syllables, and the accent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Combining marks (NFD).
_ACUTE = "́"
_MACRON = "̄"
_CIRCUMFLEX = "̂"
_INV_BREVE_ABOVE = "̑"
_RING_BELOW = "̥"
_DOT_BELOW = "̣"
_INV_BREVE_BELOW = "̯"

_LARYNGEAL_DIGIT = {"1": "1", "2": "2", "3": "3", "₁": "1", "₂": "2", "₃": "3", "ₓ": "1"}
_APOSTROPHES = "'’ʹ"

# Voiced stops and voiced aspirates: s before them is [z].
VOICED = frozenset({"b", "d", "ǵ", "g", "gʷ", "bʰ", "dʰ", "ǵʰ", "gʰ", "gʷʰ"})
# n before these is [ŋ] (phonology.txt lists k g kʷ gʷ and their aspirates).
VELAR = frozenset({"k", "g", "kʷ", "gʷ", "gʰ", "gʷʰ"})
_ASPIRATE = {"b": "bʰ", "d": "dʰ", "g": "gʰ", "ǵ": "ǵʰ", "gʷ": "gʷʰ"}
_LABIAL = {"k": "kʷ", "g": "gʷ", "gʰ": "gʷʰ"}
_PALATAL = {"k": "ḱ", "g": "ǵ", "gʰ": "ǵʰ"}

# Unaccented words spoken without stress (enclitics, the negative).
UNSTRESSED = frozenset({"ne", "kʷe", "kwe", "we", "wē", "ge"})

PAUSES = frozenset(",!?.")
_WORD_CHARS = r"\ẁ-ͯʰ-˿₀-ₜ'’\-\*"
_TOKEN_RE = re.compile(rf"[{_WORD_CHARS}]+|[^\s]")


@dataclass
class Ph:
    """One phoneme. Kinds: V vowel, R resonant, H laryngeal, C obstruent."""

    kind: str
    base: str  # V: a e i o u; R: r l m n y w; H: 1 2 3; C: p t ḱ k kʷ b d ǵ g gʷ bʰ … s
    long: bool = False
    accent: bool = False
    syllabic: bool = False  # R and H
    written_syllabic: bool = False  # R with a ring / dot below, or an acute
    forced_glide: bool = False  # i̯ u̯ written: always non-syllabic
    glide: bool = False  # y/w as the second half of a diphthong
    quality: str = ""  # V: surface quality after laryngeal coloring
    voiced: bool = False  # s → [z]
    velar: bool = False  # n → [ŋ]

    @property
    def nucleus(self) -> bool:
        return self.kind == "V" or (self.kind in "RH" and self.syllabic)

    def key(self) -> tuple:
        """Surface identity, for spotting geminates."""
        return (self.kind, self.base, self.syllabic, self.glide, self.voiced, self.velar)


@dataclass
class Word:
    syllables: list[list[Ph]] = field(default_factory=list)
    stress: int | None = None

    def phonemes(self) -> list[tuple[int, Ph]]:
        return [(i, p) for i, syl in enumerate(self.syllables) for p in syl]


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split a line into ("word", raw) and ("pause", mark) items.

    ; and : become a period. Parentheses, quotes, and other marks drop.
    """
    out: list[tuple[str, str]] = []
    for raw in _TOKEN_RE.findall(unicodedata.normalize("NFC", text.strip())):
        if re.search(r"[^\W\d_]", raw):
            out.append(("word", raw))
            continue
        mark = "." if raw in ";:" else raw
        if mark in PAUSES:
            out.append(("pause", mark))
    return out


def analyze(raw: str) -> Word:
    """Phonemes of one written word, syllabified and stressed."""
    s = unicodedata.normalize("NFD", raw).replace("*", "")
    s = s.strip("'’\"")
    leading = s.startswith("-")
    s = s.strip("-")
    key = unicodedata.normalize("NFC", s.replace("-", "")).lower()
    phs = _parse(s)
    if not phs:
        return Word()
    _glides_from_vowels(phs)
    _resonants(phs)
    _laryngeals(phs)
    _surface(phs)
    syllables = _syllabify(phs)
    stress = _stress(syllables, unstressed=leading or key in UNSTRESSED)
    return Word(syllables, stress)


def words(text: str) -> list[tuple[str, object]]:
    """Tokenize and analyze: ("word", Word) and ("pause", mark) items."""
    out: list[tuple[str, object]] = []
    for kind, value in tokenize(text):
        if kind == "word":
            word = analyze(value)
            if word.syllables:
                out.append(("word", word))
        else:
            out.append(("pause", value))
    return out


# --- reading the notation ---------------------------------------------------

def _parse(s: str) -> list[Ph]:
    phs: list[Ph] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        j = i + 1
        marks = ""
        while j < n and unicodedata.combining(s[j]):
            marks += s[j]
            j += 1
        low = ch.lower()
        last = phs[-1] if phs else None

        if low in "aeiou":
            if low in "iu" and _INV_BREVE_BELOW in marks:
                if low == "u" and last and last.kind == "C" and last.base in _LABIAL:
                    last.base = _LABIAL[last.base]  # ku̯ gu̯ → kʷ gʷ
                else:
                    phs.append(Ph("R", "y" if low == "i" else "w", forced_glide=True))
            else:
                phs.append(Ph("V", low, long=_MACRON in marks, accent=_ACUTE in marks))
        elif low in "ə" or (low == "h" and j < n and s[j] in _LARYNGEAL_DIGIT):
            if j < n and s[j] in _LARYNGEAL_DIGIT:
                phs.append(Ph("H", _LARYNGEAL_DIGIT[s[j]]))
                j += 1
            else:
                phs.append(Ph("H", "1"))  # bare ə: laryngeal of unknown color
        elif low == "h":
            if ch == "h" and last and last.kind == "C" and last.base in _ASPIRATE:
                last.base = _ASPIRATE[last.base]  # older bh dh gh
            else:
                phs.append(Ph("H", "1"))  # H, hₓ: unknown color → h₁
        elif low in "rlmn":
            syll = _RING_BELOW in marks or _DOT_BELOW in marks or _ACUTE in marks
            phs.append(Ph("R", low, accent=_ACUTE in marks, written_syllabic=syll))
        elif low in "yj":
            if _ACUTE in marks:
                phs.append(Ph("V", "i", accent=True))
            else:
                phs.append(Ph("R", "y"))
        elif low == "w":
            phs.append(Ph("R", "w"))
        elif low in "kg":
            base = low
            if _ACUTE in marks or _CIRCUMFLEX in marks or _INV_BREVE_ABOVE in marks:
                base = _PALATAL[base]
            phs.append(Ph("C", base))
        elif low == "q":
            phs.append(Ph("C", "kʷ"))
        elif low in "ptbds":
            phs.append(Ph("C", low))
        elif ch == "ʰ":
            if last and last.kind == "C" and last.base in _ASPIRATE:
                last.base = _ASPIRATE[last.base]
        elif ch == "ʷ":
            if last and last.kind == "C" and last.base in _LABIAL:
                last.base = _LABIAL[last.base]
        elif ch in _APOSTROPHES:
            if last and last.kind == "C" and last.base in _PALATAL:
                last.base = _PALATAL[last.base]
        # hyphens, digits without h, and anything else: no sound
        i = j
    return phs


def _glides_from_vowels(phs: list[Ph]) -> None:
    """Older books: plain i u beside a vowel are the glides y w."""
    def candidate(p: Ph | None) -> bool:
        return p is not None and p.kind == "V" and p.base in "iu" and not p.long and not p.accent

    flips = []
    for k, p in enumerate(phs):
        if not candidate(p):
            continue
        for q in (phs[k - 1] if k else None, phs[k + 1] if k + 1 < len(phs) else None):
            if q is not None and q.kind == "V" and not candidate(q):
                flips.append(k)
                break
    for k in flips:
        phs[k] = Ph("R", "y" if phs[k].base == "i" else "w")


# --- resonants, laryngeals, and surface values -----------------------------

def _neighbors(phs: list[Ph], k: int) -> tuple[Ph | None, Ph | None]:
    return (phs[k - 1] if k else None, phs[k + 1] if k + 1 < len(phs) else None)


def _resonants(phs: list[Ph]) -> None:
    pending: list[int] = []
    for k, p in enumerate(phs):
        if p.kind != "R":
            continue
        if p.written_syllabic:
            p.syllabic = True
        elif p.forced_glide:
            p.syllabic = False
        else:
            pending.append(k)
    # Next to a vowel or a written syllabic resonant: consonant.
    undecided = []
    for k in pending:
        if any(q is not None and q.nucleus for q in _neighbors(phs, k)):
            phs[k].syllabic = False
        else:
            undecided.append(k)
    # The rest, left to right: syllabic unless a neighbor is already a nucleus.
    for k in undecided:
        phs[k].syllabic = not any(q is not None and q.nucleus for q in _neighbors(phs, k))


def _after_vowel_glide(phs: list[Ph], k: int) -> bool:
    """True when phs[k] is y/w right after a vowel (a diphthong's second half)."""
    p = phs[k]
    return p.kind == "R" and p.base in "yw" and not p.syllabic and k > 0 and phs[k - 1].kind == "V"


def _laryngeals(phs: list[Ph]) -> None:
    for k, p in enumerate(phs):
        if p.kind != "H":
            continue
        prev, nxt = _neighbors(phs, k)
        beside = any(q is not None and q.kind in "VR" and q.nucleus for q in (prev, nxt))
        if not beside and k > 0 and _after_vowel_glide(phs, k - 1):
            beside = True  # after a diphthong: part of its syllable
        p.syllabic = not beside


def _surface(phs: list[Ph]) -> None:
    for k, p in enumerate(phs):
        prev, nxt = _neighbors(phs, k)
        if p.kind == "V":
            p.quality = p.base
            if p.base == "e":
                for q in (prev, nxt):
                    if q is not None and q.kind == "H" and q.base in "23":
                        p.quality = "a" if q.base == "2" else "o"
                        break
        elif p.kind == "R":
            if p.base in "yw" and not p.syllabic and prev is not None and prev.kind == "V":
                p.glide = nxt is None or not nxt.nucleus
            if p.base == "n" and not p.syllabic and nxt is not None and nxt.kind == "C":
                p.velar = nxt.base in VELAR
        elif p.kind == "C" and p.base == "s":
            p.voiced = nxt is not None and nxt.kind == "C" and nxt.base in VOICED


# --- syllables and stress ----------------------------------------------------

def _syllabify(phs: list[Ph]) -> list[list[Ph]]:
    nuclei = [k for k, p in enumerate(phs) if p.nucleus]
    if not nuclei:
        return [phs]
    starts = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        between = b - a - 1
        starts.append(b - (1 if between >= 1 else 0))
    bounds = starts + [len(phs)]
    return [phs[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def _stress(syllables: list[list[Ph]], unstressed: bool) -> int | None:
    for i, syl in enumerate(syllables):
        if any(p.accent for p in syl):
            return i
    if unstressed:
        return None
    for i, syl in enumerate(syllables):
        if any(p.kind in "VR" and p.nucleus for p in syl):
            return i
    return None


def nucleus_of(syllable: list[Ph]) -> Ph | None:
    return next((p for p in syllable if p.nucleus), None)
