"""Koine Greek (first century AD) text → IPA, by rule.

Implements `data/grc/phonology.txt`: the vowel mergers (ει [i], αι [ɛ],
οι υ [y], η [e]), aspirated φ θ χ, stop δ, fricative β γ, [h] for the
rough breathing, labial αυ ευ ηυ, nasal and sibilant assimilation,
simplified geminates, syllables, and stress on the written accent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._greek import LABIAL_DIPHTHONGS, Unit, Word, join, tokenize

VOWELS = {
    "α": "a", "ε": "ɛ", "η": "e", "ι": "i", "ο": "o", "ω": "o", "υ": "y",
    "αι": "ɛ", "ει": "i", "οι": "y", "ου": "u", "υι": "y",
    "αυ": "a", "ευ": "ɛ", "ηυ": "e",
}
FRONT = frozenset({"ɛ", "e", "i", "y"})
PLAIN = {
    "π": ["p"], "τ": ["t"], "κ": ["k"], "φ": ["pʰ"], "θ": ["tʰ"], "χ": ["kʰ"],
    "δ": ["d"], "ζ": ["z"], "ξ": ["k", "s"], "ψ": ["p", "s"],
    "λ": ["l"], "μ": ["m"], "ρ": ["r"],
}
# Letters before which σ is [z] and the labial diphthong ends in [β].
VOICED_LETTERS = frozenset("βγδζλμνρ")
SIBILANT_VOICERS = frozenset("βγδμνλρ")
# Consonants that open a syllable together with a following λ or ρ: the
# stops, plus the letters β and γ (the old mutes), as in Ἀβραάμ [a.βraˈam].
# The β/ɸ that ends αυ ευ ηυ is not one: it closes its syllable.
STOPS = frozenset({"p", "t", "k", "pʰ", "tʰ", "kʰ", "b", "d", "g", "β", "ɣ"})
LIQUIDS = frozenset({"l", "r"})
_ASPIRATE = {("p", "pʰ"), ("t", "tʰ"), ("k", "kʰ")}


@dataclass
class Seg:
    sym: str
    kind: str  # "v" | "c" | "b"
    unit: int = -1
    offglide: bool = False  # the β/ɸ ending αυ ευ ηυ


def to_ipa(text: str) -> str:
    """Transcribe Koine Greek `text` into first-century IPA."""
    pieces: list[tuple[str, str]] = []
    for item in tokenize(text):
        if isinstance(item, str):
            pieces.append(("p", item))
        else:
            pieces.append(("w", word_ipa(item)))
    return join(pieces)


def word_ipa(word: Word) -> str:
    segs = _segments(word.units)
    syllables = _syllabify(segs)
    out: list[str] = []
    for index, syl in enumerate(syllables):
        stressed = word.stress is not None and any(s.kind == "v" and s.unit == word.stress for s in syl)
        if stressed:
            out.append("ˈ")
        elif index:
            out.append(".")
        out.append("".join(s.sym for s in syl))
    return "".join(out)


def _next(units: list[Unit], i: int) -> Unit | None:
    for u in units[i + 1:]:
        if u.kind != "b":
            return u
    return None


def _segments(units: list[Unit]) -> list[Seg]:
    segs: list[Seg] = []

    def last() -> str:
        for s in reversed(segs):
            if s.kind != "b":
                return s.sym
        return ""

    for i, u in enumerate(units):
        nxt = _next(units, i)
        if u.kind == "b":
            segs.append(Seg("", "b"))
            continue
        if u.kind == "v":
            if u.rough:
                segs.append(Seg("h", "c", i))
            segs.append(Seg(VOWELS[u.text], "v", i))
            if u.text in LABIAL_DIPHTHONGS:
                voiced = nxt is not None and (nxt.kind == "v" or nxt.text in VOICED_LETTERS)
                segs.append(Seg("β" if voiced else "ɸ", "c", i, offglide=True))
            continue
        c = u.text
        if c == "γ":
            prev = units[i - 1].text if i else ""
            if nxt is not None and nxt.text in ("γ", "κ", "χ", "ξ"):
                syms = ["ŋ"]
            elif prev == "γ":
                syms = ["g"]
            elif nxt is not None and nxt.kind == "v" and VOWELS[nxt.text] in FRONT:
                syms = ["ʝ"]
            else:
                syms = ["ɣ"]
        elif c == "ν":
            if nxt is not None and nxt.text in ("π", "β", "φ", "ψ", "μ"):
                syms = ["m"]
            elif nxt is not None and nxt.text in ("κ", "γ", "χ", "ξ"):
                syms = ["ŋ"]
            else:
                syms = ["n"]
        elif c == "σ":
            syms = ["z" if nxt is not None and nxt.text in SIBILANT_VOICERS else "s"]
        elif c == "β":
            syms = ["b" if last() == "m" else "β"]
        else:
            syms = PLAIN[c]
        for sym in syms:
            prev = last()
            if prev and (prev == sym or (prev, sym) in _ASPIRATE) and segs[-1].kind == "c":
                segs.pop()  # a doubled consonant is one consonant
            segs.append(Seg(sym, "c", i))
    return segs


def _syllabify(segs: list[Seg]) -> list[list[Seg]]:
    nuclei = [k for k, s in enumerate(segs) if s.kind == "v"]
    real = [s for s in segs if s.kind != "b"]
    if not nuclei:
        return [real] if real else []
    cuts: list[int] = []  # index in `segs` where a new syllable starts
    for a, b in zip(nuclei, nuclei[1:]):
        between = list(range(a + 1, b))
        forced = [k for k in between if segs[k].kind == "b"]
        if forced:
            cuts.append(forced[0])
            continue
        cons = between
        if len(cons) <= 1:
            cuts.append(cons[0] if cons else b)
        elif (
            segs[cons[-2]].sym in STOPS
            and not segs[cons[-2]].offglide
            and segs[cons[-1]].sym in LIQUIDS
        ):
            cuts.append(cons[-2])
        else:
            cuts.append(cons[-1])
    out: list[list[Seg]] = []
    start = 0
    for cut in cuts + [len(segs)]:
        syl = [s for s in segs[start:cut] if s.kind != "b"]
        out.append(syl)
        start = cut
    return out
