"""Koine Greek spelling → Modern Greek respelling, in one pass.

No IPA. The letters go straight to the spelling that `respell_el` gives
for the first-century sounds: η αι → ε, ει υ οι υι → ι, ω → ο, φ θ χ →
π τ κ, δ → ντ, αυ ευ ηυ → αβ/αφ εβ/εφ, no breathing, no geminates.
"""

from __future__ import annotations

from ._greek import LABIAL_DIPHTHONGS, Unit, Word, join, tokenize
from .respell_el import DIAERESIS_AFTER, TONOS, finish

_VOWELS = {
    "α": "α", "ε": "ε", "η": "ε", "ι": "ι", "ο": "ο", "ω": "ο", "υ": "ι",
    "αι": "ε", "ει": "ι", "οι": "ι", "ου": "ου", "υι": "ι",
    "αυ": "α", "ευ": "ε", "ηυ": "ε",
}
_CONSONANTS = {
    "β": "β", "γ": "γ", "δ": "ντ", "ζ": "ζ", "θ": "τ", "κ": "κ", "λ": "λ",
    "μ": "μ", "ν": "ν", "ξ": "ξ", "π": "π", "ρ": "ρ", "σ": "σ", "τ": "τ",
    "φ": "π", "χ": "κ", "ψ": "ψ",
}
_VOICED = frozenset("βγδζλμνρ")
_SIBILANT_VOICERS = frozenset("βγδμνλρ")


def to_modern_greek(text: str) -> str:
    """Respell Koine Greek `text` as Modern Greek, with no IPA step."""
    pieces: list[tuple[str, str]] = []
    sentence = True
    for item in tokenize(text):
        if isinstance(item, str):
            mark = ";" if item == "?" else item
            pieces.append(("p", mark))
            sentence = sentence or mark in ".!;"
            continue
        spelled = word_el(item)
        if spelled and sentence:
            spelled = spelled[0].upper() + spelled[1:]
            sentence = False
        pieces.append(("w", spelled))
    return join(pieces)


def _next(units: list[Unit], i: int) -> Unit | None:
    for u in units[i + 1:]:
        if u.kind != "b":
            return u
    return None


def word_el(word: Word) -> str:
    units = [u for u in word.units if u.kind != "b"]
    stress = word.units[word.stress] if word.stress is not None else None
    out: list[str] = []
    prev_vowel = ""
    for i, u in enumerate(units):
        nxt = _next(units, i)
        if u.kind == "v":
            spelled = _VOWELS[u.text]
            if spelled == "ι" and prev_vowel and prev_vowel[-1] in DIAERESIS_AFTER:
                spelled = "ϊ"
            prev_vowel = spelled
            if u is stress:
                spelled = TONOS[spelled]
            out.append(spelled)
            if u.text in LABIAL_DIPHTHONGS:
                voiced = nxt is not None and (nxt.kind == "v" or nxt.text in _VOICED)
                out.append("β" if voiced else "φ")
                prev_vowel = ""
            continue
        prev_vowel = ""
        c = u.text
        n = nxt.text if nxt is not None else ""
        if c == "ν" and n in ("π", "β", "φ", "ψ", "μ"):
            out.append("μ")
        elif c == "ν" and n in ("κ", "γ", "χ", "ξ"):
            out.append("γ")
        elif c in ("σ", "ζ") and n in _SIBILANT_VOICERS:
            out.append("σ")
        elif c == "β" and out and out[-1] == "μ":
            out.append("π")
        elif c in ("κ", "π") and n == "σ":
            out.append("ξ" if c == "κ" else "ψ")
            units[i + 1] = Unit("c", "")
        elif not c:
            continue
        else:
            out.append(_CONSONANTS[c])
    return finish("".join(out))
