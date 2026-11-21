"""Koine Greek. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

import re

from . import _mock

GRAPHEMES = {
    "ει": "i", "αι": "ɛ", "οι": "y", "ου": "u", "αυ": "av", "ευ": "ev",
    "γγ": "ŋg", "γκ": "ŋk",
    "φ": "pʰ", "θ": "tʰ", "χ": "kʰ", "ψ": "ps", "ξ": "ks",
    "β": "β", "γ": "ɣ", "δ": "d", "ζ": "z",
    "α": "a", "ε": "e", "η": "i", "ι": "i", "ο": "o", "υ": "y", "ω": "o",
    "κ": "k", "λ": "l", "μ": "m", "ν": "n", "π": "p", "ρ": "r",
    "σ": "s", "ς": "s", "τ": "t",
}

TO_MODERN_GREEK = {
    "pʰ": "φ", "tʰ": "θ", "kʰ": "χ", "ps": "ψ", "ks": "ξ",
    "ŋg": "γγ", "ŋk": "γκ",
    "β": "β", "ɣ": "γ", "ð": "δ", "d": "ντ", "z": "ζ",
    "a": "α", "e": "ε", "ɛ": "ε", "i": "ι", "o": "ο", "y": "υ", "u": "ου",
    "k": "κ", "l": "λ", "m": "μ", "n": "ν", "p": "π", "r": "ρ",
    "s": "σ", "t": "τ", "v": "β",
}


def _final_sigma(line: str) -> str:
    return re.sub(r"σ(?![\w])", "ς", line)


ENGINE = _mock.engine(
    "grc",
    GRAPHEMES,
    {"el": TO_MODERN_GREEK},
    finish={"el": _final_sigma},
)
