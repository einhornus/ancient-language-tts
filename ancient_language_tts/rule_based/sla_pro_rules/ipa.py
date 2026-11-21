"""Proto-Slavic (Early Common Slavic) text → IPA, by rule.

Implements `data/sla-pro/phonology.txt`: every jer is a vowel, the liquid
diphthongs are unmetathesized and close their syllable, v is [w], the nasal
vowels stay nasal, syllables are open otherwise, and the accent mark (or the
first syllable) takes ˈ.
"""

from __future__ import annotations

import unicodedata

from .spelling import analyze, tokens

_IPA = {
    "i": "iː", "y": "ɨː", "u": "uː", "ě": "æː", "a": "ɑː",
    "ę": "ẽː", "ǫ": "õː",
    "e": "e", "o": "o", "ь": "ɪ", "ъ": "ʊ",
    "r̥": "r̩", "l̥": "l̩",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "s": "s", "z": "z", "x": "x", "f": "f",
    "š": "ʃ", "ž": "ʒ", "č": "tʃ", "c": "ts", "dz": "dz", "dž": "dʒ",
    "m": "m", "n": "n", "l": "l", "r": "r", "j": "j", "v": "w",
    "ľ": "lʲ", "ň": "nʲ", "ř": "rʲ", "ť": "tʲ", "ď": "dʲ", "ś": "sʲ",
}
_VELARS = frozenset({"k", "g", "x"})


def to_ipa(text: str) -> str:
    """Transcribe Proto-Slavic `text` into IPA."""
    out: list[str] = []
    for kind, value in tokens(text.strip()):
        if kind == "pause":
            if out:
                out[-1] += value
            continue
        word = _word(value)
        if word:
            out.append(word)
    return unicodedata.normalize("NFC", " ".join(out))


def _word(raw: str) -> str:
    word = analyze(raw)
    if word is None:
        return ""
    units = word.units
    phones: list[str] = []
    for i, u in enumerate(units):
        nxt = units[i + 1].letter if i + 1 < len(units) else ""
        if u.letter == "n" and nxt in _VELARS:
            phones.append("ŋ")
        else:
            phones.append(_IPA[u.letter])
    out = ""
    for s, idxs in enumerate(word.syllables):
        body = "".join(phones[i] for i in idxs)
        if s == word.stress:
            out += "ˈ" + body
        elif s:
            out += "." + body
        else:
            out += body
    return out
