"""Proto-Germanic text → IPA, by rule.

Implements `data/gem-pro/phonology.txt`: the conventional Wiktionary values,
the stop or fricative value of b d g, [x] for non-initial h, [ŋ] before
velars, nasal and overlong vowels, syllables, and stress on the first
syllable of the root.
"""

from __future__ import annotations

import unicodedata

from . import analysis
from .analysis import Unit, Word

_VOWEL = {"a": "ɑ", "e": "e", "i": "i", "o": "ɔ", "u": "u"}
_LONG = {"a": "ɑ", "e": "ɛ", "i": "i", "o": "ɔ", "u": "u"}
_CONS = {
    "p": "p", "t": "t", "k": "k", "kw": "kʷ", "f": "ɸ", "þ": "θ", "s": "s",
    "z": "z", "m": "m", "l": "l", "r": "r", "j": "j", "w": "w",
}
_FRIC = {
    "b": ("b", "β"), "d": ("d", "ð"), "g": ("g", "ɣ"), "gw": ("gʷ", "ɣʷ"),
    "h": ("h", "x"), "hw": ("hʷ", "xʷ"), "n": ("n", "ŋ"),
}


def unit_ipa(u: Unit) -> str:
    """The IPA for one unit."""
    if u.kind == "c":
        if u.base in _FRIC:
            return _FRIC[u.base][1 if u.fric else 0]
        return _CONS.get(u.base, u.base)
    if u.kind == "d":
        first, second = u.base[0], u.base[1]
        head = _vowel(first, u.length, False, u.close)
        return head + second + "̯"
    return _vowel(u.base, u.length, u.nasal, u.close)


def _vowel(base: str, length: int, nasal: bool, close: bool) -> str:
    if base == "o":
        length = max(length, 1)
    if length == 0:
        sym = _VOWEL[base]
    else:
        sym = _LONG[base]
        if base == "e" and close:
            sym = "e"
            length = 1  # ē₂ and ê₂ are both the close long [eː]
    if nasal:
        sym += "̃"
    return sym + "ː" * length


def word_ipa(word: Word) -> str:
    """Join a word's syllables with dots; ˈ replaces the dot before stress."""
    parts = ["".join(unit_ipa(u) for u in syl) for syl in word.syllables]
    out = ""
    for i, body in enumerate(parts):
        if i == word.stress:
            out += "ˈ" + body
        elif i:
            out += "." + body
        else:
            out += body
    return unicodedata.normalize("NFC", out)


def to_ipa(text: str) -> str:
    """Transcribe Proto-Germanic `text` into IPA."""
    out: list[str] = []
    for token in analysis.tokens(text):
        mark = analysis.pause(token)
        if mark:
            if out:
                out[-1] += mark
            else:
                out.append(mark)
            continue
        spelled = word_ipa(analysis.analyze(token))
        if spelled:
            out.append(spelled)
    return " ".join(out)


# --- IPA reading (shared by the respellers) --------------------------------

_IPA_VOWELS = ("ɑ", "e", "ɛ", "i", "ɔ", "u")
_IPA_CONS = (
    "kʷ", "gʷ", "xʷ", "ɣʷ", "hʷ",
    "p", "b", "t", "d", "k", "g", "ɸ", "β", "θ", "ð", "s", "z", "x", "ɣ",
    "h", "m", "n", "ŋ", "l", "r", "j", "w", "f", "v",
)


def ipa_syllables(ipa_word: str) -> tuple[list[list[str]], int | None]:
    """Split one IPA word into syllables of phone tokens and a stress index.

    A vowel token keeps its length and nasal marks (ɑ̃ː, ɔːː); a diphthong
    is one token (ɑi̯, ɔːu̯).
    """
    s = unicodedata.normalize("NFD", ipa_word)
    syllables: list[list[str]] = [[]]
    stress: int | None = None
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "ˈ":
            if syllables[-1]:
                syllables.append([])
            stress = len(syllables) - 1
            i += 1
            continue
        if ch == ".":
            syllables.append([])
            i += 1
            continue
        if ch in _IPA_VOWELS:
            j = i + 1
            while j < n and (s[j] == "̃" or s[j] == "ː"):
                j += 1
            # diphthong: following i/u with the non-syllabic mark
            if j + 1 < n and s[j] in "iu" and s[j + 1] == "̯":
                j += 2
            syllables[-1].append(unicodedata.normalize("NFC", s[i:j]))
            i = j
            continue
        for tok in _IPA_CONS:
            if s.startswith(tok, i):
                syllables[-1].append(tok)
                i += len(tok)
                break
        else:
            i += 1
    syllables = [syl for syl in syllables if syl] or []
    return syllables, stress


def split_ipa_line(ipa: str) -> list[tuple[str, str]]:
    """Split an IPA line into (word, trailing pause mark) pairs."""
    out: list[tuple[str, str]] = []
    for raw in ipa.strip().split():
        mark = ""
        while raw and raw[-1] in ",!?.":
            mark = raw[-1]
            raw = raw[:-1]
        if raw or mark:
            out.append((raw, mark))
    return out
