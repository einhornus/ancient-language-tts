"""Proto-Indo-European spelling → Lithuanian respelling, with no IPA step.

Reads the phonemes from the spelling (`analysis`) and writes Lithuanian
letters for them, following `respelling/lt.txt`. The result is the same as
`respell_lt.respell(ipa.to_ipa(text))`.
"""

from __future__ import annotations

import unicodedata

from .analysis import Ph, Word, words
from .respell_lt import capitalize

_STOPS = {
    "p": "p", "t": "t", "k": "k", "kʷ": "kv", "b": "b", "d": "d", "g": "g", "gʷ": "gv",
    "bʰ": "bh", "dʰ": "dh", "gʰ": "gh", "gʷʰ": "ghv",
}
_SOFT = {"ḱ": "k", "ǵ": "g", "ǵʰ": "gh"}
_LARYNGEAL = {"1": "", "2": "ch", "3": "h"}
_SHORT = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u"}
_LONG = {"a": "ą", "e": "ė", "i": "y", "o": "o", "u": "ū"}
_GRAVE = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
_ACUTE = {"a": "ą́", "e": "ė́", "i": "ý", "o": "ó", "u": "ū́"}
_RESONANT = {"r": "r", "l": "l", "m": "m", "n": "n", "y": "j", "w": "v"}


def to_lithuanian(text: str) -> str:
    """Respell Proto-Indo-European `text` in Lithuanian, with no IPA."""
    out: list[str] = []
    for kind, value in words(text):
        if kind == "pause":
            if out and out[-1] == " ":
                out.pop()
            out += [value, " "]
        else:
            out += [_word(value), " "]  # type: ignore[arg-type]
    return capitalize(unicodedata.normalize("NFC", "".join(out).strip()))


def _vowel(p: Ph) -> tuple[str, bool]:
    """(quality, long) of a vowel or of a syllabic y / w."""
    if p.kind == "V":
        return p.quality, p.long
    return ("i" if p.base == "y" else "u"), False


def _is_vowel(p: Ph) -> bool:
    return p.kind == "V" or (p.kind == "R" and p.syllabic and p.base in "yw")


def _back(p: Ph | None) -> bool:
    if p is None:
        return False
    if p.kind == "H" and p.syllabic:
        return True  # [ə]
    return _is_vowel(p) and _vowel(p)[0] in "aou"


def _word(word: Word) -> str:
    phs = [p for _i, p in word.phonemes()]
    stressed_syl = word.syllables[word.stress] if word.stress is not None else []
    marked = next((p for p in stressed_syl if p.nucleus and p.kind != "H"), None)

    out: list[str] = []
    for k, p in enumerate(phs):
        prev = phs[k - 1] if k else None
        nxt = phs[k + 1] if k + 1 < len(phs) else None
        stressed = p is marked
        if _is_vowel(p):
            quality, long = _vowel(p)
            out.append(_hiatus(phs, k))
            if stressed:
                out.append(_ACUTE[quality] if long else _GRAVE[quality])
            else:
                out.append(_LONG[quality] if long else _SHORT[quality])
        elif p.kind == "R" and p.syllabic:
            out.append(("ì" if stressed else "i") + p.base)
        elif p.kind == "H" and p.syllabic:
            out.append("a" + _LARYNGEAL[p.base])
        elif prev is not None and not prev.nucleus and prev.key() == p.key():
            continue  # a geminate is written once
        elif p.kind == "H" and prev is not None and prev.kind == "H" and prev.base == p.base:
            continue  # syllabic laryngeal + the same laryngeal: written once
        elif p.kind == "R":
            if p.glide:
                out.append("i" if p.base == "y" else "u")
            else:
                out.append(_RESONANT[p.base])
        elif p.kind == "H":
            out.append(_LARYNGEAL[p.base])
        elif p.base == "s":
            out.append("z" if p.voiced else "s")
        elif p.base in _SOFT:
            out.append(_SOFT[p.base] + ("i" if _back(nxt) else ""))
        else:
            out.append(_STOPS[p.base])
    return "".join(out)


def _hiatus(phs: list[Ph], k: int) -> str:
    """j in i + e, v in u + o, so Lithuanian does not read ie or uo."""
    quality, long = _vowel(phs[k])
    j = k - 1
    while j >= 0 and phs[j].kind == "H" and not phs[j].syllabic and phs[j].base == "1":
        j -= 1
    if j < 0 or not _is_vowel(phs[j]):
        return ""
    pq, plong = _vowel(phs[j])
    if plong:
        return ""
    if pq == "i" and quality == "e" and not long:
        return "j"
    if pq == "u" and quality == "o":
        return "v"
    return ""
