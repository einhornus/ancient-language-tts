"""Proto-Indo-European IPA → Lithuanian respelling, by rule.

Implements `data/ine-pro/respelling/lt.txt`: Lithuanian letters, a grave
on a short stressed vowel and an acute on a long one, ch for /χ/, h for
/ɣʷ/, nothing for /ʔ/, k/g (or ki/gi before a back vowel) for /kʲ gʲ/.
"""

from __future__ import annotations

import re
import unicodedata

_TOKENS = (
    "gʲʱ", "gʷʱ", "kʲ", "gʲ", "kʷ", "gʷ", "bʱ", "dʱ", "gʱ", "ɣʷ",
    "aː", "eː", "iː", "oː", "uː", "i̯", "u̯",
    "r̩", "l̩", "m̩", "n̩",
    "a", "e", "i", "o", "u", "ə",
    "p", "b", "t", "d", "k", "g", "s", "z", "m", "n", "ŋ", "l", "r", "j", "w",
    "ʔ", "χ", "ˈ", ".",
)
_TOKEN_RE = re.compile("|".join(re.escape(t) for t in sorted(_TOKENS, key=len, reverse=True)))

_VOWELS = {
    "a": "a", "aː": "ą", "e": "e", "eː": "ė", "i": "i", "iː": "y",
    "o": "o", "oː": "o", "u": "u", "uː": "ū", "ə": "a",
}
_GRAVE = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù", "ə": "à"}
_ACUTE = {"aː": "ą́", "eː": "ė́", "iː": "ý", "oː": "ó", "uː": "ū́"}
_RESONANT = {"r̩": "r", "l̩": "l", "m̩": "m", "n̩": "n"}
_GLIDES = {"i̯": "i", "u̯": "u"}
_CONSONANTS = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g", "s": "s", "z": "z",
    "m": "m", "n": "n", "ŋ": "n", "l": "l", "r": "r", "j": "j", "w": "v",
    "kʷ": "kv", "gʷ": "gv", "gʷʱ": "ghv", "bʱ": "bh", "dʱ": "dh", "gʱ": "gh",
    "χ": "ch", "ɣʷ": "h", "ʔ": "",
}
_SOFT = {"kʲ": "k", "gʲ": "g", "gʲʱ": "gh"}
_BACK = frozenset({"a", "aː", "o", "oː", "u", "uː", "ə"})
_NUCLEI = frozenset(_VOWELS) | frozenset(_RESONANT)


def respell(ipa: str) -> str:
    """Respell a Proto-Indo-European IPA line in Lithuanian spelling."""
    out: list[str] = []
    for raw in unicodedata.normalize("NFC", ipa.strip()).split():
        word, pause = (raw[:-1], raw[-1]) if raw[-1] in ",!?." else (raw, "")
        if word:
            out.append(spell_word(word))
        if pause:
            out.append(pause)
        out.append(" ")
    text = "".join(out).strip()
    text = re.sub(r" ([,!?.])", r"\1", text)
    return capitalize(unicodedata.normalize("NFC", text))


def capitalize(text: str) -> str:
    """Capital at the start of the line and after . ! ?"""
    chars = list(text)
    start = True
    for k, ch in enumerate(chars):
        if start and ch.isalpha():
            chars[k] = ch.upper()
            start = False
        elif ch in ".!?":
            start = True
    return "".join(chars)


def spell_word(ipa_word: str) -> str:
    # (token, stressed syllable?) for every phone, with syllable marks gone.
    phones: list[tuple[str, bool]] = []
    stressed = False
    for tok in _TOKEN_RE.findall(unicodedata.normalize("NFD", ipa_word)):
        tok = unicodedata.normalize("NFC", tok)
        if tok == "ˈ":
            stressed = True
        elif tok == ".":
            stressed = False
        else:
            phones.append((tok, stressed))
    # Only the first nucleus of the stressed syllable takes the mark.
    marked = -1
    for k, (tok, st) in enumerate(phones):
        if st and tok in _NUCLEI:
            marked = k
            break

    out: list[str] = []
    for k, (tok, _st) in enumerate(phones):
        prev = phones[k - 1][0] if k else ""
        nxt = phones[k + 1][0] if k + 1 < len(phones) else ""
        if tok in _VOWELS:
            out.append(_hiatus(phones, k))
            if k == marked:
                out.append(_GRAVE[tok] if tok in _GRAVE else _ACUTE[tok])
            else:
                out.append(_VOWELS[tok])
        elif tok in _RESONANT:
            out.append(("ì" if k == marked else "i") + _RESONANT[tok])
        elif tok in _GLIDES:
            out.append(_GLIDES[tok])
        elif tok == prev:
            continue  # a geminate is written once
        elif tok in _SOFT:
            out.append(_SOFT[tok] + ("i" if nxt in _BACK else ""))
        else:
            out.append(_CONSONANTS[tok])
    return "".join(out)


def _hiatus(phones: list[tuple[str, bool]], k: int) -> str:
    """j in i + e, v in u + o, so Lithuanian does not read ie or uo."""
    tok = phones[k][0]
    j = k - 1
    while j >= 0 and phones[j][0] == "ʔ":
        j -= 1
    prev = phones[j][0] if j >= 0 else ""
    if prev == "i" and tok == "e":
        return "j"
    if prev == "u" and tok in ("o", "oː"):
        return "v"
    return ""
