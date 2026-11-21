"""Proto-Germanic spelling → Icelandic respelling, with no IPA step.

Reads the written word (`analysis`), then writes each letter the way
`respelling/is.txt` writes the sound behind it: the stop or fricative value
of b d g h decides b/v, d/ð, h/g; vowel length is dropped; ai and au are
æ and á; z is s; an unstressed prefix is a separate word.
"""

from __future__ import annotations

from . import analysis
from .analysis import Unit

_DIPH = {"ai": "æ", "au": "á", "eu": "eú", "iu": "jú", "ou": "ó", "oi": "oj", "ei": "ei"}
_PLAIN = {"p": "p", "t": "t", "k": "k", "m": "m", "n": "n", "s": "s", "l": "l",
          "r": "r", "j": "j", "f": "f", "þ": "þ", "z": "s", "w": "v", "kw": "kv"}
_FRIC = {"b": ("b", "v"), "d": ("d", "ð"), "g": ("g", "g"), "gw": ("gv", "gv"),
         "h": ("h", "g"), "hw": ("hv", "gv")}
_GEMINATE = {"m": "mm", "s": "ss", "r": "rr", "d": "dd", "g": "gg", "b": "bb",
             "f": "ff", "z": "ss", "l": "l", "p": "p", "t": "t", "k": "k",
             "j": "j", "w": "v", "þ": "þ", "h": "g"}


def to_icelandic(text: str) -> str:
    """Respell Proto-Germanic `text` as Icelandic, with no IPA step."""
    out: list[str] = []
    sentence = True
    for token in analysis.tokens(text):
        mark = analysis.pause(token)
        if mark:
            if out:
                out[-1] += mark
            else:
                out.append(mark)
            if mark in ".!?":
                sentence = True
            continue
        word = analysis.analyze(token)
        spelled = _word(word)
        if not spelled:
            continue
        if sentence:
            spelled = spelled[0].upper() + spelled[1:]
            sentence = False
        out.append(spelled)
    return " ".join(out)


def _word(word: analysis.Word) -> str:
    if word.stress:
        return _flat(word.syllables[:word.stress]) + " " + _flat(word.syllables[word.stress:])
    return _flat(word.syllables)


def _vowel(u: Unit) -> str:
    if u.kind == "d":
        return _DIPH.get(u.base, u.base)
    if u.base == "i":
        return "í" if u.length and not u.nasal else "i"
    if u.base == "u":
        return "ú"
    return u.base  # a e o


def _cons(u: Unit) -> str:
    if u.base in _FRIC:
        return _FRIC[u.base][1 if u.fric else 0]
    return _PLAIN.get(u.base, "")


def _flat(syllables: list[list[Unit]]) -> str:
    units: list[tuple[Unit, int]] = []
    for k, syl in enumerate(syllables):
        units.extend((u, k) for u in syl)
    out: list[str] = []
    i = 0
    while i < len(units):
        u, k = units[i]
        if u.nucleus:
            out.append(_vowel(u))
            i += 1
            continue
        nxt = units[i + 1] if i + 1 < len(units) else None
        if nxt and nxt[0] == u and nxt[1] != k:
            if u.base == "n":
                prev = units[i - 1][0] if i else None
                plain = prev is not None and prev.kind == "v" and _vowel(prev) in ("a", "e", "i", "o")
                out.append("nn" if plain else "n")
            else:
                out.append(_GEMINATE.get(u.base, _cons(u)))
            i += 2
            continue
        out.append(_cons(u))
        i += 1
    return "".join(out)
