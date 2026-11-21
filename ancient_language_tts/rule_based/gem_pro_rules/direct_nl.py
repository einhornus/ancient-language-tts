"""Proto-Germanic spelling → Dutch respelling, with no IPA step.

Reads the written word (`analysis`), then writes each letter the way
`respelling/nl.txt` writes the sound behind it: u is oe, ī is ie, ē and ō
are ê and ô, þ is t, non-initial h is ch, a short stressed a/e/i in an open
syllable doubles the next consonant, and the stressed vowel takes an acute
when an unstressed prefix comes first.
"""

from __future__ import annotations

from . import analysis
from .analysis import Unit

_PLAIN = {"p": "p", "t": "t", "k": "k", "m": "m", "n": "n", "l": "l", "r": "r",
          "s": "s", "f": "f", "þ": "t", "z": "z", "w": "w", "j": "j", "kw": "kw"}
_FRIC = {"b": ("b", "v"), "d": ("d", "d"), "g": ("g", "g"), "gw": ("gw", "gw"),
         "h": ("h", "ch"), "hw": ("hw", "chw")}
_DIPH = {"eu": "eeuw", "iu": "ieuw", "ou": "ôw", "oi": "ooi", "ei": "ei"}

_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
_DIAERESIS = {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü"}
_DIGRAPHS = frozenset({"aa", "ee", "oo", "uu", "ie", "ei", "ai", "eu", "ui", "oe", "au", "ou", "oi"})


def to_dutch(text: str) -> str:
    """Respell Proto-Germanic `text` as Dutch, with no IPA step."""
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
        spelled = _word(analysis.analyze(token))
        if not spelled:
            continue
        if sentence:
            spelled = spelled[0].upper() + spelled[1:]
            sentence = False
        out.append(spelled)
    return " ".join(out)


def _cons(u: Unit) -> str:
    if u.base in _FRIC:
        return _FRIC[u.base][1 if u.fric else 0]
    return _PLAIN.get(u.base, "")


def _no_double(u: Unit) -> bool:
    """h (and ch), j, w, and hw are never doubled after a vowel."""
    return u.base in ("h", "hw", "j", "w")


def _ch(u: Unit) -> bool:
    return u.base in ("h", "hw") and u.fric


def _word(word: analysis.Word) -> str:
    units: list[tuple[Unit, int]] = []
    for k, syl in enumerate(word.syllables):
        units.extend((u, k) for u in syl)
    stress = word.stress
    last = len(word.syllables) - 1
    accent = bool(stress and stress > 0)
    out: list[str] = []
    double_next = False
    n = len(units)
    i = 0
    while i < n:
        u, k = units[i]
        nxt = units[i + 1][0] if i + 1 < n else None
        prev = units[i - 1][0] if i else None
        if u.nucleus:
            is_open = (i + 1 >= n or units[i + 1][1] != k) and k < last
            stressed = k == stress
            text = _vowel(u, is_open, nxt)
            if u.kind == "v" and u.length == 0 and u.base in "aei":
                if u.base == "i" and nxt is not None and nxt.base == "j":
                    text = "ie"
                elif is_open and nxt is not None and not nxt.nucleus and not _no_double(nxt) and (
                    stressed or u.base == "e"
                ):
                    double_next = True
            if prev is not None and prev.nucleus and out and text:
                if (out[-1][-1:] + text[0]) in _DIGRAPHS:
                    text = _DIAERESIS[text[0]] + text[1:]
            if accent and stressed:
                text = "".join(_ACUTE.get(c, c) for c in text)
            out.append(text)
            i += 1
            continue
        if nxt is not None and nxt == u and units[i + 1][1] != k:
            text = _cons(u)
            out.append("ch" if u.base == "h" else text[0] + text)
            double_next = False
            i += 2
            continue
        text = _cons(u)
        if u.base in ("g", "gw") and prev is not None and prev.base == "n" and prev.fric \
                and nxt is not None and nxt.nucleus:
            text = "g" + text
        elif double_next and text:
            text = text[0] + text
        double_next = False
        out.append(text)
        i += 1
    return "".join(out)


def _vowel(u: Unit, is_open: bool, nxt: Unit | None) -> str:
    if u.kind == "d":
        if u.base in ("ai", "au"):
            before_vowel = nxt is not None and nxt.nucleus
            return {"ai": ("ai", "aj"), "au": ("au", "auw")}[u.base][before_vowel]
        if u.base == "eu" and u.length:
            return "êw"
        return _DIPH.get(u.base, u.base)
    b = u.base
    if b == "o":
        return "ô"
    if b == "u":
        return "oe"
    if b == "i":
        return "ie" if u.length else "i"
    if b == "e" and u.length and not u.close:
        return "ê"
    if u.length:  # ā ą̄ â, ē₂
        single = b
        if is_open and not (nxt is not None and _ch(nxt)):
            return single
        return single + single
    return b
