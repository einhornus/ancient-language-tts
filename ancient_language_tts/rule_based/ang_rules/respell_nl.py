"""Old English IPA → Dutch respelling. Implements `data/ang/respelling/nl.txt`."""

from __future__ import annotations

from ._common import is_vowel, parse_ipa, render

_C = {
    "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "m": "m", "n": "n",
    "l": "l", "r": "r", "s": "s", "g": "g", "ɣ": "g", "x": "ch", "ç": "ch",
    "h": "h", "j": "j", "w": "w", "ʍ": "hw", "l̥": "hl", "r̥": "hr",
    "n̥": "hn", "f": "f", "v": "v", "z": "z", "θ": "t", "ð": "d",
    "tʃ": "tsj", "dʒ": "dzj", "ʃ": "sj",
}
_SHORT = {"ɑ": "a", "æ": "e", "e": "e", "i": "i", "o": "o", "u": "oe", "y": "u"}
_FIXED = {
    "æː": "ê", "iː": "ie", "uː": "oe",
    "æɑ̯": "ea", "æːɑ̯": "ea", "eo̯": "eo", "eːo̯": "eo",
}
_LONG = {"ɑː": ("a", "aa"), "eː": ("e", "ee"), "oː": ("o", "oo"), "yː": ("u", "uu")}
_CLOSING = frozenset({"tʃ", "dʒ", "ʃ", "x", "ç"})  # Dutch closes the syllable before these
_NO_DOUBLE = frozenset({"h", "j", "w", "ʍ", "l̥", "r̥", "n̥"}) | _CLOSING
_ONE_LETTER_GEM = frozenset({"ʃ", "x", "ç"})  # ʃ.ʃ → sj, x.x / ç.ç → ch
_DIGRAPHS = frozenset({"aa", "ai", "au", "ee", "ei", "eu", "ie", "ij", "oe", "oi",
                       "oo", "ou", "ui", "uu"})
_DIAERESIS = {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü"}
_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}


def respell(ipa: str) -> str:
    """Respell an Old English IPA line in Dutch orthography."""
    return spell_items(parse_ipa(ipa.strip()))


def spell_items(items: list[tuple]) -> str:
    return render(items, spell_word)


def _cons(p: str, following: str | None) -> str:
    if p == "ŋ":
        return "ng" if following == "g" else "n"
    return _C.get(p, p)


def spell_word(sylls: list[list[str]], stress: int | None) -> str:
    accent = stress is not None and stress > 0
    n = len(sylls)
    parts: list[str] = []
    for i, syl in enumerate(sylls):
        nxt = sylls[i + 1] if i + 1 < n else None
        nxt_first = nxt[0] if nxt else None
        nuc = next((k for k, p in enumerate(syl) if is_vowel(p)), None)
        if nuc is None:
            parts.append("".join(_cons(p, None) for p in syl))
            continue
        onset, v, coda = syl[:nuc], syl[nuc], list(syl[nuc + 1:])
        open_ = not coda
        last = nxt is None
        closing = nxt_first in _CLOSING
        stressed = i == stress
        simple = False
        if v in ("æ", "æː", "e", "eː") and coda[:1] == ["j"]:
            vs, coda = "ei", coda[1:]
        elif v == "i" and coda[:1] == ["j"]:
            vs, coda = "ie", coda[1:]
        elif v in ("æ", "e") and open_ and nxt_first == "j" and len(nxt) > 1 and is_vowel(nxt[1]):
            vs = "ei"
        elif v in _LONG:
            single, double = _LONG[v]
            ends_ok = v == "oː" or not last
            vs = single if open_ and ends_ok and not closing else double
        elif v in _FIXED:
            vs = _FIXED[v]
        else:
            vs = _SHORT.get(v, v)
            simple = v in ("ɑ", "æ", "e", "i", "o")
        doubled = ""
        if (simple and open_ and nxt_first and not is_vowel(nxt_first)
                and nxt_first not in _NO_DOUBLE and (stressed or v in ("æ", "e"))):
            doubled = _cons(nxt_first, nxt[1] if len(nxt) > 1 else None)[0]
        if accent and stressed:
            vs = "".join(_ACUTE.get(ch, ch) for ch in vs)
        elif not onset and parts and parts[-1] and parts[-1][-1] + vs[0] in _DIGRAPHS:
            vs = _DIAERESIS.get(vs[0], vs[0]) + vs[1:]
        spelled_onset = "".join(_cons(p, syl[k + 1] if k + 1 < len(syl) else None)
                                for k, p in enumerate(onset))
        spelled_coda = ""
        for k, p in enumerate(coda):
            following = coda[k + 1] if k + 1 < len(coda) else nxt_first
            if k == len(coda) - 1 and p in _ONE_LETTER_GEM and nxt_first == p:
                continue
            if p == "ŋ":
                spelled_coda += "ng" if k == len(coda) - 1 and nxt_first == "g" else "n"
            else:
                spelled_coda += _cons(p, following)
        parts.append(spelled_onset + vs + spelled_coda + doubled)
    return "".join(parts)
