"""Proto-Germanic IPA → Dutch respelling, by rule.

Implements `data/gem-pro/respelling/nl.txt`: oe for u, ie for long i, ê
and ô for the open long e and o, doubled consonants after a short vowel in
an open syllable, ch for x, t and d for θ and ð, and an acute accent on the
stressed vowel when an unstressed prefix comes first.
"""

from __future__ import annotations

import unicodedata

from .ipa import ipa_syllables, split_ipa_line


def _nfc(table: dict[str, str]) -> dict[str, str]:
    return {unicodedata.normalize("NFC", k): v for k, v in table.items()}


# Vowels whose spelling never changes with position.
_FIXED = _nfc({
    "ɛː": "ê", "ɛːː": "ê",
    "ɔː": "ô", "ɔːː": "ô", "ɔ̃ː": "ô", "ɔ̃ːː": "ô", "ɔ": "ô", "ɔ̃": "ô",
    "iː": "ie", "iːː": "ie", "ĩː": "ie",
    "uː": "oe", "uːː": "oe", "ũː": "oe",
    "u": "oe", "ũ": "oe",
    "eu̯": "eeuw", "iu̯": "ieuw", "ɔːu̯": "ôw", "ɔːi̯": "ooi", "ɛːi̯": "ei", "ɛːu̯": "êw",
})
# Long a and close long e: single in an open syllable, double elsewhere.
_LONG = _nfc({"ɑː": "a", "ɑːː": "a", "ɑ̃ː": "a", "ɑ̃ːː": "a", "eː": "e", "eːː": "e"})
# Short vowels that may need a doubled consonant after them.
_SHORT = _nfc({"ɑ": "a", "ɑ̃": "a", "e": "e", "ẽ": "e", "i": "i", "ĩ": "i"})
_DIPH = _nfc({"ɑi̯": ("ai", "aj"), "ɑu̯": ("au", "auw"), "ɑːi̯": ("ai", "aj"), "ɑːu̯": ("au", "auw")})

_CONS = {
    "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "g",
    "m": "m", "n": "n", "ŋ": "n", "l": "l", "r": "r", "s": "s",
    "ɣ": "g", "β": "v", "ɸ": "f", "f": "f", "v": "v", "θ": "t", "ð": "d",
    "z": "z", "x": "ch", "h": "h", "hʷ": "hw", "xʷ": "chw", "kʷ": "kw",
    "gʷ": "gw", "ɣʷ": "gw", "w": "w", "j": "j",
}
# Never doubled after a vowel.
_NO_DOUBLE = frozenset({"h", "j", "w", "hʷ", "x", "xʷ"})
_CH = frozenset({"x", "xʷ"})

_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
_DIAERESIS = {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü"}
_DIGRAPHS = frozenset({"aa", "ee", "oo", "uu", "ie", "ei", "ai", "eu", "ui", "oe", "au", "ou", "oi"})


def _is_vowel(phone: str) -> bool:
    return phone[:1] in "ɑeɛiɔuẽĩũ" or unicodedata.normalize("NFD", phone)[:1] in "ɑeɛiɔu"


def respell(ipa: str) -> str:
    """Respell a Proto-Germanic IPA line in Dutch orthography."""
    out: list[str] = []
    sentence = True
    for word, mark in split_ipa_line(ipa):
        if word:
            syllables, stress = ipa_syllables(word)
            spelled = spell(syllables, stress)
            if spelled:
                if sentence:
                    spelled = spelled[0].upper() + spelled[1:]
                    sentence = False
                out.append(spelled)
        if mark:
            if out:
                out[-1] += mark
            else:
                out.append(mark)
            if mark in ".!?":
                sentence = True
    return " ".join(out)


def spell(syllables: list[list[str]], stress: int | None) -> str:
    """Spell one word given as syllables of IPA phones."""
    phones: list[tuple[str, int]] = []
    for k, syl in enumerate(syllables):
        phones.extend((p, k) for p in syl)
    last_syllable = len(syllables) - 1
    accent = bool(stress and stress > 0)
    out: list[str] = []
    double_next = False
    i = 0
    n = len(phones)
    while i < n:
        p, k = phones[i]
        nxt = phones[i + 1][0] if i + 1 < n else ""
        prev = phones[i - 1][0] if i else ""
        if _is_vowel(p):
            is_open = (i + 1 >= n or phones[i + 1][1] != k) and k < last_syllable
            stressed = k == stress
            if p in _FIXED:
                text = _FIXED[p]
            elif p in _LONG:
                base = _LONG[p]
                text = base if is_open and nxt not in _CH else base + base
            elif p in _DIPH:
                plain, before_vowel = _DIPH[p]
                text = before_vowel if nxt and _is_vowel(nxt) else plain
            elif p in _SHORT:
                text = _SHORT[p]
                if text == "i" and nxt == "j":
                    text = "ie"
                elif is_open and nxt and not _is_vowel(nxt) and nxt not in _NO_DOUBLE and (
                    (stressed and text in "aei") or (not stressed and text == "e")
                ):
                    double_next = True
            else:
                text = ""
            if prev and _is_vowel(prev) and out and text:
                if (out[-1][-1:] + text[0]) in _DIGRAPHS:
                    text = _DIAERESIS[text[0]] + text[1:]
            if accent and stressed:
                text = "".join(_ACUTE.get(c, c) if c in "aeiou" else c for c in text)
            out.append(text)
            i += 1
            continue
        # consonants
        if nxt == p and phones[i + 1][1] != k:
            text = _CONS.get(p, "")
            out.append("ch" if p == "x" else (text[0] + text if text else ""))
            i += 2
            double_next = False
            continue
        text = _CONS.get(p, "")
        if p in ("g", "gʷ") and prev == "ŋ" and nxt and _is_vowel(nxt):
            text = "g" + text
        elif double_next and text:
            text = text[0] + text
        double_next = False
        out.append(text)
        i += 1
    return "".join(out)
