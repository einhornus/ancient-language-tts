"""Proto-Slavic spelling → Ukrainian respelling, in one pass.

No IPA. The letters, syllables and stress come from `spelling`, and each
letter is spelled with the rules of `data/sla-pro/respelling/uk.txt`, so the
result equals `respell_uk.respell(ipa.to_ipa(text))`.
"""

from __future__ import annotations

from .respell_uk import ACUTE, capitalize
from .spelling import PALATALS, analyze, tokens

_CYR_VOWELS = frozenset("аеиіоуяєюї")
_SOFT_INITIAL = frozenset("яєюї")

_CONSONANTS = {
    "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "ґ",
    "s": "с", "z": "з", "x": "х", "f": "ф", "š": "ш", "ž": "ж",
    "č": "ч", "c": "ц", "dz": "дз", "dž": "дж",
    "m": "м", "n": "н", "l": "л", "r": "р", "v": "в",
}
_PALATAL = {"ľ": "л", "ň": "н", "ř": "р", "ť": "т", "ď": "д", "ś": "с"}
_LABIALS = frozenset({"p", "b", "m"})

# Letter spellings: plain, after j, after a palatal. "N" is the nasal letter.
_PLAIN = {
    "i": "і", "y": "и", "ь": "и", "e": "е", "ě": "я", "a": "а",
    "o": "о", "u": "у", "ъ": "у", "ę": "еN", "ǫ": "оN",
    "r̥": "ир", "l̥": "ил",
}
_AFTER_J = {
    "a": "я", "ě": "я", "e": "є", "i": "ї", "u": "ю", "ъ": "ю",
    "o": "йо", "ь": "йи", "y": "йи", "ę": "єN", "ǫ": "йоN",
}
_AFTER_PALATAL = {
    "a": "я", "ě": "я", "e": "є", "u": "ю", "ъ": "ю", "o": "ьо",
    "i": "і", "ь": "і", "y": "і", "ę": "єN", "ǫ": "ьоN",
}


def to_ukrainian(text: str) -> str:
    """Respell Proto-Slavic `text` as Ukrainian, with no IPA step."""
    words: list[str] = []
    sentence = False
    for kind, value in tokens(text.strip()):
        if kind == "pause":
            if words:
                words[-1] += value
                sentence = sentence or value in ".!?"
            continue
        spelled = _word(value)
        if spelled:
            words.append(spelled)
    return capitalize(words, sentence)


def _word(raw: str) -> str:
    word = analyze(raw)
    if word is None:
        return ""
    units = [u.letter for u in word.units]
    stressed = word.stressed_unit()
    out: list[str] = []
    for i, letter in enumerate(units):
        prev = units[i - 1] if i else None
        nxt = units[i + 1] if i + 1 < len(units) else None
        if letter in _PLAIN:
            text = _vowel(letter, prev, nxt)
            if i == stressed:
                text = _stress(text)
        else:
            text = _consonant(letter, prev, nxt)
        out.append(text)
    return "".join(out)


def _consonant(letter: str, prev: str | None, nxt: str | None) -> str:
    vowel_next = nxt in _AFTER_J
    if letter == "j":
        if not vowel_next:
            return "й"
        assert nxt is not None
        if prev is not None and prev not in _PLAIN and _AFTER_J[nxt][0] in _SOFT_INITIAL:
            return "'"
        return ""
    if letter in PALATALS:
        return _PALATAL[letter] + ("" if vowel_next else "ь")
    return _CONSONANTS[letter]


def _vowel(letter: str, prev: str | None, nxt: str | None) -> str:
    if prev == "j" and letter in _AFTER_J:
        text = _AFTER_J[letter]
    elif prev in PALATALS and letter in _AFTER_PALATAL:
        text = _AFTER_PALATAL[letter]
    elif letter == "ě" and (prev is None or prev in _PLAIN):
        text = "е"  # at the start of a word or after a vowel
    else:
        text = _PLAIN[letter]
    nasal = "м" if nxt in _LABIALS else "н"
    return text.replace("N", nasal)


def _stress(text: str) -> str:
    for k, ch in enumerate(text):
        if ch in _CYR_VOWELS:
            return text[:k + 1] + ACUTE + text[k + 1:]
    return text
