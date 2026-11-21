"""Proto-Slavic IPA → Ukrainian respelling, by rule.

Implements `data/sla-pro/respelling/uk.txt`: lowercase Cyrillic with a
combining acute (U+0301) after the vowel letter of each stressed syllable,
ґ for [g], я for [æ], the nasal vowels as vowel + н/м, jers as и/у (і after
a palatal), the glide j folded into я є ю ї, and the palatals marked by the
following letter or ь.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

ACUTE = "́"
_CYR_VOWELS = frozenset("аеиіоуяєюї")
_PAUSES = ",!?."

# IPA symbols (NFD) → phoneme keys, longest first.
_NASAL = "̃"
_SYL = ("̩", "̥")
_VOWELS = {
    "i": "i", "ɨ": "ɨ", "u": "u", "æ": "æ", "ɑ": "ɑ", "a": "ɑ",
    "e": "e", "ɛ": "e", "o": "o", "ɔ": "o", "ɪ": "ɪ", "ʊ": "ʊ",
}
_AFFRICATES = {"tʃ": "tʃ", "ts": "ts", "dz": "dz", "dʒ": "dʒ"}
_CONSONANTS = {
    "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "ґ", "ɡ": "ґ",
    "s": "с", "z": "з", "x": "х", "f": "ф", "ʃ": "ш", "ʒ": "ж",
    "tʃ": "ч", "ts": "ц", "dz": "дз", "dʒ": "дж",
    "m": "м", "n": "н", "ŋ": "н", "l": "л", "r": "р", "w": "в", "v": "в",
}
_PALATAL = {"lʲ": "л", "nʲ": "н", "rʲ": "р", "tʲ": "т", "dʲ": "д", "sʲ": "с"}
_LABIALS = frozenset({"p", "b", "m"})

# Vowel spellings: plain, after j, after a palatal. "N" is the nasal letter.
_PLAIN = {
    "i": "і", "ɨ": "и", "ɪ": "и", "e": "е", "æ": "я", "ɑ": "а",
    "o": "о", "u": "у", "ʊ": "у", "ẽ": "еN", "õ": "оN",
    "r̩": "ир", "l̩": "ил",
}
_AFTER_J = {
    "ɑ": "я", "æ": "я", "e": "є", "i": "ї", "u": "ю", "ʊ": "ю",
    "o": "йо", "ɪ": "йи", "ɨ": "йи", "ẽ": "єN", "õ": "йоN",
}
_AFTER_PALATAL = {
    "ɑ": "я", "æ": "я", "e": "є", "u": "ю", "ʊ": "ю", "o": "ьо",
    "i": "і", "ɪ": "і", "ɨ": "і", "ẽ": "єN", "õ": "ьоN",
}
_SOFT_INITIAL = frozenset("яєюї")


@dataclass
class _Seg:
    ph: str
    nucleus: bool
    syllable: int


def respell(ipa: str) -> str:
    """Respell a Proto-Slavic IPA line as Ukrainian."""
    words: list[str] = []
    sentence = False
    for token in ipa.split():
        core = token.rstrip(_PAUSES)
        marks = token[len(core):]
        sentence = sentence or any(m in ".!?" for m in marks)
        spelled = _word(core) if core else ""
        if not spelled and not words:
            continue
        if spelled:
            words.append(spelled + marks)
        else:
            words[-1] += marks
    return capitalize(words, sentence)


def capitalize(words: list[str], sentence: bool) -> str:
    """Join words; capitalize each sentence start when the line is a sentence.

    A lone citation form (no . ! ?) stays lowercase, as in the examples.
    """
    out: list[str] = []
    start = sentence
    for w in words:
        if start and w:
            w = w[0].upper() + w[1:]
        start = sentence and w[-1:] in (".", "!", "?")
        out.append(w)
    return unicodedata.normalize("NFC", " ".join(out))


def _parse(word: str) -> tuple[list[_Seg], int | None]:
    s = unicodedata.normalize("NFD", word)
    segs: list[_Seg] = []
    syllable = 0
    stress: int | None = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == ".":
            syllable += 1
            i += 1
            continue
        if ch == "ˈ":
            if segs:
                syllable += 1
            stress = syllable
            i += 1
            continue
        if ch in "ːˑ" or unicodedata.combining(ch):
            i += 1
            continue
        two = s[i:i + 2]
        if two in _AFFRICATES:
            segs.append(_Seg(two, False, syllable))
            i += 2
            continue
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt == "ʲ" and ch + "ʲ" in _PALATAL:
            segs.append(_Seg(ch + "ʲ", False, syllable))
            i += 2
            continue
        if ch in _VOWELS:
            key = _VOWELS[ch]
            if nxt == _NASAL and key in ("e", "o"):
                key = {"e": "ẽ", "o": "õ"}[key]
                i += 1
            segs.append(_Seg(key, True, syllable))
            i += 1
            continue
        if ch in ("r", "l") and nxt in _SYL:
            segs.append(_Seg(ch + "̩", True, syllable))
            i += 2
            continue
        if ch in _CONSONANTS or ch == "j":
            segs.append(_Seg(ch, False, syllable))
        i += 1
    return segs, stress


def _word(word: str) -> str:
    segs, stress = _parse(word)
    out: list[str] = []
    for i, seg in enumerate(segs):
        prev = segs[i - 1] if i else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if not seg.nucleus:
            out.append(_consonant(seg, prev, nxt))
            continue
        text = _vowel(seg, prev, nxt)
        if stress is not None and seg.syllable == stress:
            text = _stress(text)
        out.append(text)
    return "".join(out)


def _consonant(seg: _Seg, prev: _Seg | None, nxt: _Seg | None) -> str:
    vowel_next = nxt is not None and nxt.nucleus and nxt.ph in _AFTER_J
    if seg.ph == "j":
        if not vowel_next:
            return "й"
        assert nxt is not None
        if prev is not None and not prev.nucleus and _AFTER_J[nxt.ph][0] in _SOFT_INITIAL:
            return "'"  # б'я, т'є
        return ""  # folded into the next vowel letter
    if seg.ph in _PALATAL:
        return _PALATAL[seg.ph] + ("" if vowel_next else "ь")
    return _CONSONANTS[seg.ph]


def _vowel(seg: _Seg, prev: _Seg | None, nxt: _Seg | None) -> str:
    if prev is not None and prev.ph == "j" and seg.ph in _AFTER_J:
        text = _AFTER_J[seg.ph]
    elif prev is not None and prev.ph in _PALATAL and seg.ph in _AFTER_PALATAL:
        text = _AFTER_PALATAL[seg.ph]
    elif seg.ph == "æ" and (prev is None or prev.nucleus):
        text = "е"  # at the start of a word or after a vowel
    else:
        text = _PLAIN[seg.ph]
    nasal = "м" if nxt is not None and nxt.ph in _LABIALS else "н"
    return text.replace("N", nasal)


def _stress(text: str) -> str:
    for k, ch in enumerate(text):
        if ch in _CYR_VOWELS:
            return text[:k + 1] + ACUTE + text[k + 1:]
    return text
