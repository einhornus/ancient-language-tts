"""Koine Greek IPA → Modern Greek respelling, by rule.

Implements `data/grc/respelling/el.txt`: monotonic lowercase Greek with a
tonos on every stressed syllable, aspirates as π τ κ, /d/ as ντ, /y/ as ι,
no /h/, no doubled consonant letters, and a capital at each sentence start.
"""

from __future__ import annotations

from dataclasses import dataclass

_MULTI = ("pʰ", "tʰ", "kʰ")
_VOWEL_LETTERS = {"a": "α", "ɛ": "ε", "e": "ε", "i": "ι", "y": "ι", "o": "ο", "u": "ου"}
_CONSONANTS = {
    "p": "π", "pʰ": "π", "t": "τ", "tʰ": "τ", "k": "κ", "kʰ": "κ",
    "d": "ντ", "β": "β", "ɣ": "γ", "ʝ": "γ", "ɸ": "φ", "ŋ": "γ", "g": "γ",
    "s": "σ", "m": "μ", "n": "ν", "l": "λ", "r": "ρ", "h": "",
    "f": "φ", "v": "β", "x": "κ", "θ": "τ", "ð": "ντ", "j": "ι", "w": "ου",
}
_VOICED = frozenset({"β", "ɣ", "ʝ", "d", "m", "n", "l", "r", "b", "g", "ŋ", "v", "ð"})
_PAUSES = {",": ",", ".": ".", "!": "!", "?": ";", ";": ";"}

TONOS = {"α": "ά", "ε": "έ", "ι": "ί", "ο": "ό", "ϊ": "ΐ", "ου": "ού", "υ": "ύ", "η": "ή", "ω": "ώ"}
DIAERESIS_AFTER = frozenset("αεο")


@dataclass
class _Seg:
    sym: str
    syl: int
    stressed: bool


def respell(ipa: str) -> str:
    """Respell a Koine Greek IPA line in Modern Greek letters."""
    out: list[str] = []
    sentence = True
    for raw in ipa.split():
        body = raw.rstrip(",.!?;")
        pauses = raw[len(body):]
        word = _word(body)
        if word:
            if sentence:
                word = word[0].upper() + word[1:]
                sentence = False
            out.append(word)
        for mark in pauses:
            if mark in _PAUSES and out:
                out[-1] += _PAUSES[mark]
                if mark in ".!?;":
                    sentence = True
    return " ".join(out)


def _parse(body: str) -> list[_Seg]:
    segs: list[_Seg] = []
    syl, stressed = 0, False
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "ˈˌ.":
            if segs:
                syl += 1
            stressed = ch == "ˈ"
            i += 1
            continue
        pair = body[i:i + 2]
        if pair in _MULTI:
            segs.append(_Seg(pair, syl, stressed))
            i += 2
            continue
        if ch in "ːˑʰ" or ch not in _VOWEL_LETTERS and ch not in _CONSONANTS and ch not in "bz":
            i += 1
            continue
        segs.append(_Seg(ch, syl, stressed))
        i += 1
    return segs


def _word(body: str) -> str:
    segs = _parse(body)
    letters: list[str] = []
    prev_vowel = ""  # spelled letters of the vowel just before, if adjacent
    i = 0
    while i < len(segs):
        seg = segs[i]
        sym = seg.sym
        nxt = segs[i + 1].sym if i + 1 < len(segs) else ""
        if sym in _VOWEL_LETTERS:
            spelled = _VOWEL_LETTERS[sym]
            if spelled == "ι" and prev_vowel[-1:] in DIAERESIS_AFTER and prev_vowel:
                spelled = "ϊ"
            if seg.stressed:
                spelled = TONOS[spelled]
            letters.append(spelled)
            prev_vowel = _VOWEL_LETTERS[sym]
            i += 1
            continue
        prev_vowel = ""
        if sym in ("k", "p") and nxt == "s":
            letters.append("ξ" if sym == "k" else "ψ")
            i += 2
            continue
        if sym == "z":
            letters.append("σ" if nxt in _VOICED else "ζ")
        elif sym == "b":
            previous = segs[i - 1].sym if i else ""
            letters.append("π" if previous == "m" else "μπ")
        else:
            letters.append(_CONSONANTS[sym])
        i += 1
    return finish("".join(letters))


def finish(word: str) -> str:
    """No doubled consonant letters except γγ; final ς."""
    out: list[str] = []
    for ch in word:
        if out and ch == out[-1] and ch in "βδζθκλμνξπρστφχψ":
            continue
        out.append(ch)
    if out and out[-1] == "σ":
        out[-1] = "ς"
    return "".join(out)
