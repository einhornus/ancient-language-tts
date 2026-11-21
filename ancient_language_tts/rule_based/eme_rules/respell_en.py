"""Early Modern English IPA → modern English respelling, by rule.

Implements `data/eme/respelling/en.txt`: each IPA symbol maps to an English
spelling, stress and syllable dots are dropped, a word is one lowercase
string, capitalized only at the start of a sentence.
"""

from __future__ import annotations

import re

_TOKENS = (
    "əɪ", "əʊ", "ɔɪ", "ɪu", "iː", "eː", "ɔː", "oː", "uː", "tʃ", "dʒ",
    "iː", "ɛ", "ɪ", "ə", "a", "ɔ", "ʊ",
    "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ",
    "m", "n", "ŋ", "l", "r", "j", "w", "ʍ", "h",
)
_TOKEN_RE = re.compile("|".join(re.escape(t) for t in sorted(set(_TOKENS), key=len, reverse=True)))

_SPELL = {
    # vowels
    "iː": "ee", "eː": "ey", "ɛ": "eh", "ɪ": "ih", "a": "ae", "ɔ": "aw",
    "ɔː": "aw", "oː": "oh", "uː": "oo", "ʊ": "uh", "ə": "uh",
    "əɪ": "eye", "əʊ": "ow", "ɔɪ": "oy", "ɪu": "ew",
    # consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g", "f": "f",
    "v": "v", "s": "s", "z": "z", "m": "m", "n": "n", "l": "l", "w": "w",
    "h": "h", "j": "y", "θ": "th", "ð": "th", "ʃ": "sh", "ʒ": "zh",
    "tʃ": "ch", "dʒ": "j", "ŋ": "ng", "r": "r", "ʍ": "hw",
}

_PAUSES = ",.!?"


def respell(ipa: str) -> str:
    """Respell an Early Modern English IPA line in modern English letters."""
    pieces: list[str] = []
    for raw in ipa.strip().split():
        word = raw.rstrip(_PAUSES)
        marks = raw[len(word):]
        if word:
            pieces.append(spell_syllables(_syllables(word)))
        pieces += list(marks)
    return join(pieces)


def _syllables(word: str) -> list[list[str]]:
    return [_TOKEN_RE.findall(part) for part in re.split("[ˈ.]", word) if part]


def spell_syllables(syllables: list[list[str]]) -> str:
    """Spell a word given as syllables of IPA phones."""
    out: list[str] = []
    for syl in syllables:
        for i, phone in enumerate(syl):
            nxt = syl[i + 1] if i + 1 < len(syl) else ""
            if phone == "a" and nxt == "r":
                out.append("ah")  # ae before r would read as the vowel of air
            elif phone == "ɪ" and nxt == "ŋ":
                out.append("i")  # /ˈsɪŋ/ is sing, /ˈfɪŋ.gər/ is fingguhr
            else:
                out.append(_SPELL.get(phone, ""))
    return "".join(out)


def join(pieces: list[str]) -> str:
    """Join spelled words and pause marks; capitalize each sentence."""
    out: list[str] = []
    sentence = True
    for piece in pieces:
        if piece in _PAUSES:
            if out and out[-1] == " ":
                out.pop()
            out += [piece, " "]
            if piece in ".!?":
                sentence = True
            continue
        if not piece:
            continue
        if sentence:
            piece = piece[0].upper() + piece[1:]
            sentence = False
        out += [piece, " "]
    return "".join(out).strip()
