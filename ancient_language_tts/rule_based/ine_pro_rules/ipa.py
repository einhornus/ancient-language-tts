"""Proto-Indo-European text → IPA, by rule.

Implements `data/ine-pro/phonology.txt`: one word at a time, syllables
separated by a dot, ˈ before the accented syllable, pauses kept.
"""

from __future__ import annotations

import unicodedata

from .analysis import Ph, Word, words

_STOPS = {
    "p": "p", "t": "t", "ḱ": "kʲ", "k": "k", "kʷ": "kʷ",
    "b": "b", "d": "d", "ǵ": "gʲ", "g": "g", "gʷ": "gʷ",
    "bʰ": "bʱ", "dʰ": "dʱ", "ǵʰ": "gʲʱ", "gʰ": "gʱ", "gʷʰ": "gʷʱ",
}
_LARYNGEAL = {"1": "ʔ", "2": "χ", "3": "ɣʷ"}
_SYLLABIC = {"r": "r̩", "l": "l̩", "m": "m̩", "n": "n̩", "y": "i", "w": "u"}
_CONSONANT = {"r": "r", "l": "l", "m": "m", "n": "n", "y": "j", "w": "w"}
_GLIDE = {"y": "i̯", "w": "u̯"}


def to_ipa(text: str) -> str:
    """Transcribe Proto-Indo-European `text` into IPA."""
    out: list[str] = []
    for kind, value in words(text):
        if kind == "pause":
            if out and out[-1] == " ":
                out.pop()
            out += [value, " "]
        else:
            out += [word_ipa(value), " "]  # type: ignore[arg-type]
    return unicodedata.normalize("NFC", "".join(out).strip())


def word_ipa(word: Word) -> str:
    parts = ["".join(phone(p) for p in syl) for syl in word.syllables]
    text = ""
    for i, body in enumerate(parts):
        if i == word.stress:
            text += "ˈ" + body
        elif i:
            text += "." + body
        else:
            text += body
    return text


def phone(p: Ph) -> str:
    if p.kind == "V":
        return p.quality + ("ː" if p.long else "")
    if p.kind == "H":
        return ("ə" if p.syllabic else "") + _LARYNGEAL[p.base]
    if p.kind == "R":
        if p.syllabic:
            return _SYLLABIC[p.base]
        if p.glide:
            return _GLIDE[p.base]
        if p.velar:
            return "ŋ"
        return _CONSONANT[p.base]
    if p.base == "s":
        return "z" if p.voiced else "s"
    return _STOPS[p.base]
