"""Classical Latin text → IPA, by rule.

Implements `data/lat/phonology.txt`. The analysis (graphemes, syllables,
stress) lives in `spelling`; this module writes it out: syllables joined by
`.`, `ˈ` in place of the dot before the stressed syllable, words separated
by spaces, pause marks attached to the word before them.
"""

from __future__ import annotations

from .spelling import Syl, analyse


def to_ipa(text: str) -> str:
    """Transcribe Classical Latin `text` into IPA."""
    out = ""
    for item in analyse(text):
        if isinstance(item, str):
            out += item
        else:
            out += (" " if out else "") + format_word(item)
    return out


def format_word(word: list[Syl]) -> str:
    out = ""
    for i, syl in enumerate(word):
        body = "".join(syl.phones)
        if syl.stressed:
            out += "ˈ" + body
        else:
            out += ("." if i else "") + body
    return out
