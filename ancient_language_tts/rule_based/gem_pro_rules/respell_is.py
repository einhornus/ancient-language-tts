"""Proto-Germanic IPA → Icelandic respelling, by rule.

Implements `data/gem-pro/respelling/is.txt`: vowel quality without length,
æ and á for the diphthongs ai and au, þ ð v f for the fricatives, s for z,
g for ɣ and x, a separate word for an unstressed prefix, and the
Icelandic limits on doubled letters.
"""

from __future__ import annotations

import unicodedata

from .ipa import ipa_syllables, split_ipa_line

_VOWELS = {
    "ɑ": "a", "ɑː": "a", "ɑːː": "a",
    "e": "e", "eː": "e", "eːː": "e", "ɛː": "e", "ɛːː": "e",
    "i": "i", "iː": "í", "iːː": "í",
    "ɔː": "o", "ɔːː": "o", "ɔ": "o",
    "u": "ú", "uː": "ú", "uːː": "ú",
    "ɑ̃": "a", "ɑ̃ː": "a", "ɑ̃ːː": "a",
    "ẽ": "e", "ẽː": "e",
    "ĩ": "i", "ĩː": "i",
    "ũ": "ú", "ũː": "ú",
    "ɔ̃ː": "o", "ɔ̃ːː": "o", "ɔ̃": "o",
    "ɑi̯": "æ", "ɑu̯": "á", "eu̯": "eú", "iu̯": "jú",
    "ɔːu̯": "ó", "ɔːi̯": "oj", "ɛːi̯": "ei", "ɑːi̯": "æ", "ɑːu̯": "á", "ɛːu̯": "eú",
}
_VOWELS = {unicodedata.normalize("NFC", k): v for k, v in _VOWELS.items()}

_CONS = {
    "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "g",
    "m": "m", "n": "n", "ŋ": "n", "s": "s", "l": "l", "r": "r",
    "ɣ": "g", "β": "v", "ɸ": "f", "f": "f", "v": "v", "θ": "þ", "ð": "ð",
    "z": "s", "x": "g", "h": "h", "hʷ": "hv", "xʷ": "gv", "kʷ": "kv",
    "gʷ": "gv", "ɣʷ": "gv", "w": "v", "j": "j",
}

# Geminates (same consonant either side of a syllable break).
_GEMINATE = {
    "m": "mm", "s": "ss", "r": "rr", "d": "dd", "g": "gg", "b": "bb",
    "ɸ": "ff", "z": "ss",
    "l": "l", "p": "p", "t": "t", "k": "k", "j": "j", "w": "v",
    "θ": "þ", "ð": "ð", "β": "v", "x": "g", "ɣ": "g",
}
# nn is written nn only after these vowel letters.
_NN_AFTER = ("a", "e", "i", "o")


def respell(ipa: str) -> str:
    """Respell a Proto-Germanic IPA line in Icelandic orthography."""
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
    if stress and stress > 0:
        head = _spell_flat(syllables[:stress])
        tail = _spell_flat(syllables[stress:])
        return head + " " + tail
    return _spell_flat(syllables)


def _spell_flat(syllables: list[list[str]]) -> str:
    phones: list[tuple[str, int]] = []  # (phone, syllable index)
    for k, syl in enumerate(syllables):
        phones.extend((p, k) for p in syl)
    out: list[str] = []
    i = 0
    while i < len(phones):
        p, k = phones[i]
        nxt = phones[i + 1] if i + 1 < len(phones) else None
        if p in _VOWELS:
            out.append(_VOWELS[p])
            i += 1
            continue
        if nxt and nxt[0] == p and nxt[1] != k:
            if p == "n":
                prev = phones[i - 1][0] if i else ""
                plain = prev in _VOWELS and "̯" not in prev
                out.append("nn" if plain and _VOWELS[prev] in _NN_AFTER else "n")
            else:
                out.append(_GEMINATE.get(p, _CONS.get(p, "")))
            i += 2
            continue
        out.append(_CONS.get(p, ""))
        i += 1
    return "".join(out)
