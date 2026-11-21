"""Classical Latin IPA → Modern Italian respelling, by rule.

Implements `data/lat/respelling/it.txt`. `respell_it` parses an IPA line into
syllables and spells it; `spell_items` is the speller itself, shared with
`direct_it`, which feeds it the analysis of the Latin spelling instead.
"""

from __future__ import annotations

from .spelling import Syl, is_vowel

_DIPHTHONGS = ("ae̯", "oe̯", "au̯", "ei̯", "eu̯", "ui̯")
_MODIFIERS = frozenset("ːʰʷ̃")
_PAUSE_CHARS = ",!?."

# Plain and stressed spelling of each vowel.
_PLAIN = {
    "a": "a", "aː": "a", "ɛ": "e", "eː": "e", "ɪ": "i", "i": "i", "iː": "i",
    "y": "i", "yː": "i", "ɔ": "o", "oː": "o", "ʊ": "u", "uː": "u",
    "ãː": "an", "ẽː": "en", "ĩː": "in", "õː": "on", "ũː": "un", "ỹː": "in",
    "ae̯": "ae", "oe̯": "oe", "au̯": "au", "eu̯": "eu", "ei̯": "ei", "ui̯": "ui",
}
_STRESSED = {
    "a": "à", "aː": "à", "ɛ": "è", "eː": "é", "ɪ": "ì", "i": "ì", "iː": "ì",
    "y": "ì", "yː": "ì", "ɔ": "ò", "oː": "ó", "ʊ": "ù", "uː": "ù",
    "ãː": "àn", "ẽː": "én", "ĩː": "ìn", "õː": "ón", "ũː": "ùn", "ỹː": "ìn",
    "ae̯": "àe", "oe̯": "óe", "au̯": "àu", "eu̯": "éu", "ei̯": "éi", "ui̯": "ùi",
}
_LONG_ORAL = frozenset({"aː", "eː", "iː", "oː", "uː", "yː"})
_NASAL = frozenset({"ãː", "ẽː", "ĩː", "õː", "ũː", "ỹː"})
_CONSONANTS = {
    "kʷ": "qu", "gʷ": "gu", "pʰ": "p", "tʰ": "t", "h": "", "ŋ": "n", "j": "i",
    "w": "u", "ɫ": "l", "l": "l", "z": "z",
    "p": "p", "b": "b", "t": "t", "d": "d", "f": "f", "m": "m", "n": "n", "r": "r",
}


def respell_it(ipa: str) -> str:
    """Respell a Classical Latin IPA line in Italian orthography."""
    return spell_items(parse_ipa(ipa))


# --- IPA parsing ----------------------------------------------------------

def parse_ipa(ipa: str) -> list:
    items: list = []
    for token in ipa.split():
        end = len(token)
        while end and token[end - 1] in _PAUSE_CHARS:
            end -= 1
        if end:
            items.append(_parse_word(token[:end]))
        items.extend(token[end:])
    return items


def _parse_word(word: str) -> list[Syl]:
    syls = [Syl(False)]
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in "ˈ.":
            if syls[-1].phones:
                syls.append(Syl(False))
            if ch == "ˈ":
                syls[-1].stressed = True
            i += 1
            continue
        diph = next((d for d in _DIPHTHONGS if word.startswith(d, i)), "")
        if diph:
            syls[-1].phones.append(diph)
            i += len(diph)
            continue
        phone = ch
        i += 1
        while i < len(word) and word[i] in _MODIFIERS:
            phone += word[i]
            i += 1
        syls[-1].phones.append(phone)
    return [s for s in syls if s.phones]


# --- spelling -------------------------------------------------------------

def spell_items(items: list) -> str:
    """Spell analysed words and pauses as one Italian line."""
    out = ""
    start = True
    for item in items:
        if isinstance(item, str):
            out += item
            if item in ".!?":
                start = True
            continue
        word = spell_word(item)
        if start and word:
            word = word[0].upper() + word[1:]
            start = False
        out += (" " if out else "") + word
    return out


def _front(ph: str) -> bool:
    return is_vowel(ph) and ph[0] in "eɛiɪyẽĩỹ"


def _i_like(ph: str) -> bool:
    return is_vowel(ph) and ph[0] in "iɪyĩỹ"


def spell_word(word: list[Syl]) -> str:
    flat = [(ph, si) for si, syl in enumerate(word) for ph in syl.phones]
    last = len(word) - 1
    out: list[str] = []
    j = 0
    while j < len(flat):
        ph, si = flat[j]
        prev = flat[j - 1][0] if j else ""
        nxt = flat[j + 1][0] if j + 1 < len(flat) else ""
        nxt2 = flat[j + 2][0] if j + 2 < len(flat) else ""
        if is_vowel(ph):
            stressed = word[si].stressed
            text = _STRESSED[ph] if stressed else _PLAIN[ph]
            if ph in _LONG_ORAL:
                syl = word[si].phones
                is_open = syl.index(ph) == len(syl) - 1
                before_ss = nxt == "s" and is_vowel(nxt2)
                if not (stressed and is_open and si != last and not before_ss):
                    text += _PLAIN[ph]
            out.append(text)
        elif ph in ("k", "kʰ"):
            if nxt == "s":
                out.append("x")
                j += 2
                continue
            out.append("ch" if _front(nxt) else "c")
        elif ph == "g":
            soft = _front(nxt) or (nxt in ("l", "ɫ") and _i_like(nxt2))
            out.append("gh" if soft else "g")
        elif ph == "s":
            oral = is_vowel(prev) and prev not in _NASAL
            out.append("ss" if oral and is_vowel(nxt) else "s")
        else:
            out.append(_CONSONANTS.get(ph, ""))
        j += 1
    return "".join(out)
