"""Middle English IPA → Modern Dutch respelling, by rule.

Implements `data/enm/respelling/nl.txt`.
"""

from __future__ import annotations

from ._common import VOWELS, Item, Word, flat, parse_ipa, render

_LONG = {"ɛː": "ê", "ɔː": "ô", "iː": "ie", "uː": "oe"}
_SHORT = {"a": "a", "ɛ": "e", "ɪ": "i", "ɔ": "o", "ʊ": "oe"}
_CONS = {
    "θ": "t", "ð": "d", "x": "ch", "ç": "ch", "ʍ": "hw",
    "ʃ": "sj", "tʃ": "tsj", "dʒ": "dzj",
}
# Spellings that may be doubled to close a short vowel.
_DOUBLE_OK = frozenset("bdfgklmnprstvz") | {"sj"}
# A Dutch syllable can begin with these + l/r.
_ONSET_FIRST = frozenset("pbtdkgfv")
_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
_VOWEL_LETTERS = frozenset("aeiouáéíóúêôë")
_EI = ("ei", "éí")


def respell(ipa: str) -> str:
    """Respell a Middle English IPA line for a Dutch reader."""
    return spell_items(parse_ipa(ipa))


def spell_items(items: list[Item]) -> str:
    return render(items, spell_word)


def spell_word(word: Word) -> str:
    sylls = word.syllables
    phones = flat(word)
    stress = word.stress_index()
    accent = stress is not None and stress > 0
    last_of = {}
    for k, (_ph, si) in enumerate(phones):
        last_of[si] = k
    out: list[str] = []
    double_next = False
    for k, (ph, si) in enumerate(phones):
        nxt = phones[k + 1] if k + 1 < len(phones) else None
        if ph in VOWELS:
            open_ = last_of[si] == k
            more = si + 1 < len(sylls)
            vowel_next = nxt is not None and nxt[0] in VOWELS
            sp = _vowel(ph, open_, more, vowel_next, out)
            if ph in ("a", "ɛ", "ɪ", "ɔ") and (sylls[si].stressed or ph == "ɛ") and more:
                double_next = _needs_double(sylls, si, open_)
            if accent and si == stress:
                sp = _acute(sp)
            out.append(sp)
            continue
        if ph == "ŋ":
            same_g = nxt is not None and nxt[0] == "g" and nxt[1] == si
            sp = "n" if same_g or (nxt is not None and nxt[0] == "k") else "ng"
        else:
            sp = _CONS.get(ph, ph)
        if double_next:
            if sp in _DOUBLE_OK:
                sp = sp[0] + sp
            double_next = False
        out.append(sp)
    return "".join(out)


def _vowel(ph: str, open_: bool, more: bool, vowel_next: bool, out: list[str]) -> str:
    if ph == "aː":
        return "a" if open_ and more else "aa"
    if ph == "eː":
        return "e" if open_ and more else "ee"
    if ph == "oː":
        return "o" if open_ else "oo"
    if ph in _LONG:
        return _LONG[ph]
    if ph in _SHORT:
        return _SHORT[ph]
    if ph == "ə":
        # ei ends in a glide, as in Dutch eieren: no diaeresis after it.
        if out and out[-1] and out[-1][-1] in _VOWEL_LETTERS and not out[-1].endswith(_EI):
            return "ë"
        return "e"
    if ph == "æi":
        return "ei"
    if ph == "au":
        return "auw" if vowel_next else "au"
    if ph == "ɔu":
        return "ouw" if vowel_next else "ou"
    if ph == "iu":
        return "ieuw"
    if ph == "ɛu":
        return "eeuw"
    if ph == "ɔi":
        return "oj" if vowel_next else "oi"
    return ""


def _needs_double(sylls, si: int, open_: bool) -> bool:
    """Close a short vowel's Dutch syllable by doubling the next consonant."""
    onset = sylls[si + 1].phones
    if not onset or onset[0] in VOWELS:
        return False
    if open_:
        return True
    phones = sylls[si].phones
    vi = max(i for i, p in enumerate(phones) if p in VOWELS)
    coda = phones[vi + 1:]
    return (
        len(coda) == 1
        and coda[0] != onset[0]
        and coda[0] in _ONSET_FIRST
        and onset[0] in ("l", "r")
    )


def _acute(sp: str) -> str:
    if sp.endswith(("ieuw", "eeuw")):
        base, glide = sp[:-2], sp[-2:]
    elif sp in ("auw", "ouw"):
        base, glide = sp[:-1], "w"
    elif sp == "oj":
        base, glide = "o", "j"
    else:
        base, glide = sp, ""
    return "".join(_ACUTE.get(ch, ch) for ch in base) + glide
