"""Middle English IPA → Modern English respelling, by rule.

Implements `data/enm/respelling/en.txt`.
"""

from __future__ import annotations

from ._common import Item, Word, flat, parse_ipa, render

_VOWEL = {
    "a": "ah", "aː": "ah", "ɛ": "eh", "ɛː": "eh", "eː": "ey", "ɪ": "ih", "iː": "ee",
    "ɔ": "aw", "ɔː": "aw", "oː": "oh", "ʊ": "uh", "uː": "oo", "ə": "uh",
    "æi": "eye", "au": "ow", "ɔu": "oh", "iu": "yoo", "ɛu": "ew", "ɔi": "oy",
}
_CONS = {
    "j": "y", "θ": "th", "ð": "th", "ʃ": "sh", "tʃ": "ch", "dʒ": "j",
    "x": "kh", "ç": "kh", "ʍ": "hw",
}


def respell(ipa: str) -> str:
    """Respell a Middle English IPA line for an English reader."""
    return spell_items(parse_ipa(ipa))


def spell_items(items: list[Item]) -> str:
    return render(items, spell_word)


def spell_word(word: Word) -> str:
    phones = flat(word)
    out: list[str] = []
    for k, (ph, si) in enumerate(phones):
        if ph in _VOWEL:
            out.append(_VOWEL[ph])
        elif ph == "ŋ":
            nxt = phones[k + 1] if k + 1 < len(phones) else None
            # ŋg in one syllable is ng; ŋ.g is ngg; ŋk is nk.
            if nxt and ((nxt[0] == "g" and nxt[1] == si) or nxt[0] == "k"):
                out.append("n")
            else:
                out.append("ng")
        else:
            out.append(_CONS.get(ph, ph))
    return "".join(out)
