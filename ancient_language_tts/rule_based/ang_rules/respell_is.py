"""Old English IPA → Icelandic respelling. Implements `data/ang/respelling/is.txt`."""

from __future__ import annotations

from ._common import is_vowel, parse_ipa, render

_V = {
    "ɑ": "a", "ɑː": "a", "æ": "e", "æː": "e", "e": "e", "eː": "e",
    "i": "i", "iː": "í", "o": "o", "oː": "ó", "u": "ú", "uː": "ú",
    "y": "u", "yː": "u",
    "æɑ̯": "ea", "æːɑ̯": "ea", "eo̯": "eo", "eːo̯": "eo",
}
_C = {
    "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "m": "m", "s": "s",
    "g": "g", "ɣ": "g", "j": "j", "x": "g", "ç": "g", "h": "h", "ʍ": "hv",
    "l̥": "hl", "r̥": "hr", "n̥": "hn", "w": "v", "f": "f", "v": "v",
    "θ": "þ", "ð": "ð", "z": "s", "r": "r", "l": "l", "n": "n",
    "tʃ": "tj", "dʒ": "dj", "ʃ": "sj", "ŋ": "n",
}
# Geminates Icelandic would misread when doubled.
_GEM = {"l": "l", "p": "p", "t": "t", "k": "k", "θ": "þ", "ʃ": "sj", "x": "g", "ç": "g"}
_NN_SINGLE = frozenset({"iː", "oː", "u", "uː"})  # ú, í, ó: nn reads [tn]
_TWO_SYLLABLE_PREFIXES = frozenset({"o.fer", "o.ver", "un.der"})


def respell(ipa: str) -> str:
    """Respell an Old English IPA line in Icelandic orthography."""
    return spell_items(parse_ipa(ipa.strip()))


def spell_items(items: list[tuple]) -> str:
    return render(items, spell_word)


def spell_word(sylls: list[list[str]], stress: int | None) -> str:
    if not stress:
        return _seq(sylls)
    pre, words, i = sylls[:stress], [], 0
    while i < len(pre):
        if i + 1 < len(pre) and "".join(pre[i]) + "." + "".join(pre[i + 1]) in _TWO_SYLLABLE_PREFIXES:
            words.append(_seq(pre[i:i + 2]))
            i += 2
        else:
            words.append(_seq(pre[i:i + 1]))
            i += 1
    words.append(_seq(sylls[stress:]))
    return " ".join(w for w in words if w)


def _seq(sylls: list[list[str]]) -> str:
    flat = [(p, si) for si, syl in enumerate(sylls) for p in syl]
    out: list[str] = []
    k = 0
    while k < len(flat):
        p, si = flat[k]
        nxt = flat[k + 1] if k + 1 < len(flat) else None
        if nxt and nxt[0] == p and nxt[1] != si and not is_vowel(p):
            if p == "n":
                out.append("n" if _long_before(flat, k) else "nn")
            elif p in _GEM:
                out.append(_GEM[p])
            else:
                c = _C.get(p, p)
                out.append(c + c)
            k += 2
            continue
        if is_vowel(p):
            out.append(_V.get(p, p))
        elif (p == "j" and k > 0 and flat[k - 1][0] in ("æ", "æː", "e", "eː")
              and (nxt is None or not is_vowel(nxt[0]))):
            out.append("i")
        else:
            out.append(_C.get(p, p))
        k += 1
    return "".join(out)


def _long_before(flat: list[tuple[str, int]], k: int) -> bool:
    for p, _ in reversed(flat[:k]):
        if is_vowel(p):
            return p in _NN_SINGLE or p.endswith("̯")
    return False
