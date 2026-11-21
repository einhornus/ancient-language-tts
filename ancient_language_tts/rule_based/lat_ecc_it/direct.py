"""Ecclesiastical Latin spelling → Italian respelling, in one pass.

No IPA. Letter rules follow `phonology.txt` and `respelling/it.txt`:
macrons only choose stress, ae/oe become e, and the Italian spelling is
what a modern Italian voice will say.
"""

from __future__ import annotations

import re
import unicodedata

_ACCENT = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
_CLITIC = frozenset("in ad ab ex de cum et sed si ut ne quam".split())
_FAKE_ENCLITIC = frozenset({"itaque", "utique", "undique", "denique"})
_FINAL_STRESS = frozenset({"illic", "adhuc", "istic", "illuc"})
_EU = ("heu", "eheu", "seu", "neu", "ceu", "neut", "europ", "orphe")
_EI = frozenset({"ei", "hei", "dein", "deinde", "deinceps"})
_J_WORDS = ("maior", "peior", "eius", "cuius", "huius", "troia")
_J_STEMS = ("iung", "iunx", "iunct", "iur", "iac", "ieci", "iect", "iuven", "iuv")
_SU = ("suav", "suad", "suesc", "suet")
_STOP = frozenset({"p", "t", "k", "c", "ch", "b", "d", "g", "gh", "f"})
_LONG = str.maketrans({
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ȳ": "y",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U", "Ȳ": "Y",
})
_TOKEN = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūȲȳÆæŒœÄäËëÏïÖöÜüŸÿ]+|[^\s]")
# A diaeresis vowel is in hiatus: "|" before it blocks every digraph.
_DIAERESIS = {"ä": "a", "ë": "e", "ï": "i", "ö": "o", "ü": "u", "ÿ": "y"}
_AE_HIATUS = ("michael", "raphael", "israel", "ismael", "misael", "nathanael")


class _A:
    __slots__ = ("text", "vowel", "heavy", "units")

    def __init__(self, text: str, vowel: bool = False, heavy: bool = False, units: int = 1):
        self.text = text
        self.vowel = vowel
        self.heavy = heavy
        self.units = 0 if vowel else units


def to_italian(text: str) -> str:
    """Respell Ecclesiastical Latin `text` as Italian, with no IPA step."""
    out: list[str] = []
    sentence = True
    for raw in _TOKEN.findall(text.strip()):
        if len(raw) == 1 and not raw.isalpha():
            mark = "." if raw in ";:" else raw
            if mark not in ",.!?":
                continue
            if out and out[-1] == " ":
                out.pop()
            out += [mark, " "]
            if mark == ".":
                sentence = True
            continue
        spelled = _word(raw)
        if sentence and spelled:
            spelled = spelled[0].upper() + spelled[1:]
            sentence = False
        out += [spelled, " "]
    return "".join(out).strip()


def _word(raw: str) -> str:
    s, longs = _prep(raw)
    parts = _atoms(s, longs)
    stress = _stress(parts, s)
    letters = [_accent(a.text) if a.vowel and i == stress else a.text for i, a in enumerate(parts)]
    return "".join(letters)


def _prep(raw: str) -> tuple[str, list[bool]]:
    chars: list[str] = []
    longs: list[bool] = []
    for ch in unicodedata.normalize("NFC", raw):
        if unicodedata.combining(ch):
            continue
        long = "MACRON" in unicodedata.name(ch, "") or ch in "āēīōūȳĀĒĪŌŪȲ"
        base = ch.translate(_LONG).lower()
        if base in _DIAERESIS:
            chars += ["|", _DIAERESIS[base]]
            longs += [False, False]
        elif base == "æ":
            chars += ["a", "e"]
            longs += [False, False]
        elif base == "œ":
            chars += ["o", "e"]
            longs += [False, False]
        else:
            chars.append(base)
            longs.append(long)
    s = "".join(chars)
    if s.startswith(_AE_HIATUS):
        k = s.index("ae") + 1
        s = s[:k] + "|" + s[k:]
        longs.insert(k, False)
    return s, longs


def _atoms(s: str, longs: list[bool]) -> list[_A]:
    out: list[_A] = []
    i, n = 0, len(s)

    def at(k: int) -> str:
        return s[i + k] if i + k < n else ""

    while i < n:
        c, d = s[i], at(1)
        pair = c + d
        if pair in ("ae", "oe"):
            out.append(_A("e", True, True))
            i += 2
        elif pair == "au":
            out.append(_A("au", True, True))
            i += 2
        elif pair == "eu" and any(s.startswith(p) for p in _EU):
            out.append(_A("eu", True, True))
            i += 2
        elif pair == "ei" and s in _EI and not longs[i + 1]:
            out.append(_A("ei", True, True))
            i += 2
        elif pair == "ph":
            out.append(_A("f"))
            i += 2
        elif pair == "th":
            out.append(_A("t"))
            i += 2
        elif pair == "rh":
            out.append(_A("r"))
            i += 2
        elif pair == "ch":
            out.append(_A("ch" if _front(s, i + 2) else "c"))
            i += 2
        elif pair == "qu":
            out.append(_A("qu"))
            i += 2
        elif c == "s" and d == "c" and at(2) == "h":
            out.append(_A("sch", units=2))
            i += 3
        elif c == "s" and d == "c" and _front(s, i + 2):
            out.append(_A("sc"))
            i += 2
        elif c == "x" and d == "c" and _front(s, i + 2):
            out.append(_A("k"))
            out.append(_A("sc"))
            i += 2
        elif i == 0 and pair == "ex" and at(2) and (at(2) in "aeiouyh|"):
            out.append(_A("e", True, longs[i]))
            out.append(_A("gs", units=2))
            i += 2
        elif i == 0 and pair == "ex" and at(2) == "s" and at(3) and at(3) in "aeiouy":
            out.append(_A("e", True, longs[i]))
            out.append(_A("gs", units=2))
            i += 3
        elif pair == "gn":
            out.append(_A("gn", units=2))
            i += 2
        elif c == "n" and d == "g" and at(2) == "u" and len(at(3)) == 1 and at(3) in "aeiouy":
            out.append(_A("n"))
            out.append(_A("gu"))
            i += 3
        elif c == "t" and d == "i" and len(at(2)) == 1 and at(2) in "aeiouy" and (i == 0 or s[i - 1] not in "stx") and s[i:] != "tier":
            out.append(_A("z"))
            out.append(_A("i", True, longs[i + 1]))
            i += 2
        elif c == "h":
            if s.startswith(("mihi", "nihil")):
                out.append(_A("ch"))
            i += 1
        elif c in "ij" and _i_glide(s, i):
            out.append(_A("i"))
            i += 1
        elif c == "y":
            out.append(_A("i", True, longs[i]))
            i += 1
        elif c == "u" and _u_glide(s, i):
            out.append(_A("v"))
            i += 1
        elif c in "aeiou":
            out.append(_A(c, True, longs[i]))
            i += 1
        elif c == "x":
            out.append(_A("x", units=2))
            i += 1
        elif c == "z":
            out.append(_A("z"))
            i += 1
        elif c == "g" and d == "l" and at(2) == "i":
            out.append(_A("gh"))
            i += 1
        elif c == "k":
            out.append(_A("ch" if d in {"e", "i", "y"} else "c"))
            i += 1
        elif c == "s" and i > 0 and i + 1 < n and s[i - 1] in "aeiouy" and s[i + 1] in "aeiouy":
            out.append(_A("ss"))
            i += 1
        elif c in "pbtdcgmnlrfvs":
            out.append(_A(c))
            i += 1
        else:
            i += 1
    return out


def _front(s: str, i: int) -> bool:
    if i < len(s) and s[i] == "|":
        i += 1
    return i < len(s) and (s[i] in "eiy" or s[i:i + 2] in ("ae", "oe"))


def _i_glide(s: str, i: int) -> bool:
    if s[i] == "j":
        return True
    if i + 1 >= len(s) or s[i + 1] not in "aeiouy":
        return False
    if i == 0:
        return True
    if any(s.startswith(w) for w in _J_WORDS) and s[i - 1] in "aeiouy":
        return True
    return any(s[i:].startswith(stem) for stem in _J_STEMS)


def _u_glide(s: str, i: int) -> bool:
    return i > 0 and s[i - 1] == "s" and i + 1 < len(s) and s[i + 1] in "aeiouy" and any(
        s.startswith(p) for p in _SU
    )


def _stress(parts: list[_A], word: str) -> int | None:
    vowels = [i for i, a in enumerate(parts) if a.vowel]
    if not vowels or word in _CLITIC:
        return None
    if word in _FINAL_STRESS:
        return vowels[-1]
    if word not in _FAKE_ENCLITIC and _enclitic(word) and len(vowels) >= 2:
        return vowels[-2]
    if len(vowels) <= 2:
        return vowels[0]
    penult, final = vowels[-2], vowels[-1]
    if parts[penult].heavy or _closed(parts, penult, final):
        return penult
    return vowels[-3]


def _enclitic(word: str) -> bool:
    if len(word) <= 2:
        return False
    if word.endswith(("que", "ve")):
        return True
    return word.endswith("ne") and word[-3] not in "aeiouy"


def _closed(parts: list[_A], left: int, right: int) -> bool:
    mid = [a for a in parts[left + 1:right] if not a.vowel]
    if not mid:
        return False
    if (
        len(mid) >= 2
        and mid[-2].text in _STOP
        and mid[-1].text in "lr"
        and mid[-2].units == 1
        and mid[-1].units == 1
    ):
        coda = mid[:-2]
        return bool(coda)
    return bool(mid[:-1]) or mid[-1].units >= 2


def _accent(text: str) -> str:
    return _ACCENT.get(text[0], text[0]) + text[1:]
