"""Early Modern English (London, c. 1600) text → IPA, by rule.

Implements `data/eme/phonology.txt`. The spelling is read the way the spec
asks: digraphs, silent letters, magic e, open and closed syllables, prefixes
and suffixes, and the lexical sets in `_lexicon.py` where the spelling cannot
tell (meat vs head, love vs move). The values are the 1600 values, not
today's.

`analyze(text)` gives the words as phonemes and syllables. `to_ipa` formats
that analysis; `direct_en` spells the same analysis without an IPA string.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import _lexicon as L

VOWELS = frozenset({
    "iː", "eː", "ɛ", "ɪ", "ə", "a", "ɔ", "ɔː", "oː", "uː", "ʊ",
    "əɪ", "əʊ", "ɔɪ", "ɪu",
})
_IPA_TOKENS = sorted(
    VOWELS | {"tʃ", "dʒ", "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s",
              "z", "ʃ", "ʒ", "m", "n", "ŋ", "l", "r", "j", "w", "ʍ", "h",
              "ˈ", "."},
    key=len, reverse=True,
)
_IPA_RE = re.compile("|".join(re.escape(t) for t in _IPA_TOKENS))

_VOWEL_LETTERS = frozenset("aeiou")
_STOP_F = frozenset("pbtdkgf")
_SIBILANT = frozenset({"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"})
_VOICELESS = frozenset({"p", "t", "k", "f", "θ", "s", "ʃ", "tʃ"})
_LONG = {"a": "eː", "e": "iː", "i": "əɪ", "y": "əɪ", "o": "oː", "u": "ɪu"}
_SHORT = {"a": "a", "e": "ɛ", "i": "ɪ", "y": "ɪ", "o": "ɔ", "u": "ʊ"}
_ONSETS = frozenset("""bl br cl cr dr fl fr gl gr pl pr sc sk sl sm sn sp st sw
    tr tw th sh ch wh wr ph qu kn gn""".split())
# Consonant spellings that close the syllable before them (two sounds).
_CLOSING = frozenset({"ng", "nk", "x", "tch", "sch", "qu", "ngu"})
_DIGRAPHS = ("ee", "ea", "ei", "ey", "ai", "ay", "oa", "oo", "ou", "ow", "oi",
             "oy", "au", "aw", "ew", "eu", "ue", "ui", "ie", "oe", "uy", "ye",
             "ae")

# Prefixes (spec: be-, de-, re-, e-, pre- are [ɪ]; a prefix boundary
# overrides syllabification). Longest first.
_PREFIXES = (
    ("with", ["w", "ɪ", "ð"]), ("dis", ["d", "ɪ", "s"]), ("mis", ["m", "ɪ", "s"]),
    ("con", ["k", "ə", "n"]), ("com", ["k", "ə", "m"]), ("pre", ["p", "r", "ɪ"]),
    ("pro", ["p", "r", "ə"]), ("per", ["p", "ə", "r"]), ("for", ["f", "ə", "r"]),
    ("be", ["b", "ɪ"]), ("de", ["d", "ɪ"]), ("re", ["r", "ɪ"]),
    ("un", ["ə", "n"]), ("ad", ["ə", "d"]), ("ob", ["ə", "b"]),
    ("ex", ["ɛ", "k", "s"]), ("a", ["ə"]),
)

# Endings that put the stress on the syllable before them.
_ATTRACT = ("ssion", "tion", "sion", "cion", "cian", "tian", "gion", "geon",
            "cious", "tious", "xious", "ious", "eous", "ial", "ian", "ient",
            "ience", "iency", "iant", "iance", "ical", "ics", "ic", "ity",
            "ety", "ify")
_FINAL_STRESS = ("eer", "ee", "oon", "ette", "esque", "ique", "self", "selves",
                 "ade")


@dataclass
class Phone:
    sym: str
    stress: bool = False
    glide: bool = False  # unstressed ɪ before a vowel: [j] in verse

    @property
    def vowel(self) -> bool:
        return self.sym in VOWELS


@dataclass
class Word:
    key: str
    phones: list[Phone]
    bounds: set[int] = field(default_factory=set)
    stressed: bool = True
    aux: bool = False
    alt: tuple[list[Phone], set[int]] | None = None  # verse form, one syllable fewer

    def syllables(self) -> list[tuple[bool, list[str]]]:
        """[(stressed, phones)]."""
        out: list[tuple[bool, list[str]]] = []
        for syl in syllabify(self.phones, self.bounds):
            stress = self.stressed and any(p.stress for p in syl)
            out.append((stress, [p.sym for p in syl]))
        return out

    def count(self) -> int:
        return max(1, sum(1 for p in self.phones if p.vowel))

    def first_phone(self) -> str:
        return self.phones[0].sym if self.phones else ""

    def glides(self) -> list[int]:
        return [i for i, p in enumerate(self.phones) if p.glide]


# --------------------------------------------------------------------------
# Public API


def to_ipa(text: str) -> str:
    """Transcribe Early Modern English `text` into IPA (phonology.txt)."""
    out: list[str] = []
    for item in analyze(text):
        if isinstance(item, str):
            if out and out[-1] == " ":
                out.pop()
            out += [item, " "]
        else:
            out += [format_word(item), " "]
    return "".join(out).strip()


def format_word(word: Word) -> str:
    parts: list[str] = []
    for index, (stress, syms) in enumerate(word.syllables()):
        text = "".join(syms)
        if stress:
            parts.append("ˈ" + text)
        else:
            parts.append(("." if index else "") + text)
    return "".join(parts)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’]+|[,.!?;:]|\n")


def analyze(text: str) -> list[Word | str]:
    """Words (as `Word`) and pause marks (`, ! ? .`), in order."""
    items: list[Word | str] = []
    lines: list[int] = []  # item index where each new line starts
    for match in _TOKEN_RE.finditer(text.strip()):
        tok = match.group()
        if tok == "\n":
            lines.append(len(items))
            continue
        if tok in ",.!?;:":
            items.append("." if tok in ";:" else tok)
            continue
        word = _make_word(tok)
        if word is not None:
            items.append(word)
    _context(items)
    _verse(items, lines)
    return items


# --------------------------------------------------------------------------
# Sentence context: weak forms, auxiliaries, attributive stress


def _context(items: list[Word | str]) -> None:
    for i, item in enumerate(items):
        if isinstance(item, str):
            continue
        nxt = items[i + 1] if i + 1 < len(items) else None
        nxt_word = nxt if isinstance(nxt, Word) else None
        if item.key == "to":
            sym = "ʊ" if nxt_word and nxt_word.phones and nxt_word.phones[0].vowel else "ə"
            item.phones = [Phone("t"), Phone(sym)]
        if item.aux:
            item.stressed = nxt_word is None or nxt_word.key in L.MAIN_AFTER
        if item.key in L.ATTRIBUTIVE and nxt_word is not None:
            if nxt_word.stressed and not nxt_word.aux and nxt_word.key not in L.MAIN_AFTER:
                item.phones, item.bounds = _parse_ipa(L.ATTRIBUTIVE[item.key])


# --------------------------------------------------------------------------
# Verse: ten syllables, or eleven with an unstressed ending (spec, Stress).


def _verse(items: list[Word | str], lines: list[int]) -> None:
    units: list[list[Word]] = []
    current: list[Word] = []
    breaks = set(lines)
    for i, item in enumerate(items):
        if lines and i in breaks and current:
            units.append(current)
            current = []
        if isinstance(item, Word):
            current.append(item)
        elif not lines and item in ".!?":
            if current:
                units.append(current)
            current = []
    if current:
        units.append(current)
    for unit in units:
        _fit_meter(unit)


def _fit_meter(unit: list[Word]) -> None:
    n = sum(w.count() for w in unit)
    if n <= 10:
        return
    last = unit[-1]
    sylls = last.syllables()
    feminine = not sylls[-1][0]

    def allowed(k: int) -> bool:
        return k % 10 == 0 or (feminine and k % 10 == 1)

    if allowed(n):
        return
    candidates: list[tuple[Word, int | None]] = []
    for w in unit:
        for i in w.glides():
            candidates.append((w, i))
    for w in unit:
        if w.alt is not None:
            candidates.append((w, None))
    for target in range(n - 1, max(n - 3, 9), -1):
        if allowed(target) and n - target <= len(candidates):
            for w, i in candidates[: n - target]:
                if i is None:
                    w.phones, w.bounds = w.alt  # type: ignore[misc]
                    w.alt = None
                else:
                    w.phones[i] = Phone("j")
            return


# --------------------------------------------------------------------------
# One word


def _make_word(tok: str) -> Word | None:
    key, syl_ed = _normalize(tok)
    if not key:
        return None
    if key not in L.LEX:
        key = key.strip("'")
        if not key:
            return None
    phones, bounds = _pron(key, syl_ed=syl_ed, top=True)
    if not phones:
        return None
    word = Word(key=key, phones=phones, bounds=bounds)
    if key in L.FUNCTION:
        word.stressed = False
    elif key in L.AUX:
        word.aux = True
        word.stressed = False
    if key in L.VERSE_SHORT:
        word.alt = _parse_ipa(L.VERSE_SHORT[key])
    _mark_glides(word.phones)
    return word


def _normalize(tok: str) -> tuple[str, bool]:
    s = tok.replace("’", "'")
    nfd = unicodedata.normalize("NFD", s)
    syl_ed = bool(re.search("e[̀́]d$", nfd.lower()))
    s = "".join(ch for ch in nfd if not unicodedata.combining(ch)).lower()
    s = s.replace("æ", "ae").replace("œ", "e")
    return _modernize(s), syl_ed


def _modernize(s: str) -> str:
    """Original spelling → the same word in modern spelling (spec)."""
    if s in L.ORIGINAL:
        return L.ORIGINAL[s]
    chars = list(s)
    if len(chars) > 1 and chars[0] == "v" and chars[1] not in _VOWEL_LETTERS:
        chars[0] = "u"  # vnto, vp
    if len(chars) > 2 and chars[0] == "i" and chars[1] in _VOWEL_LETTERS:
        chars[0] = "j"  # iust, ioy
    for i in range(1, len(chars) - 1):
        prev, nxt = chars[i - 1], chars[i + 1]
        if chars[i] == "u" and prev in _VOWEL_LETTERS and nxt in _VOWEL_LETTERS:
            chars[i] = "v"  # loue, haue, euer
        elif chars[i] == "i" and prev in "aeo" and nxt in "aeou" and i > 1:
            chars[i] = "j"  # maiestie
    s = "".join(chars)
    if s in L.ORIGINAL:
        return L.ORIGINAL[s]
    m = re.match(r"^(.*[aeiouy].*?)([bcdfgkmnprtvz])\2e$", s)
    if m:
        s = m.group(1) + m.group(2)  # sonne, starre
    elif re.search(r"[aeiouy].*(ll|ss|ff)e$", s):
        s = s[:-1]  # goodnesse, stille
    return s


def _mark_glides(phones: list[Phone]) -> None:
    for i in range(1, len(phones) - 1):
        p = phones[i]
        p.glide = (
            p.sym == "ɪ" and not p.stress
            and not phones[i - 1].vowel and phones[i + 1].vowel
        )


def _parse_ipa(ipa: str) -> tuple[list[Phone], set[int]]:
    phones: list[Phone] = []
    bounds: set[int] = set()
    stress_next = False
    for tok in _IPA_RE.findall(ipa):
        if tok in ("ˈ", "."):
            if phones:
                bounds.add(len(phones))
            stress_next = stress_next or tok == "ˈ"
            continue
        p = Phone(tok)
        if stress_next and p.vowel:
            p.stress = True
            stress_next = False
        phones.append(p)
    if not any(p.stress for p in phones):
        for p in phones:
            if p.vowel:
                p.stress = True
                break
    # A bound only matters between two vowels; keep it where it falls.
    return phones, bounds


def _vowel_groups(s: str) -> int:
    return len(re.findall(r"[aeiouy]+", s))


def _pron(s: str, syl_ed: bool = False, top: bool = False,
          override: str | None = None, strip: bool = True) -> tuple[list[Phone], set[int]]:
    if s in L.LEX and (top or s not in L.FUNCTION):
        return _parse_ipa(L.LEX[s])
    if override is None:
        override = L.STRESSED_VOWEL.get(s)
    pre = _prefix(s)
    if pre is not None:
        name, root = pre
        r_ph, r_b = _pron(root, syl_ed=syl_ed, override=L.STRESSED_VOWEL.get(s))
        p_ph = [Phone(x) for x in dict(_PREFIXES)[name]]
        if name in ("de", "re", "pre") and len(r_ph) > 1 and r_ph[0].sym == "s" and r_ph[1].vowel:
            r_ph[0] = Phone("z")
        if r_ph and p_ph[-1].sym == r_ph[0].sym:
            p_ph = p_ph[:-1]
        shift = len(p_ph)
        return p_ph + r_ph, {shift} | {b + shift for b in r_b}
    suf = _suffix(s, syl_ed) if strip else None
    if suf is not None:
        stem, kind = suf
        # A stem is not stripped again: horses → horse, not hors → hor.
        ph, b = _pron(stem, override=L.STRESSED_VOWEL.get(stem, override),
                      strip=kind == "SAME")
        return _attach(ph, kind, syl_ed), b
    return _core(s, override)


def _prefix(s: str) -> tuple[str, str] | None:
    if "'" in s or _vowel_groups(s) < 2:
        return None
    if any(s.startswith(w) for w in L.NO_PREFIX):
        return None
    for name, _ph in _PREFIXES:
        if not s.startswith(name):
            continue
        root = s[len(name):]
        if name == "a":
            return (name, root) if s in L.A_WORDS else None
        if name == "with":
            return (name, root) if s in L.WITH_WORDS else None
        if name == "for":
            return (name, root) if s in L.FOR_WORDS else None
        if len(root) < 3 or not re.search("[aeiouy]", root):
            continue
        if root[0] in "aeiouy" or (name == "un" and len(root) < 3):
            continue
        if root[1] not in "aeiouy" and root[:2] not in _ONSETS:
            continue
        if name in ("be", "de", "re", "pre") and root[0] == root[1]:
            continue
        if name == "com" and root[0] not in "bmp":
            continue
        if name == "con" and root[0] in "bmp":
            continue
        return name, root
    return None


def _restore_e(stem: str) -> str:
    """lov → love, nam → name (the e dropped before -ed, -ing)."""
    if re.fullmatch(r"[^aeiouy]*[aeiouy][bcdfgklmnprstvz]", stem):
        return stem + "e"
    return stem


def _fix_stem(stem: str) -> str:
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouylsfz":
        return stem[:-1]  # stopped → stop
    return _restore_e(stem)


def _known(stem: str) -> bool:
    return stem in L.LEX or stem in L.STRESSED_VOWEL


def _suffix(s: str, syl_ed: bool) -> tuple[str, str] | None:
    def vowel(x: str) -> bool:
        return bool(re.search("[aeiouy]", x))

    if s.endswith("'s") and len(s) > 2:
        return s[:-2], "S"
    if s.endswith("s'") and len(s) > 2:
        return s[:-1], "SAME"
    if s.endswith("'d") and len(s) > 2:
        return _fix_stem(s[:-2]), "D"
    if s.endswith("'st") and len(s) > 3:
        return s[:-3], "ST"
    if "'" in s:
        return s.replace("'", ""), "SAME"
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y", "S"
    if s.endswith("ied") and len(s) > 4:
        return s[:-3] + "y", "ED"
    if s.endswith("es") and re.search(r"(s|x|z|sh|ch)es$", s) and vowel(s[:-2]):
        return s[:-2], "S"
    if (s.endswith("s") and len(s) > 2 and not s.endswith(("ss", "us", "is"))
            and vowel(s[:-1]) and s[-2] not in "ui"):
        return s[:-1], "S"
    if s.endswith("ed") and len(s) > 3 and s not in L.ED_ROOT and not s.endswith("eed"):
        stem = s[:-2]
        if vowel(stem):
            return _fix_stem(stem), "ED"
    if s.endswith("ing") and len(s) > 4 and vowel(s[:-3]):
        return _fix_stem(s[:-3]), "ING"
    if (s.endswith("ly") and len(s) > 4 and vowel(s[:-2]) and not s.endswith("ply")
            and (s[-3] not in "aeiouyl" or s[-3] == "e")):
        return s[:-2], "LY"
    for ending, kind in (("ness", "NESS"), ("less", "LESS"), ("ful", "FUL")):
        stem = s[: -len(ending)]
        if s.endswith(ending) and len(stem) >= 3 and vowel(stem):
            return stem, kind
    if s.endswith("ment") and len(s) >= 8 and vowel(s[:-4]) and s[-5] not in "aiouy":
        return s[:-4], "MENT"
    for ending, kind in (("est", "EST"), ("eth", "ETH")):
        stem = s[: -len(ending)]
        if s.endswith(ending) and len(stem) >= 2:
            if _known(stem + "e"):
                return stem + "e", kind
            if _known(stem):
                return stem, kind
    if s.endswith("er") and s[:-2] in L.SING_VERBS:
        return s[:-2], "ER"
    return None


_SUFFIX_PHONES = {
    "ING": ["ɪ", "n"], "LY": ["l", "ɪ"], "NESS": ["n", "ɪ", "s"],
    "LESS": ["l", "ɪ", "s"], "FUL": ["f", "ə", "l"], "MENT": ["m", "ə", "n", "t"],
    "EST": ["ɪ", "s", "t"], "ETH": ["ɪ", "θ"], "ER": ["ə", "r"], "ST": ["s", "t"],
}


def _attach(phones: list[Phone], kind: str, syl_ed: bool) -> list[Phone]:
    last = phones[-1].sym if phones else ""
    if kind == "SAME":
        return phones
    if kind == "S":
        add = ["ɪ", "z"] if last in _SIBILANT else (["s"] if last in _VOICELESS else ["z"])
    elif kind == "ED":
        if syl_ed or last in ("t", "d"):
            add = ["ɪ", "d"]
        else:
            add = ["t"] if last in _VOICELESS else ["d"]
    elif kind == "D":
        add = ["t"] if last in _VOICELESS else ["d"]
    else:
        add = list(_SUFFIX_PHONES[kind])
    if add and add[0] == last and add[0] not in VOWELS:
        add = add[1:]
    return phones + [Phone(x) for x in add]


# --------------------------------------------------------------------------
# Spelling → phonemes for one simple stem


@dataclass
class _U:
    kind: str  # "V" nucleus, "C" consonant, "E" silent final e
    text: str
    pos: int
    phones: list[str] = field(default_factory=list)
    double: bool = False
    silent: bool = False
    vowel: str = ""


def _is_vowel_at(s: str, i: int) -> bool:
    c = s[i]
    nxt = s[i + 1] if i + 1 < len(s) else ""
    if c in _VOWEL_LETTERS:
        return True
    if c == "y":
        return not (nxt in _VOWEL_LETTERS and (i == 0 or s[i - 1] in _VOWEL_LETTERS))
    if c == "w" and i > 0 and s[i - 1] in "aeo":
        if s[i - 1] == "a" and nxt in _VOWEL_LETTERS and not (nxt == "e" and i + 2 == len(s)):
            return False
        return True
    return False


def _scan(s: str) -> list[_U]:
    units: list[_U] = []
    n = len(s)
    i = 0
    while i < n:
        if _is_vowel_at(s, i):
            j = i
            while j < n and _is_vowel_at(s, j):
                j += 1
            units += _split_group(s, i, j, units)
            i = j
            continue
        i = _consonant(s, i, units)
    return units


def _split_group(s: str, i: int, j: int, before: list[_U]) -> list[_U]:
    group = s[i:j]
    n = len(s)
    has_nucleus = any(u.kind == "V" for u in before)
    if group == "e" and j == n and has_nucleus and before and before[-1].kind == "C":
        prev = before[-1]
        if (prev.text in ("l", "r") and len(before) > 1 and before[-2].kind == "C"
                and not before[-2].silent):
            # noble, sepulchre: a syllable [əl], [ər] before the l or r.
            before.insert(len(before) - 1, _U("V", "ə", prev.pos, vowel="ə"))
            return []
        return [_U("E", "e", i)]
    out: list[_U] = []
    k = 0
    rest_after = s[j:]
    while k < len(group):
        pos = i + k
        tri = group[k:k + 3]
        pair = group[k:k + 2]
        if tri in ("eau", "ieu", "iew"):
            out.append(_U("V", tri, pos))
            k += 3
            continue
        if len(pair) == 2 and pair in _DIGRAPHS and not _hiatus(s, pair, k + 2 == len(group), rest_after):
            out.append(_U("V", pair, pos))
            k += 2
            continue
        out.append(_U("V", group[k], pos))
        k += 1
    # A final lone e after a vowel letter group belongs to that group (due).
    return out


def _hiatus(s: str, pair: str, at_end: bool, rest: str) -> bool:
    if not at_end:
        return False
    if pair == "ie":
        return rest.startswith(("nt", "nc")) or rest in ("t", "ty", "ts")
    if pair == "ue":
        return rest not in ("", "s", "d")
    if pair == "oe":
        return rest not in ("", "s", "d", "th", "st")
    if pair == "ea":
        return s.startswith(L.EA_HIATUS)
    if pair == "ui":
        return s.startswith(L.UI_HIATUS)
    return False


def _consonant(s: str, i: int, units: list[_U]) -> int:
    n = len(s)
    rest = s[i:]
    c = s[i]
    nxt = s[i + 1] if i + 1 < n else ""
    nxt2 = s[i + 2] if i + 2 < n else ""

    def add(text: str, phones: list[str], step: int, **kw) -> int:
        units.append(_U("C", text, i, list(phones), **kw))
        return i + step

    if rest.startswith("tch"):
        return add("tch", ["tʃ"], 3)
    if rest.startswith("sch"):
        return add("sch", ["s", "k"], 3)
    if rest.startswith("ch"):
        hard = s in L.CH_K or nxt2 == "r" or any(s.startswith(w) for w in ("chor", "christ", "chao"))
        return add("ch", ["k"] if hard else ["tʃ"], 2)
    if rest.startswith("sh"):
        return add("sh", ["ʃ"], 2)
    if rest.startswith("th"):
        return add("th", ["TH"], 2)
    if rest.startswith("ph"):
        return add("ph", ["f"], 2)
    if rest.startswith("wh"):
        return add("wh", ["h"] if s in L.WH_H else ["ʍ"], 2)
    if rest.startswith("gh"):
        return add("gh", ["GH"], 2)
    if rest.startswith("ck"):
        return add("ck", ["k"], 2, double=True)
    if rest.startswith("dg"):
        return add("dg", ["dʒ"], 2, double=True)
    if rest.startswith("qu"):
        return add("qu", ["k", "w"], 2)
    if rest.startswith("ngu") and i + 3 < n and s[i + 3] in _VOWEL_LETTERS:
        units.append(_U("C", "ng", i, ["ŋ", "g"]))
        return add("u", ["w"], 3)
    if rest.startswith("gu") and nxt2 in _VOWEL_LETTERS and nxt2:
        return add("gu", ["g"], 2)
    if rest.startswith("ng"):
        return add("ng", ["NG"], 2)
    if rest.startswith("nk"):
        return add("nk", ["ŋ", "k"], 2)
    if rest.startswith("gn"):
        return add("gn", ["GN"], 2)
    if rest in ("mb", "mn"):
        return add(rest, ["m"], 2)
    if rest.startswith("bt") and i > 0:
        return add("bt", ["t"], 2)
    if rest in ("sten", "stle") and s != "often":
        return add("st", ["s"], 2)
    if rest.startswith("rh"):
        return add("rh", ["r"], 2)
    if c == "x":
        return add("x", ["z"] if i == 0 else ["k", "s"], 1)
    if c == nxt and c not in "aeiouwyh":
        if c == "c":
            return add("cc", ["k", "s"] if nxt2 in "eiy" and nxt2 else ["k"], 2, double=True)
        if c == "s":
            return add("ss", ["s"], 2, double=True)
        return add(c + c, [_plain(c)], 2, double=True)
    if c == "c":
        return add("c", ["s"] if nxt and nxt in "eiy" else ["k"], 1)
    if c == "s" and nxt == "c" and nxt2 and nxt2 in "eiy":
        return add("sc", ["s"], 2)
    if c == "g":
        if nxt and nxt in "eiy" and s not in L.HARD_G:
            if nxt in "ei" and nxt2 and nxt2 in "aou":
                return add("g" + nxt, ["dʒ"], 2)  # region, George
            return add("g", ["dʒ"], 1)
        return add("g", ["g"], 1)
    if c == "j":
        return add("j", ["dʒ"], 1)
    if c == "q":
        return add("q", ["k"], 1)
    if c == "y":
        return add("y", ["j"], 1)
    if c == "h":
        return add("h", ["H"], 1)
    if c == "s":
        return add("s", ["S"], 1)
    if c.isalpha():
        return add(c, [_plain(c)], 1)
    return i + 1


def _plain(c: str) -> str:
    return {"c": "k", "j": "dʒ", "q": "k", "y": "j"}.get(c, c)


def _core(s: str, override: str | None) -> tuple[list[Phone], set[int]]:
    units = _scan(s)
    nuclei = [k for k, u in enumerate(units) if u.kind == "V"]
    if not nuclei:
        phones = [Phone(p) for u in units for p in _resolve_consonant(s, units, units.index(u)) ]
        return phones, set()
    stress = _stress(s, units, nuclei)
    mono = len(nuclei) == 1
    for rank, k in enumerate(nuclei):
        units[k].vowel = _vowel(s, units, nuclei, rank, rank == stress or mono,
                                override if rank == stress else None)
    phones: list[Phone] = []
    for k, u in enumerate(units):
        if u.kind == "V":
            phones.append(Phone(u.vowel, stress=(nuclei.index(k) == stress)))
        elif u.kind == "C" and not u.silent:
            phones += [Phone(p) for p in _resolve_consonant(s, units, k)]
    return phones, set()


def _glideable(units: list[_U], k: int, first: int) -> bool:
    u = units[k]
    return (u.text in ("i", "e") and k != first and k + 1 < len(units)
            and units[k + 1].kind == "V" and units[k - 1].kind == "C")


def _stress(s: str, units: list[_U], nuclei: list[int]) -> int:
    n = len(nuclei)
    if n == 1:
        return 0
    if s.endswith(_FINAL_STRESS):
        return n - 1
    if s not in L.IC_INITIAL:
        for ending in _ATTRACT:
            if s.endswith(ending) and len(s) > len(ending):
                start = len(s) - len(ending)
                before = [r for r, k in enumerate(nuclei) if units[k].pos < start]
                if before:
                    return before[-1]
                break
    return 0


def _following(units: list[_U], k: int) -> tuple[list[_U], str]:
    """Consonant units after unit k, and what ends them: V, E or end."""
    fol: list[_U] = []
    j = k + 1
    while j < len(units) and units[j].kind == "C":
        fol.append(units[j])
        j += 1
    end = units[j].kind if j < len(units) else "end"
    return fol, end


def _vowel(s: str, units: list[_U], nuclei: list[int], rank: int, stressed: bool,
           override: str | None) -> str:
    k = nuclei[rank]
    u = units[k]
    label = u.text
    fol, end = _following(units, k)
    after = s[u.pos + len(label):]
    hiatus = not fol and end == "V"
    final = not fol and end == "end"
    is_last = rank == len(nuclei) - 1
    first_r = bool(fol) and fol[0].text == "r"
    coda_r = first_r and (len(fol) > 1 or end == "end")
    r_magic = first_r and len(fol) == 1 and end == "E"
    magic = len(fol) == 1 and end == "E" and not fol[0].double
    if label == "ə":
        return "ə"
    if stressed:
        if override:
            _silence_l(u, fol, label, after)
            return override
        if len(label) > 1:
            return _digraph_stressed(label, after, coda_r or r_magic, final, len(nuclei))
        return _single_stressed(s, units, nuclei, rank, fol, end, after,
                                coda_r, r_magic, magic, hiatus, final)
    if len(label) > 1:
        return _digraph_unstressed(label, after, final, is_last)
    v = label
    if v in "iy":
        if r_magic or after.startswith(("gh", "gn")):
            return "əɪ"
        if magic and fol[0].phones == ["S"] or after.startswith("ze"):
            return "əɪ"
        return "ɪ"
    if v == "e":
        if hiatus or (u.pos == 0 and rank == 0):
            return "ɪ"  # the e- prefix: eternal, event
        if coda_r:
            return "ə"
        tail = after[:-1] if after.endswith("e") and end == "E" else after
        if is_last and tail in ("s", "st", "ss", "t", "th", "sh", "ch", "dge",
                                "ge", "ck", "d", "x", "ts", "tt"):
            return "ɪ"
        return "ə"
    if v == "a":
        if is_last and after in ("ge", "ce"):
            return "ɪ"
        return "ə"
    if v == "o":
        return "oː" if final else "ə"
    if v == "u":
        if final or hiatus:
            return "ɪu"
        if r_magic or coda_r:
            return "ə"
        if len(fol) == 1 and end == "V":
            return "ɪu"
        return "ə"
    return "ə"


def _silence_l(u: _U, fol: list[_U], label: str, after: str) -> None:
    if label in ("a", "o") and fol and fol[0].text == "l" and after[1:2] in ("k", "m", "f", "v"):
        if not (label == "o" and after[1:2] != "k"):
            fol[0].silent = True


def _digraph_stressed(label: str, after: str, r_after: bool, final: bool, n: int) -> str:
    if label == "ea":
        return "iː" if r_after else "eː"
    if label == "ou":
        return "ɔː" if after.startswith("ght") else "əʊ"
    if label == "ie":
        return "əɪ" if final and n == 1 else "iː"
    return {
        "ee": "iː", "ei": "eː", "ey": "eː", "ai": "eː", "ay": "eː", "oa": "oː",
        "oo": "uː", "ow": "əʊ", "oi": "əɪ", "oy": "ɔɪ", "au": "ɔː", "aw": "ɔː",
        "ew": "ɪu", "eu": "ɪu", "ue": "ɪu", "ui": "ɪu", "eau": "ɪu", "ieu": "ɪu",
        "iew": "ɪu", "oe": "oː", "uy": "əɪ", "ye": "əɪ", "ae": "iː",
    }.get(label, "ə")


def _digraph_unstressed(label: str, after: str, final: bool, is_last: bool) -> str:
    if label in ("ai", "ay"):
        return "ɪ" if is_last and after.startswith("n") else "eː"
    if label == "ey":
        return "ɪ" if final else "eː"
    return {
        "ou": "ə", "ow": "oː", "ie": "ɪ", "ei": "ɪ", "ee": "iː", "ea": "ə",
        "oo": "ʊ", "oa": "oː", "ew": "ɪu", "ue": "ɪu", "ui": "ɪu", "eu": "ɪu",
        "eau": "ɪu", "iew": "ɪu", "ieu": "ɪu", "au": "ɔː", "aw": "ɔː",
        "oi": "əɪ", "oy": "ɔɪ", "oe": "oː", "uy": "əɪ", "ye": "əɪ", "ae": "iː",
    }.get(label, "ə")


def _single_stressed(s: str, units: list[_U], nuclei: list[int], rank: int,
                     fol: list[_U], end: str, after: str, coda_r: bool,
                     r_magic: bool, magic: bool, hiatus: bool, final: bool) -> str:
    k = nuclei[rank]
    v = units[k].text
    if v == "a":
        if after.startswith("ll") and (len(after) == 2 or after[2] not in "aeiouy"):
            return "ɔː"
        if after.startswith("lk"):
            fol[0].silent = True
            return "ɔː"
        if after.startswith(("lm", "lf", "lv")):
            fol[0].silent = True
            return "a"
        if after.startswith(("lt", "ld")):
            return "ɔː"
        if (after.startswith("nge") and s not in L.HARD_G) or after in ("ste", "stes") or after.startswith("stry"):
            return "eː"
    if v == "o":
        if after.startswith(("ld", "lt")) or after == "ll":
            return "oː"
        if after.startswith("lk"):
            fol[0].silent = True
            return "oː"
        if after == "mb":
            return "oː"
    if v in "iy" and (after in ("nd", "ld", "gn", "mb") or after.startswith("gh")):
        return "əɪ"
    prev = units[k - 1] if k > 0 else None
    if coda_r:
        if v == "a":
            return "a"
        if v == "o":
            return "ə" if prev is not None and prev.text == "w" else "ɔː"
        return "ə"
    if r_magic:
        return _LONG[v]
    open_syl = end == "V" and (
        (len(fol) == 1 and not fol[0].double and fol[0].text not in _CLOSING)
        or (len(fol) == 2 and fol[0].text in _STOP_F and fol[1].text in ("l", "r"))
    )
    first = nuclei[0]
    eff_after = sum(1 for r in range(rank + 1, len(nuclei))
                    if not _glideable(units, nuclei[r], first))
    laxing = False
    if open_syl and v != "u" and eff_after == 1:
        last = units[nuclei[-1]]
        tail = s[last.pos:]
        if tail in ("y", "ie", "ey") and v in "eiy":
            laxing = True
        if tail in ("ic", "ics", "ish", "id", "it"):
            laxing = True
    long = (magic or final or hiatus
            or (open_syl and (v == "u" or eff_after == 1) and not laxing))
    return _LONG[v] if long else _SHORT[v]


def _resolve_consonant(s: str, units: list[_U], k: int) -> list[str]:
    u = units[k]
    prev = units[k - 1] if k > 0 else None
    nxt = units[k + 1] if k + 1 < len(units) else None
    prev_v = prev is not None and prev.kind == "V"
    next_v = nxt is not None and nxt.kind in ("V", "E")
    out: list[str] = []
    for p in u.phones:
        if p == "S":
            if u.double or s in L.VOICELESS_S:
                out.append("s")
            else:
                out.append("z" if prev_v and next_v else "s")
        elif p == "TH":
            voiced = (k == 0 and s in L.TH_VOICED) or (
                prev_v and next_v and s not in L.THETA_MEDIAL)
            out.append("ð" if voiced else "θ")
        elif p == "GH":
            if k == 0:
                out.append("g")
            elif s in L.GH_F:
                out.append("f")
        elif p == "NG":
            if nxt is not None and nxt.kind == "E":
                out += ["ŋ", "g"] if s in L.HARD_G else ["n", "dʒ"]
            elif nxt is not None and nxt.kind == "V":
                if nxt.text[0] in "eiy" and s not in L.HARD_G:
                    out += ["n", "dʒ"]
                else:
                    out += ["ŋ", "g"]
            elif nxt is not None and nxt.text in ("l", "r", "w"):
                out += ["ŋ", "g"]
            else:
                out.append("ŋ")
        elif p == "GN":
            out += ["g", "n"] if k == 0 or (nxt is not None and nxt.kind == "V") else ["n"]
        elif p == "H":
            if k == 0 and s.startswith(L.FRENCH_H):
                continue
            if prev_v and not (nxt is not None and nxt.kind == "V"):
                continue
            if prev is not None and prev.kind == "C" and prev.text in ("r", "x"):
                continue
            out.append("h")
        elif p == "t" and _before_glide(units, k):
            out.append("t" if prev is not None and prev.phones[-1:] == ["S"] else "s")
        else:
            out.append(p)
    return out


def _before_glide(units: list[_U], k: int) -> bool:
    nxt = units[k + 1] if k + 1 < len(units) else None
    after = units[k + 2] if k + 2 < len(units) else None
    return (nxt is not None and nxt.kind == "V" and nxt.text == "i"
            and after is not None and after.kind == "V" and nxt.vowel == "ɪ")


# --------------------------------------------------------------------------
# Syllables (spec, Syllables)


def syllabify(phones: list[Phone], bounds: set[int]) -> list[list[Phone]]:
    vidx = [i for i, p in enumerate(phones) if p.vowel]
    if len(vidx) < 2:
        return [phones]
    cuts: list[int] = []
    for a, b in zip(vidx, vidx[1:]):
        forced = [x for x in range(a + 1, b + 1) if x in bounds]
        if forced:
            cuts.append(forced[0])
            continue
        cons = list(range(a + 1, b))
        if not cons:
            cuts.append(b)
            continue
        start = cons[-1]
        if len(cons) >= 2:
            last, prev = phones[cons[-1]].sym, phones[cons[-2]].sym
            if (prev in _STOP_F and last in ("l", "r")) or (last in ("j", "w") and prev != "ŋ"):
                start = cons[-2]
        if phones[start].sym == "ŋ":
            start += 1
        cuts.append(start)
    out: list[list[Phone]] = []
    last_cut = 0
    for cut in cuts:
        out.append(phones[last_cut:cut])
        last_cut = cut
    out.append(phones[last_cut:])
    return out
