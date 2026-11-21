"""London Middle English (about 1400) text → IPA, by rule.

Implements `data/enm/phonology.txt`. The spelling does not show every
distinction the spec asks for (open vs close long e and o, ou as [uː] or
[ɔu], French vs native words, silent French h, cup-vowel o). Those come from
the small word lists below, with the spec's stated defaults elsewhere.
"""

from __future__ import annotations

import re
import unicodedata

from ._common import Item, Pause, Syllable, Word, format_ipa, parse_word

# --- word lists ---------------------------------------------------------------

# Monosyllables (and the possessives oure, youre) that take no stress mark.
_CLITICS = frozenset(
    "the a an of to in on at by with for fro and or but that as if whan his hir my thy oure youre".split()
)

# Whole words with a fixed transcription.
_FIXED = {
    "the": "ðə", "a": "a", "he": "ˈheː", "me": "ˈmeː", "she": "ˈʃeː", "we": "ˈweː",
    "be": "ˈbeː", "thee": "ˈðeː", "i": "ˈiː", "yen": "ˈiː.ən",
}
_YE_EYE = "ˈiː.ə"
_YE_YOU = "ˈjeː"
# A determiner or adjective before ye makes it the noun "eye".
_EYE_BEFORE = frozenset(
    "the his hir myn thyn myne thyne oure youre open an oon bothe two hise".split()
)

# th is [ð] at the start of these.
_TH_VOICED = frozenset(
    "the that this thise thes thou thee thy thyn thyne they thanne than then ther there "
    "therto therof therfore therwith thider though thogh thus thilke thennes".split()
)

# Long e that is open [ɛː] (modern ea, or short e from a long vowel).
_OPEN_E = frozenset(
    "heeth breeth eek ek clene mete speke grete greet breke heed deed deef leef teche lene ese"
    .split()
)
# Long o that is open [ɔː] (modern [oʊ], from Old English ā).
_OPEN_O = frozenset(
    "hooly hoolie hoom goon stoon so go no open hope hoost boot bothe oon noon ooth loth rood "
    "goost more moore sore soore boon bone stone home lore wo agoon ago"
    .split()
)
# ou / ow that is [ɔu] (modern [oʊ] or [ɔː]); ou before gh is [ɔu] anyway.
_OU_OPEN_STEMS = ("know", "grow", "blow", "throw", "snow", "crow", "soul", "sow", "folw")
_OU_OPEN = frozenset("owe owen lowe bowe slowe rowe".split())
# ew / eu that is [ɛu].
_EU_OPEN = frozenset("fewe lewed shewe shewen sheweth beautee beaute".split())
# i / y long [iː] in a closed syllable (modern [aɪ]).
_LONG_I = frozenset(
    "lyf lif wyf wif myn min thyn lyk whyt fyr fyn fyve wys wyse syde tyd ys lyves wyves by my thy"
    .split()
)
_LONG_A = frozenset({"cas", "caas"})
# o next to m, n, u, v, w that keeps [ɔ].
_NOT_CUP = frozenset(
    "on upon lond londe long longe longen song songe songes strong stronge wrong bond hond honde "
    "fond noble rome"
    .split()
)
# o that is [ʊ] for other reasons (modern cup / drought vowel).
_O_UU = frozenset({"droghte", "corage"})

# French and Latin loans: stress the last full syllable; g before e/i/y is [dʒ].
_FRENCH_STEMS = (
    "aprill", "licour", "vertu", "engendr", "inspir", "melod", "honou", "honest", "verray",
    "parfit", "gentil", "perc", "cours", "flour", "certeyn", "corag", "natur", "mesur", "servys",
    "servic", "citee", "caus", "straung", "sergeant", "digne", "signe", "benign", "pilgrim",
    "resoun", "beaut", "habit", "humbl", "herit", "hour", "hoost", "joye", "point", "veyn",
    "curteis", "compaign", "chivalr", "licou",
)
_FRENCH_WORDS = frozenset({"age", "coy", "cas"})
_FRENCH_SUFFIXES = (
    "our", "ours", "oun", "ouns", "ioun", "age", "ages", "ure", "ures", "aunce", "ence", "ance",
    "itee", "tee", "ye", "yes", "able", "esse", "ous",
)
# Silent h of French loans.
_H_SILENT = ("honou", "hono", "hour", "hoost", "habit", "humbl", "herit", "honest")
# Latin or Greek names: Latin stress (index of the stressed vowel), no lengthening.
_LATIN = {"zephirus": 0}
# French loans Chaucer stresses on the first syllable in verse.
_VERSE_INITIAL = frozenset({"verray", "parfit", "gentil", "tendre", "servyse"})
# Silent medial e. Always in every and evere; the rest only in verse.
_SYNCOPE = {"every": "evry", "evere": "evre"}
_SYNCOPE_VERSE = {"foweles": "fowles", "palmeres": "palmers", "sovereyn": "sovreyn"}
# Unstressed native prefixes (and Latin in-), by word.
_PREFIXES = ("with", "for", "be", "bi", "to", "a", "in")
_PREFIXED = frozenset(
    "bigynne bigan biforn bifore bifel bisyde bihynde biside bitwene bitwixe bicause forsake "
    "forgete agoon ago aboute above agayn ayeyn awey togidre togider withouten withinne inspired"
    .split()
)
# gg that is [d.dʒ].
_GG_DZH = ("brigg", "egg", "segg", "rigg", "begg", "hegg")
# Weak words before which final -e is dropped as before a vowel.
_WEAK_H = frozenset("he his him hir hem hath hadde how here".split())

_VOW = frozenset("aeiouy")
_FRONT = frozenset("eiy")
_STOP_F = frozenset("pbtdkgf")
_FRONT_PH = frozenset({"ɪ", "iː", "ɛ", "ɛː", "eː", "ai", "ɔi"})
_ENDINGS = ("s", "es", "e", "en", "ed", "eth", "er")

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|[,.!?;:]")


class _Seg:
    __slots__ = ("kind", "ph", "g", "long", "prefix", "final_e", "ending_e", "closes")

    def __init__(self, kind: str, ph: str = "", g: str = "") -> None:
        self.kind = kind  # "V" or "C"
        self.ph = ph
        self.closes = False  # written double (ssh, ck): the vowel before stays short
        self.g = g  # vowel grapheme class
        self.long = False  # forced long (before French gn)
        self.prefix = False  # the y- / i- prefix vowel
        self.final_e = False
        self.ending_e = False


# --- public -------------------------------------------------------------------


def to_ipa(text: str) -> str:
    """Transcribe Middle English `text` into IPA."""
    return format_ipa(analyze(text))


def analyze(text: str) -> list[Item]:
    """Middle English `text` as words (syllables of phones) and pause marks."""
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFC", text.strip()))
    verse = sum(1 for t in tokens if t[0].isalpha()) >= 6
    items: list[Item] = []
    for idx, tok in enumerate(tokens):
        if not tok[0].isalpha():
            items.append(Pause("." if tok in ";:" else tok))
            continue
        prev = tokens[idx - 1] if idx and tokens[idx - 1][0].isalpha() else ""
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        word = _word(tok, prev, nxt, verse)
        if word.syllables:
            items.append(word)
    return items


# --- one word -----------------------------------------------------------------


def _clean(raw: str) -> str:
    text = raw.lower().replace("’", "").replace("'", "")
    if text.endswith("é"):
        text = text[:-1] + "ee"
    text = text.replace("þ", "th").replace("ð", "th")
    if text.startswith("ȝ"):
        text = "Y" + text[1:]
    text = text.replace("ȝ", "gh")
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)
    )
    return text


def _word(raw: str, prev: str, nxt: str, verse: bool) -> Word:
    low = _clean(raw)
    key = low.replace("-", "")
    if key == "ye":
        return _fixed(_YE_EYE if _clean(prev) in _EYE_BEFORE else _YE_YOU, key)
    if key in _FIXED:
        return _fixed(_FIXED[key], key)

    spelled = _SYNCOPE.get(key) or (_SYNCOPE_VERSE.get(key) if verse else None)
    if spelled:
        low = key = spelled
    french = _is_french(key)
    latin = key in _LATIN

    # Prefix split: hyphen, the participle y-, or a listed prefixed word.
    if "-" in low:
        prefix, root = low.split("-", 1)
        root = root.replace("-", "")
    elif len(key) >= 4 and key[0] == "y" and key[1] not in _VOW:
        prefix, root = "y", key[1:]
    elif key in _PREFIXED:
        prefix = next(p for p in _PREFIXES if key.startswith(p))
        root = key[len(prefix):]
    else:
        prefix, root = "", key

    segs: list[_Seg] = []
    if prefix in ("y", "i"):
        seg = _Seg("V", g="i")
        seg.prefix = True
        segs.append(seg)
    elif prefix:
        _scan(_letters(prefix), french, key, segs)
    root_start = len(segs)
    _scan(_letters(root), french, key, segs)

    nuclei = [i for i, s in enumerate(segs) if s.kind == "V"]
    if not nuclei:
        return Word(())
    _mark_endings(segs, nuclei)

    root_nuclei = [i for i in nuclei if i >= root_start] or nuclei
    if latin:
        main = nuclei[min(_LATIN[key], len(nuclei) - 1)]
    elif verse and key in _VERSE_INITIAL:
        main = root_nuclei[0]
    elif french:
        full = [i for i in root_nuclei if not (segs[i].final_e or segs[i].ending_e)]
        main = (full or root_nuclei)[-1]
    else:
        main = root_nuclei[0]

    for i in nuclei:
        segs[i].ph = _vowel(segs, i, main, key, french, latin)
    _consonants(segs, key)

    if _elides(nxt) and segs[-1].kind == "V" and segs[-1].final_e:
        segs.pop()
        if len(segs) >= 2 and segs[-1].kind == "C" and segs[-1].ph == segs[-2].ph:
            segs.pop()  # thanne he → [ˈðan]: no geminate is left at the word end

    return _syllabify(segs, main, root_start, stressed=key not in _CLITICS)


def _fixed(ipa: str, key: str) -> Word:
    word = parse_word(ipa)
    if key in _CLITICS:
        word = Word(tuple(Syllable(s.phones, False) for s in word.syllables))
    elif len(word.syllables) == 1:
        word = Word((Syllable(word.syllables[0].phones, True),))
    return word


def _is_french(key: str) -> bool:
    if key in _FRENCH_WORDS or key.startswith(_FRENCH_STEMS):
        return True
    groups = len(re.findall(r"[aeiouy]+", key))
    return groups >= 2 and key.endswith(_FRENCH_SUFFIXES)


def _lex(key: str, table: frozenset[str]) -> bool:
    cands = {key}
    for suf in _ENDINGS:
        if key.endswith(suf) and len(key) > len(suf) + 1:
            stem = key[: -len(suf)]
            cands |= {stem, stem + "e"}
    return bool(cands & table)


def _ou_open(key: str) -> bool:
    return key.startswith(_OU_OPEN_STEMS) or _lex(key, _OU_OPEN)


def _elides(nxt: str) -> bool:
    """Final -e drops before a vowel, a silent h, or a weak h-word on the same line."""
    if not nxt or not nxt[0].isalpha():
        return False
    if nxt[0].isupper() and nxt != "I":
        return False  # a verse line (or a name) begins here
    low = _clean(nxt).replace("-", "")
    if low in _WEAK_H or low.startswith(_H_SILENT):
        return True
    first = low[0]
    if first == "y":
        return len(low) > 1 and low[1] not in _VOW
    if first == "v":
        return len(low) > 1 and low[1] not in _VOW
    return first in "aeiou"


# --- graphemes ----------------------------------------------------------------


def _letters(part: str) -> str:
    """Resolve u/v and consonantal y: v = [v], u = vowel, Y = [j]."""
    out = list(part)
    n = len(part)
    for i, ch in enumerate(part):
        prev = part[i - 1] if i else ""
        nxt = part[i + 1] if i + 1 < n else ""
        if ch == "v":
            if i == 0 and nxt and nxt not in _VOW:
                out[i] = "u"
        elif ch == "u":
            if nxt and nxt in _VOW and prev != "q" and (i == 0 or (prev and prev in _VOW)):
                out[i] = "v"
        elif ch == "y" and i == 0 and nxt and nxt in _VOW:
            out[i] = "Y"
    return "".join(out)


def _scan(s: str, french: bool, key: str, segs: list[_Seg]) -> None:
    n = len(s)
    i = 0

    def at(k: int) -> str:
        return s[i + k] if 0 <= i + k < n else ""

    def cons(*phones: str) -> None:
        for ph in phones:
            segs.append(_Seg("C", ph))

    def vowel(g: str, width: int) -> None:
        nonlocal i
        segs.append(_Seg("V", g=g))
        i += width

    while i < n:
        c, d, e = at(0), at(1), at(2)
        if c in _VOW:
            pair = c + d
            if c == "e" and d == "a" and e == "u":
                vowel("eu", 3)
            elif pair in ("aa", "ee", "oo"):
                vowel(pair, 2)
            elif pair in ("ai", "ay", "ei", "ey"):
                vowel("ai", 2)
            elif pair in ("au", "aw"):
                vowel("au", 2)
            elif pair in ("ou", "ow"):
                vowel("ou", 2)
            elif pair in ("eu", "ew"):
                vowel("eu", 2)
            elif pair in ("oi", "oy"):
                vowel("oi", 2)
            elif pair == "ie" and i + 2 == n and any(x.kind == "V" for x in segs):
                vowel("ie", 2)
            else:
                vowel(c, 1)
            continue
        step = 1
        if c == "s" and d == "s" and e == "h":
            cons("ʃ"); step = 3
            segs[-1].closes = True  # fresshe [ˈfrɛ.ʃə]: one sound, but a short vowel
        elif c == "s" and d == "c" and e == "h":
            cons("ʃ"); step = 3
        elif c == "s" and d == "h":
            cons("ʃ"); step = 2
        elif c == "t" and d == "c" and e == "h":
            cons("tʃ"); step = 3
        elif c == "c" and d == "h":
            cons("tʃ"); step = 2
        elif c == "t" and d == "h":
            cons("th"); step = 2
        elif c == "g" and d == "h":
            cons("g" if i == 0 else "gh"); step = 2
        elif c == "p" and d == "h":
            cons("f"); step = 2
        elif c == "w" and d == "h":
            cons("ʍ"); step = 2
        elif c == "c" and d == "k":
            cons("k"); step = 2
            segs[-1].closes = True
        elif c == "q":
            cons("kw"); step = 2 if d == "u" else 1
        elif c == "d" and d == "g":
            cons("dʒ"); step = 2
        elif c == "g" and d == "n" and i > 0:
            cons("n"); step = 2
            for seg in reversed(segs):
                if seg.kind == "V":
                    seg.long = True
                    break
        elif c == "g" and d == "g":
            if key.startswith(_GG_DZH):
                cons("d", "dʒ")
            else:
                cons("g", "g")
            step = 2
        elif c == "c" and d == "c":
            cons("k", "s" if e in _FRONT and e else "k"); step = 2
        elif c == "c":
            cons("s" if d and d in _FRONT else "k")
        elif c == "g":
            if french and d and d in _FRONT:
                cons("dʒ")
                if d == "e" and e in ("a", "o") and e:
                    step = 2  # sergeant: the e only softens the g
            else:
                cons("g")
        elif c == "j":
            cons("dʒ")
        elif c == "Y":
            cons("j")
        elif c == "x":
            cons("k", "s")
        elif c == "h":
            if i == 0 and not key.startswith(_H_SILENT):
                cons("h")
        elif c == "s":
            if d == "s":
                cons("s", "s"); step = 2
            else:
                prev = s[i - 1] if i else ""
                voiced = prev and prev in _VOW and d and d in _VOW
                cons("z" if voiced else "s")
        elif c == d and c in "bdfklmnprtvz":
            cons(c, c); step = 2
        elif c in "bdfklmnprtvwz":
            cons(c)
        i += step


def _mark_endings(segs: list[_Seg], nuclei: list[int]) -> None:
    if len(nuclei) < 2:
        return
    last = segs[nuclei[-1]]
    if last.g != "e":
        return
    tail = segs[nuclei[-1] + 1:]
    if not tail:
        last.final_e = True
    elif len(tail) == 1 and tail[0].ph in ("s", "z", "d", "n", "th", "r"):
        last.ending_e = True


# --- vowels and consonants in context ----------------------------------------


def _vowel(segs: list[_Seg], i: int, main: int, key: str, french: bool, latin: bool) -> str:
    seg = segs[i]
    g = seg.g
    foll: list[str] = []
    j = i + 1
    while j < len(segs) and segs[j].kind == "C":
        foll.append(segs[j].ph)
        j += 1
    next_v = j < len(segs)
    open_ = not foll or (len(foll) == 1 and next_v and not segs[i + 1].closes)
    before_gh = bool(foll) and foll[0] == "gh"
    cluster = tuple(foll[:2])

    if g == "ai":
        return "ai"
    if g == "au":
        return "au"
    if g == "oi":
        return "ɔi"
    if g == "eu":
        return "ɛu" if _lex(key, _EU_OPEN) else "iu"
    if g == "ou":
        if french:
            return "uː"
        return "ɔu" if before_gh or _ou_open(key) else "uː"
    if g == "aa":
        return "aː"
    if g == "ee":
        return "ɛː" if _lex(key, _OPEN_E) else "eː"
    if g == "oo":
        return "ɔː" if _lex(key, _OPEN_O) else "oː"
    if g == "ie" or seg.prefix:
        return "ɪ"
    if latin:
        return {"a": "a", "e": "ɛ", "i": "ɪ", "y": "ɪ", "o": "ɔ", "u": "ʊ"}[g]

    if i != main:
        if g == "e":
            return "ə" if i > main else "ɛ"
        if g in "iy":
            return "ɪ"
        if g == "a":
            return "a"
        if g == "o":
            return "ʊ" if key in _O_UU else "ɔ"
        return "iu" if french and open_ else "ʊ"

    long_ok = (open_ or seg.long) and not before_gh
    if g == "a":
        return "aː" if long_ok or key in _LONG_A else "a"
    if g == "e":
        if long_ok:
            return "ɛː" if _lex(key, _OPEN_E) else "eː"
        return "ɛ"
    if g in "iy":
        if not before_gh and (long_ok or cluster in (("l", "d"), ("n", "d"), ("m", "b")) or _lex(key, _LONG_I)):
            return "iː"
        return "ɪ"
    if g == "o":
        near = {segs[i - 1].ph if i else "", segs[i + 1].ph if i + 1 < len(segs) else ""}
        if key in _O_UU or (near & {"m", "n", "w", "v"} and key not in _NOT_CUP):
            return "ʊ"
        if cluster == ("l", "d"):
            return "ɔː"
        if long_ok:
            return "ɔː" if _lex(key, _OPEN_O) else "oː"
        return "ɔ"
    return "iu" if french and open_ else "ʊ"


def _consonants(segs: list[_Seg], key: str) -> None:
    for k, seg in enumerate(segs):
        if seg.kind != "C":
            continue
        prev = segs[k - 1] if k else None
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if seg.ph == "th":
            if k == 0:
                seg.ph = "ð" if key in _TH_VOICED else "θ"
            elif (
                prev is not None and nxt is not None and nxt.kind == "V"
                and (prev.kind == "V" or prev.ph in ("r", "l", "m", "n"))
            ):
                seg.ph = "ð"
            else:
                seg.ph = "θ"
        elif seg.ph == "gh":
            vowel = next((s.ph for s in reversed(segs[:k]) if s.kind == "V"), "")
            seg.ph = "ç" if vowel in _FRONT_PH else "x"
        elif seg.ph == "n" and nxt is not None and nxt.ph in ("k", "g", "kw"):
            seg.ph = "ŋ"


# --- syllables ----------------------------------------------------------------


def _syllabify(segs: list[_Seg], main: int, root_start: int, stressed: bool) -> Word:
    nuclei = [i for i, s in enumerate(segs) if s.kind == "V"]
    cuts: list[int] = []
    for a, b in zip(nuclei, nuclei[1:]):
        gap = list(range(a + 1, b))
        if root_start and a < root_start <= b:
            cut = root_start
        elif not gap:
            cut = b
        elif len(gap) == 1:
            cut = gap[0]
        elif segs[gap[-2]].ph in _STOP_F and segs[gap[-1]].ph in ("l", "r"):
            cut = gap[-2]
        else:
            cut = gap[-1]
        cuts.append(cut)
    bounds = [0] + cuts + [len(segs)]
    main_syl = nuclei.index(main) if main in nuclei else 0
    syllables: list[Syllable] = []
    for si in range(len(bounds) - 1):
        phones: list[str] = []
        for seg in segs[bounds[si]:bounds[si + 1]]:
            phones.extend(("k", "w") if seg.ph == "kw" else (seg.ph,))
        syllables.append(Syllable(tuple(phones), stressed and si == main_syl))
    return Word(tuple(syllables))
