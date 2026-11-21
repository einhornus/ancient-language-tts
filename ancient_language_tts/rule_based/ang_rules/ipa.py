"""Old English (Late West Saxon, c. 1000) text → IPA, by rule.

Implements `data/ang/phonology.txt`. `analyse` does the work: it reads the
spelling (macrons, dots, prefixes, compounds), decides every sound, splits
syllables and places the stress. It returns words as syllables of phones,
which `to_ipa` formats and the direct respellers map straight to a modern
spelling without going through an IPA string.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- symbols

NONSYL = "̯"  # ̯
VOICELESS = "̥"  # ̥
EA, EA_LONG = "æɑ" + NONSYL, "æːɑ" + NONSYL
EO, EO_LONG = "eo" + NONSYL, "eːo" + NONSYL

VOWELS = frozenset("aæeiouy")
_MONO = {"a": "ɑ", "æ": "æ", "e": "e", "i": "i", "o": "o", "u": "u", "y": "y"}
_PLAIN = {"p": "p", "b": "b", "t": "t", "d": "d", "m": "m", "n": "n", "l": "l",
          "r": "r", "w": "w", "k": "k", "j": "j", "q": "k"}
_FRIC = {"f": "f", "s": "s", "þ": "θ"}
_VOICE = {"f": "v", "s": "z", "θ": "ð"}
_VOICED_C = frozenset({"m", "n", "ŋ", "l", "r", "w", "j", "b", "d", "g", "ɣ",
                       "v", "z", "ð", "dʒ"})
_STOPS_F = frozenset({"p", "b", "t", "d", "k", "g", "f"})

_MACRON, _ACUTE, _DOT = "̄", "́", "̇"
_HYPHENS = "-‐‑"
_BASE_MAP = {"ð": "þ", "ƿ": "w", "ȝ": "g", "v": "f", "z": "s"}

# ---------------------------------------------------------------- word lists

# Unstressed monosyllables: articles/demonstratives, prepositions,
# conjunctions, the particle þe, the negative ne.
_FUNCTION = frozenset({
    "se", "sēo", "þæt", "þā", "þām", "þǣm", "þæs", "þȳ", "þon",
    "on", "in", "tō", "of", "mid", "æt", "be", "for", "fram", "from", "wiþ", "ymb",
    "and", "ond", "ac", "gif", "swā", "þe", "ne",
})
# Possessives: unstressed before their noun, stressed after it.
_POSSESSIVE = frozenset({"mīn", "þīn", "his", "sīn", "ūre", "ēower", "hiera",
                         "hire", "hira", "heora", "ūser", "uncer", "incer"})

# Restoration for texts that print no macrons (and/or no dots).
_LEXICON = (
    "wē hē þū hū nū swā tō ūre þīn mīn sīn sī mē hwā hwǣr þǣr ēac ēċe rīċe "
    "sēċan tǣċan þenċan wyrċan benċ ċiriċe cyċene hūs stān bōc dēop lēoht ēage "
    "ēast þēod hlāf hrīm rīdan rīsan ārīsan hāliġ ġē ġēar cēne gēs wīf līf ōr "
    "ēower þā þǣre sēo ūs ān dōm bēag ġenōg cnēo dēad dēaþ frēond fēond sōþ "
    "tīd hwīl nēah hēah lēof ūt drīfan wrītan scōl"
).split()

# First elements of compounds (stress stays on them).
_FIRST_ELEMENTS = (
    "ġēar þēod heofon mōd wuldor ēad gār middan eorþ folc gūþ hilde hild wīġ "
    "beadu ealdor ælf æl frēo hēah god gold mann man land sige wīn wine hūs "
    "sunnan wæl woruld worold ēþel cyning cyne wæter eorl"
).split()

# Prefixes: (bare form, class, index of a vowel that must be long or -1).
# class: "u" never stressed, "s" always stressed, "v" by part of speech.
_PREFIXES = (
    ("under", "v", -1), ("ofer", "v", -1), ("and", "s", -1), ("ond", "s", -1),
    ("for", "u", -1), ("wiþ", "v", -1), ("ymb", "v", -1), ("æt", "v", -1),
    ("to", "v", 1), ("ut", "v", 0), ("ge", "u", -1), ("be", "u", -1),
    ("un", "s", -1), ("ed", "s", -1), ("of", "v", -1), ("on", "v", -1),
    ("a", "v", 0),
)
_PREFIX_FORMS = frozenset(p[0] for p in _PREFIXES)
_NOT_PREFIX = {
    "for": ("form", "forst", "ford", "forht", "forþ", "fore", "foran", "forn"),
    "be": ("bet",),
    "a": ("ag", "asc", "aht", "an"),
    "of": ("off",),
    "un": ("unn",),
}
_VERBAL = ("ian", "an", "ode", "ede", "ade", "de", "te", "on", "aþ", "eþ", "iþ",
           "ende", "enne", "anne", "st")
# o before a nasal stays o in these.
_NO_O_TO_A = frozenset({"þonne", "þone", "hwonne", "ponne"})
_SK_WORDS = ("discipul",)  # Latin loans keep [sk]
_CG_AS_GG = ("docga", "docgan")

_WORD_RE = re.compile(
    r"(?P<w>(?:[^\W\d_]|[̀-ͯ])+(?:[-‐‑'’](?:[^\W\d_]|[̀-ͯ])+)*)"
    r"|(?P<a>[⁊&])|(?P<p>[,.!?;:])"
)


# ---------------------------------------------------------------- data types

@dataclass
class _L:
    base: str
    long: bool = False
    dotted: bool = False
    from_ie: bool = False


@dataclass
class _U:
    kind: str  # "V" | "C"
    base: str
    i: int
    elem: int
    long: bool = False
    dotted: bool = False
    ipa: str = ""
    pal_before: bool = False  # front from the start: palatalizes c/g before it
    front_before: bool = False
    front_after: bool = False  # front monophthong
    diacritic_e: bool = False  # ġeo- / sceo-: the e only marks the consonant


@dataclass
class _S:
    ipa: str
    kind: str  # "v" | "c"
    elem: int
    cand: bool = False  # f/s/þ that may voice
    diacritic_e: bool = False


Word = tuple[list[list[str]], "int | None"]


# ---------------------------------------------------------------- public API

def to_ipa(text: str) -> str:
    """Transcribe Old English `text` into IPA."""
    out: list[str] = []
    for item in analyse(text):
        if item[0] == "p":
            if out and out[-1] == " ":
                out.pop()
            out += [item[1], " "]
        else:
            out += [format_word(item[1], item[2]), " "]
    return "".join(out).strip()


def format_word(sylls: list[list[str]], stress: int | None) -> str:
    parts = ["".join(s) for s in sylls]
    if not parts:
        return ""
    out = ("ˈ" if stress == 0 else "") + parts[0]
    for i, body in enumerate(parts[1:], start=1):
        out += ("ˈ" if i == stress else ".") + body
    return out


def analyse(text: str) -> list[tuple]:
    """Words as ("w", syllables, stress) and pauses as ("p", mark)."""
    text = unicodedata.normalize("NFC", text.strip())
    nfd = unicodedata.normalize("NFD", text)
    has_length = _MACRON in nfd or _ACUTE in nfd
    has_dots = _DOT in nfd
    tokens: list[tuple[str, str]] = []
    for m in _WORD_RE.finditer(text):
        if m.group("w"):
            tokens.append(("w", m.group("w")))
        elif m.group("a"):
            tokens.append(("w", "and"))
        else:
            mark = m.group("p")
            tokens.append(("p", "." if mark in ";:" else mark))
    items: list[tuple] = []
    for k, (kind, value) in enumerate(tokens):
        if kind == "p":
            items.append(("p", value))
            continue
        next_is_word = k + 1 < len(tokens) and tokens[k + 1][0] == "w"
        word = _word(value, has_length, has_dots, next_is_word)
        if word is not None:
            items.append(("w", word[0], word[1]))
    return items


# ---------------------------------------------------------------- letters

def _letters(raw: str) -> tuple[list[_L], set[int]]:
    s = unicodedata.normalize("NFD", raw.lower())
    letters: list[_L] = []
    bounds: set[int] = set()
    for ch in s:
        if ch in (_MACRON, _ACUTE):
            if letters:
                letters[-1].long = True
            continue
        if ch == _DOT:
            if letters:
                letters[-1].dotted = True
            continue
        if unicodedata.combining(ch):
            continue
        if ch in _HYPHENS:
            if letters:
                bounds.add(len(letters))
            continue
        if ch in "'’":
            continue
        letters.append(_L(_BASE_MAP.get(ch, ch)))
    # Older West Saxon ie/īe → y/ȳ, io/īo → eo/ēo.
    i = 0
    while i + 1 < len(letters):
        a, b = letters[i], letters[i + 1]
        after_vowel = i > 0 and letters[i - 1].base in VOWELS
        if a.base == "i" and b.base == "e" and not after_vowel and i + 1 not in bounds:
            nxt = letters[i + 2] if i + 2 < len(letters) else None
            if nxt is not None and nxt.base == "x":
                letters[i] = _L("i", a.long or b.long)  # siex → six
            else:
                letters[i] = _L("y", a.long or b.long, from_ie=True)
            del letters[i + 1]
            bounds = {x - 1 if x > i + 1 else x for x in bounds}
        elif a.base == "i" and b.base == "o" and not after_vowel and i + 1 not in bounds:
            a.base = "e"
        i += 1
    return letters, bounds


def _bare(letters: list[_L]) -> str:
    return "".join(l.base for l in letters)


def _keyed(letters: list[_L]) -> str:
    return unicodedata.normalize(
        "NFC", "".join(l.base + (_MACRON if l.long else "") for l in letters))


def _entry(form: str) -> list[_L]:
    return _letters(form)[0]


_LEX_BARE = {_bare(_entry(w)): w for w in _LEXICON}
_LEX_KEYED = {_keyed(_entry(w)): w for w in _LEXICON}
_FIRST = sorted((_entry(w) for w in _FIRST_ELEMENTS), key=len, reverse=True)
_FUNCTION_BARE = frozenset(_bare(_entry(w)) for w in _FUNCTION)
_POSSESSIVE_BARE = frozenset(_bare(_entry(w)) for w in _POSSESSIVE)


def _restore(letters: list[_L], bounds: set[int], has_length: bool, has_dots: bool) -> list[_L]:
    if bounds or any(l.long or l.dotted for l in letters):
        if has_length and not has_dots and not any(l.dotted for l in letters):
            form = _LEX_KEYED.get(_keyed(letters))
            if form:
                for l, e in zip(letters, _entry(form)):
                    l.dotted = e.dotted
        return letters
    if not has_length:
        form = _LEX_BARE.get(_bare(letters))
        if form:
            entry = _entry(form)
            if has_dots:
                for l in entry:
                    l.dotted = False
            return entry
    return letters


# ---------------------------------------------------------------- morphology

def _nuclei(s: str) -> int:
    count, prev = 0, False
    for i, ch in enumerate(s):
        v = ch in VOWELS
        if v and not (prev and s[i - 1] == "e" and ch in "ao"):
            count += 1
        prev = v
    return count


def _prefix_at(letters: list[_L], bare: str, pos: int, end: int, has_length: bool,
               allowed: tuple[str, ...] | None) -> tuple[str, str] | None:
    whole = bare[pos:end]
    if whole in _PREFIX_FORMS:
        return None
    for form, cls, long_at in _PREFIXES:
        if allowed is not None and form not in allowed:
            continue
        if not whole.startswith(form):
            continue
        rest = whole[len(form):]
        if len(rest) < 2 or rest[0] in VOWELS or not any(c in VOWELS for c in rest):
            continue
        if any(whole.startswith(x) for x in _NOT_PREFIX.get(form, ())):
            continue
        if form == "be" and _nuclei(rest) < 2:
            continue
        vowels = [k for k, ch in enumerate(form) if ch in VOWELS]
        if has_length:
            if long_at >= 0 and not letters[pos + long_at].long:
                continue
            if long_at < 0 and any(letters[pos + k].long for k in vowels):
                continue
        elif form == "a":
            continue
        return form, cls
    return None


def _first_element(letters: list[_L], bare: str, pos: int, end: int, has_length: bool) -> int:
    """Length of a known first compound element starting at `pos`, or 0."""
    for entry in _FIRST:
        n = len(entry)
        if bare[pos:pos + n] != _bare(entry):
            continue
        rest = bare[pos + n:end]
        if len(rest) < 3 or rest[0] in VOWELS or not any(c in VOWELS for c in rest):
            continue
        if has_length and any(l.long != e.long for l, e in zip(letters[pos:pos + n], entry)):
            continue
        for l, e in zip(letters[pos:pos + n], entry):
            if not has_length:
                l.long = e.long
            if e.dotted and not l.dotted and l.base == "g":
                l.dotted = True
        return n
    return 0


def _morph(letters: list[_L], bounds: set[int], has_length: bool) -> list[tuple[int, str]]:
    """Element starts and kinds: pu/ps (prefix), root, comp, cu (ġe- inside)."""
    bare = _bare(letters)
    cuts = sorted(b for b in bounds if 0 < b < len(letters))
    parts = list(zip([0] + cuts, cuts + [len(letters)]))
    elems: list[tuple[int, str]] = []
    for pi, (a, b) in enumerate(parts):
        pos = a
        if pi == 0:
            pres: list[tuple[int, str, str]] = []
            for step in range(2):
                found = _prefix_at(letters, bare, pos, b, has_length,
                                   None if step == 0 else ("ge", "be"))
                if not found:
                    break
                pres.append((pos, found[0], found[1]))
                pos += len(found[0])
            if pres:
                verbal = len(pres) > 1 or bare[pos:b].endswith(_VERBAL)
                for p, form, cls in pres:
                    if form == "ge":
                        letters[p].dotted = True
                    if form == "ond":
                        letters[p].base = "a"
                    if cls == "v":
                        cls = "u" if verbal else "s"
                    elems.append((p, "p" + cls))
        kind = "root" if pi == 0 else "comp"
        while True:
            elems.append((pos, kind))
            n = _first_element(letters, bare, pos, b, has_length)
            if not n:
                break
            pos += n
            kind = "comp"
            if _prefix_at(letters, bare, pos, b, has_length, ("ge",)):
                letters[pos].dotted = True
                elems.append((pos, "cu"))
                pos += 2
    return elems


# ---------------------------------------------------------------- sounds

def _palatal_marker(letters: list[_L], i: int, starts: set[int], no_dots: bool) -> bool:
    """True when e at `i` follows ġ or initial sc and only marks the consonant."""
    if i == 0 or (i in starts):
        return False
    prev = letters[i - 1]
    if prev.base == "g":
        return prev.dotted or (no_dots and (i - 1) in starts)
    if prev.base == "c" and i >= 2 and letters[i - 2].base == "s":
        return (i - 2) in starts or prev.dotted
    return False


def _units(letters: list[_L], elem_of: list[int], starts: set[int], no_dots: bool) -> list[_U]:
    units: list[_U] = []
    i, n = 0, len(letters)
    while i < n:
        l = letters[i]
        if l.base in VOWELS:
            nxt = letters[i + 1] if i + 1 < n else None
            if (l.base == "e" and nxt is not None and nxt.base in "ao"
                    and elem_of[i + 1] == elem_of[i]):
                long = l.long or nxt.long
                if nxt.base == "o" and not long and _palatal_marker(letters, i, starts, no_dots):
                    units.append(_U("V", "o", i, elem_of[i], ipa="o", diacritic_e=True))
                else:
                    ipa = (EA_LONG if long else EA) if nxt.base == "a" else (EO_LONG if long else EO)
                    units.append(_U("V", "e" + nxt.base, i, elem_of[i], long=long, ipa=ipa,
                                    pal_before=True, front_before=True))
                i += 2
                continue
            front = l.base in "æeiy"
            units.append(_U("V", l.base, i, elem_of[i], long=l.long,
                            ipa=_MONO[l.base] + ("ː" if l.long else ""),
                            pal_before=l.base == "i" or l.from_ie,
                            front_before=front, front_after=front))
            i += 1
            continue
        units.append(_U("C", l.base, i, elem_of[i], dotted=l.dotted))
        i += 1
    return units


def _segments(units: list[_U], starts: set[int], no_dots: bool, bare: str) -> list[_S]:
    segs: list[_S] = []
    n = len(units)

    def at(k: int) -> _U | None:
        return units[k] if 0 <= k < n else None

    def is_v(k: int) -> bool:
        u = at(k)
        return u is not None and u.kind == "V"

    j = 0
    while j < n:
        u = units[j]
        if u.kind == "V":
            segs.append(_S(u.ipa, "v", u.elem, diacritic_e=u.diacritic_e))
            j += 1
            continue
        b = u.base
        prev, nxt = at(j - 1), at(j + 1)
        start = u.i in starts
        pair = nxt is not None and nxt.kind == "C" and nxt.elem == u.elem
        nb = nxt.base if pair else ""
        last = j == n - 1
        between = is_v(j - 1) and is_v(j + 2)

        def add(*ipas: str, cand: bool = False) -> None:
            for p in ipas:
                segs.append(_S(p, "c", u.elem, cand))

        if b == "s" and nb == "c":
            fol = at(j + 2)
            if bare.startswith(_SK_WORDS):
                sh = False
            elif start or nxt.dotted:
                sh = True
            else:
                front_prev = prev is not None and prev.kind == "V" and prev.front_after
                sh = front_prev and (j + 1 == n - 1 or (fol is not None and fol.kind == "V"
                                                         and fol.front_before))
            if sh:
                add(*(("ʃ", "ʃ") if between else ("ʃ",)))
            else:
                add("s", "k")
            j += 2
            continue
        if b == "c" and nb == "g":
            if bare.startswith(_CG_AS_GG):
                add(*(("g", "g") if between else ("g",)))
            else:
                add(*(("d", "dʒ") if between else ("dʒ",)))
            j += 2
            continue
        if b == "c" and nb == "c":
            if u.dotted or nxt.dotted:
                add(*(("t", "tʃ") if between else ("tʃ",)))
            else:
                add(*(("k", "k") if between else ("k",)))
            j += 2
            continue
        if b == "c":
            if u.dotted:
                add("tʃ")
            elif last and prev is not None and prev.kind == "V" and prev.base == "i":
                add("tʃ")
            elif no_dots and start and nxt is not None and nxt.kind == "V" and nxt.pal_before:
                add("tʃ")
            elif (no_dots and not start and prev is not None and prev.kind == "V"
                  and prev.front_after and nxt is not None and nxt.kind == "V"
                  and nxt.base in ("e", "i") and not nxt.long):
                add("tʃ")
            else:
                add("k")
            j += 1
            continue
        if b == "g" and nb == "g":
            add(*(("g", "g") if between else ("g",)))
            j += 2
            continue
        if b == "g":
            after_n = (prev is not None and prev.kind == "C" and prev.base == "n"
                       and prev.elem == u.elem)
            if u.dotted:
                add("dʒ" if after_n else "j")
            elif after_n:
                add("g")
            elif start:
                pal = no_dots and nxt is not None and nxt.kind == "V" and (
                    nxt.pal_before or nxt.diacritic_e)
                add("j" if pal else "g")
            elif prev is not None and prev.kind == "V":
                if prev.front_after:
                    if last or nxt.kind == "C" or nxt.front_before:
                        add("j")
                    else:
                        add("ɣ")
                else:
                    add("x" if last else "ɣ")
            elif prev is not None and prev.base in ("l", "r"):
                add("x" if last else "ɣ")
            else:
                add("g")
            j += 1
            continue
        if b == "h":
            if start or prev is None:
                cluster = {"w": "ʍ", "l": "l" + VOICELESS, "r": "r" + VOICELESS,
                           "n": "n" + VOICELESS}.get(nb)
                if cluster:
                    add(cluster)
                    j += 2
                else:
                    add("h")
                    j += 1
                continue
            front = prev.kind == "V" and prev.front_after
            sound = "ç" if front else "x"
            if nb == "h":
                add(*((sound, sound) if between else (sound,)))
                j += 2
            else:
                add(sound)
                j += 1
            continue
        if b == "x":
            add("k", "s")
            j += 1
            continue
        if b in _FRIC:
            if nb == b:
                add(*((_FRIC[b], _FRIC[b]) if between else (_FRIC[b],)))
                j += 2
            else:
                add(_FRIC[b], cand=not start)
                j += 1
            continue
        if b in _PLAIN:
            if nb == b:
                add(*((_PLAIN[b], _PLAIN[b]) if between else (_PLAIN[b],)))
                j += 2
            else:
                add(_PLAIN[b])
                j += 1
            continue
        j += 1
    for k in range(len(segs) - 1):
        if segs[k].ipa == "n" and segs[k + 1].ipa in ("k", "g"):
            segs[k].ipa = "ŋ"
    return segs


def _syllabify(segs: list[_S]) -> list[list[int]]:
    nuclei = [k for k, s in enumerate(segs) if s.kind == "v"]
    if not nuclei:
        return [list(range(len(segs)))] if segs else []
    cuts: list[int] = []
    for a, b in zip(nuclei, nuclei[1:]):
        cut = None
        for k in range(a + 1, b + 1):
            if segs[k].elem != segs[k - 1].elem:
                cut = k
                break
        if cut is None:
            m = b - a - 1
            if m <= 1:
                cut = a + 1
            elif segs[b - 2].ipa in _STOPS_F and segs[b - 1].ipa in ("l", "r"):
                cut = b - 2
            else:
                cut = b - 1
        cuts.append(cut)
    edges = [0] + cuts + [len(segs)]
    return [list(range(edges[i], edges[i + 1])) for i in range(len(edges) - 1)]


def _word(raw: str, has_length: bool, has_dots: bool, next_is_word: bool) -> Word | None:
    letters, bounds = _letters(raw)
    if not letters:
        return None
    letters = _restore(letters, bounds, has_length, has_dots)
    if _bare(letters) == "ond":
        letters[0].base = "a"
    elems = _morph(letters, bounds, has_length)
    starts = {s for s, _ in elems}
    elem_of: list[int] = []
    e = 0
    for i in range(len(letters)):
        while e + 1 < len(elems) and elems[e + 1][0] <= i:
            e += 1
        elem_of.append(e)
    no_dots = not has_dots
    bare = _bare(letters)
    units = _units(letters, elem_of, starts, no_dots)
    segs = _segments(units, starts, no_dots, bare)
    if not segs:
        return None
    sylls = _syllabify(segs)
    syl_of = {k: si for si, ks in enumerate(sylls) for k in ks}

    # Function word?
    key = _keyed(letters) if has_length else bare
    func = _FUNCTION if has_length else _FUNCTION_BARE
    poss = _POSSESSIVE if has_length else _POSSESSIVE_BARE
    unstressed = len(sylls) == 1 and (key in func or (key in poss and next_is_word))

    # o before a nasal in a stressed root → ɑ.
    if not unstressed and bare not in _NO_O_TO_A:
        for ei, (_, kind) in enumerate(elems):
            if kind not in ("root", "comp"):
                continue
            first = next((k for k, s in enumerate(segs) if s.elem == ei and s.kind == "v"), None)
            if (first is not None and segs[first].ipa == "o" and not segs[first].diacritic_e
                    and first + 1 < len(segs) and segs[first + 1].ipa in ("m", "n", "ŋ")):
                segs[first].ipa = "ɑ"

    # Stress.
    stress: int | None = None
    stressish: set[int] = set()
    if not unstressed:
        for ei, (_, kind) in enumerate(elems):
            first = next((k for k, s in enumerate(segs) if s.elem == ei and s.kind == "v"), None)
            if first is None:
                continue
            if kind in ("ps", "root", "comp"):
                if stress is None:
                    stress = syl_of[first]
                stressish.add(syl_of[first])
        if stress is None:
            stress = 0
            stressish.add(0)

    # Voicing of f, s, þ.
    for k, s in enumerate(segs):
        if not s.cand or k == 0 or k == len(segs) - 1:
            continue
        before, after = segs[k - 1], segs[k + 1]
        voiced = lambda x: x.kind == "v" or x.ipa in _VOICED_C  # noqa: E731
        if voiced(before) and voiced(after) and syl_of[k - 1] in stressish:
            s.ipa = _VOICE[s.ipa]

    return [[segs[k].ipa for k in ks] for ks in sylls], stress


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "ang" / "examples.txt"
    rows = [ln.split("→", 1) for ln in path.read_text(encoding="utf-8").splitlines() if "→" in ln]
    bad = 0
    for src, want in rows:
        got = to_ipa(src.strip())
        if got != want.strip():
            bad += 1
            print("src:", src.strip(), "\nwant:", want.strip(), "\n got:", got, "\n")
    print(f"{len(rows) - bad}/{len(rows)} matched")
