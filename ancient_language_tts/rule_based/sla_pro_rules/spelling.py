"""Proto-Slavic spelling → letters, syllables and stress.

Shared by `ipa` (letters → IPA) and the direct respellers (letters → a modern
spelling). It reads the notations listed in `data/sla-pro/phonology.txt`
("Spelling of the input") and reduces each word to the Wiktionary letters:

    vowels      i y u ě a ę ǫ e o ь ъ
    syllabic    r̥ l̥
    consonants  p b t d k g s z x š ž č c dz dž m n l r j v f
    palatals    ľ ň ř ť ď ś

Accent marks are read for stress only; tone and the macron are dropped.
Nothing here produces IPA.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

VOWELS = frozenset({"i", "y", "u", "ě", "a", "ę", "ǫ", "e", "o", "ь", "ъ"})
SYLLABIC = frozenset({"r̥", "l̥"})
NUCLEI = VOWELS | SYLLABIC
PALATALS = frozenset({"ľ", "ň", "ř", "ť", "ď", "ś"})
JERS = frozenset({"ь", "ъ"})

# Separate unstressed words (prepositions, conjunctions, particles, enclitics).
CLITICS = frozenset(
    "vъ sъ kъ na za po do otъ u iz bez i a ni ne že li bo mi ti si sę jь".split()
)

PAUSES = frozenset(",!?.")

# Combining marks (NFD).
_GRAVE, _ACUTE, _CIRCUMFLEX, _TILDE = "̀", "́", "̂", "̃"
_MACRON, _BREVE, _CARON, _DOUBLE_ACUTE = "̄", "̆", "̌", "̋"
_DOUBLE_GRAVE, _INV_BREVE = "̏", "̑"
_DOT_BELOW, _RING_BELOW, _OGONEK, _LINE_BELOW = "̣", "̥", "̨", "̩"
_ACCENTS = frozenset({_GRAVE, _ACUTE, _CIRCUMFLEX, _TILDE, _DOUBLE_ACUTE, _DOUBLE_GRAVE, _INV_BREVE})
_SYLLABIC_MARKS = frozenset({_RING_BELOW, _DOT_BELOW, _LINE_BELOW})
_APOSTROPHES = frozenset("'’ʼ")
_WORD_EXTRA = frozenset("*-") | _APOSTROPHES

_PLAIN = frozenset("pbtdkgszxmnlrjvfaeiouy") | JERS


@dataclass
class Unit:
    """One letter of the word, in Wiktionary notation."""

    letter: str
    marked: bool = False  # carries an accent mark

    @property
    def nucleus(self) -> bool:
        return self.letter in NUCLEI

    @property
    def consonant(self) -> bool:
        return self.letter not in NUCLEI


@dataclass
class Word:
    units: list[Unit]
    syllables: list[list[int]]  # unit indices per syllable
    stress: int | None  # stressed syllable index

    @property
    def key(self) -> str:
        return "".join(u.letter for u in self.units)

    def stressed_unit(self) -> int | None:
        """Index of the nucleus of the stressed syllable."""
        if self.stress is None:
            return None
        for i in self.syllables[self.stress]:
            if self.units[i].nucleus:
                return i
        return None


# --- tokenizing ------------------------------------------------------------

def tokens(text: str) -> list[tuple[str, str]]:
    """Split a line into ("word", raw) and ("pause", mark) tokens.

    A semicolon or colon is a long pause and becomes a period. Parentheses,
    quotation marks and other symbols are dropped.
    """
    out: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(("word", "".join(buf)))
            buf.clear()

    for ch in unicodedata.normalize("NFD", text):
        if ch.isalpha() or unicodedata.combining(ch) or ch in _WORD_EXTRA:
            buf.append(ch)
            continue
        flush()
        mark = "." if ch in ";:" else ch
        if mark in PAUSES:
            if out and out[-1] == ("pause", mark):
                continue
            out.append(("pause", mark))
    flush()
    return out


# --- one word ----------------------------------------------------------------

def analyze(raw: str) -> Word | None:
    """Letters, syllables and stress of one written word (None if no letters)."""
    units = _letters(raw)
    if not units:
        return None
    _mark_syllabic(units)
    syllables = _syllabify(units)
    return Word(units, syllables, _stress(units, syllables))


def _clusters(raw: str) -> list[tuple[str, set[str]]]:
    """Base letter + its combining marks. "-" survives as a boundary."""
    out: list[tuple[str, set[str]]] = []
    for ch in unicodedata.normalize("NFD", raw.lower()):
        if ch == "*":
            continue
        if unicodedata.combining(ch):
            if out and out[-1][0] != "-":
                out[-1][1].add(ch)
            continue
        if ch in _APOSTROPHES:
            if out and out[-1][0] in "lnrtd":
                out[-1][1].add("'")
            continue
        out.append((ch, set()))
    return out


def _letters(raw: str) -> list[Unit]:
    clusters = _clusters(raw)
    units: list[Unit | str] = []  # str "-" = morpheme boundary
    pending_acute_liquid: list[int] = []
    for base, marks in clusters:
        if base == "-":
            units.append("-")
            continue
        accent = bool(marks & _ACCENTS)
        caron = _CARON in marks
        letter: str | None = None
        if base == "e":
            letter = "ę" if _OGONEK in marks else "ě" if (caron or _MACRON in marks) else "e"
        elif base == "o":
            letter = "ǫ" if _OGONEK in marks else "o"
        elif base == "i":
            letter = "ь" if (caron or _BREVE in marks) else "i"
        elif base == "u":
            letter = "ъ" if (caron or _BREVE in marks) else "u"
        elif base in ("a", "y", "ь", "ъ"):
            letter = base
        elif base in ("c", "s", "z", "n", "t", "d", "l", "r", "ʒ"):
            letter, accent = _consonant(base, marks, accent)
            if letter in ("l", "r") and _ACUTE in marks:
                # ĺ ŕ: palatal, or a stressed liquid after a jer (decided below).
                pending_acute_liquid.append(len(units))
            if letter in ("l", "r") and marks & _SYLLABIC_MARKS:
                letter += "̥"
        elif base == "đ":
            letter = "ď"
        elif base == "h":
            letter = "x"
        elif base == "w":
            letter = "v"
        elif base in "pbkgxmjvf":
            letter = base
        if letter is None:
            continue  # unknown symbol: not a Proto-Slavic letter
        units.append(Unit(letter, accent))

    for idx in pending_acute_liquid:
        u = units[idx]
        assert isinstance(u, Unit)
        nxt = _neighbour(units, idx, 1)
        if len(u.letter) > 1 or (nxt is not None and nxt.consonant):
            # *vьĺkъ, *vĺ̥kъ: before a consonant the acute stresses the
            # syllable of the liquid (a jer + liquid, or a syllabic liquid).
            u.marked = True
        else:
            u.letter = {"l": "ľ", "r": "ř"}[u.letter]  # ĺ ŕ before a vowel

    return _digraphs(units)


def _consonant(base: str, marks: set[str], accent: bool) -> tuple[str, bool]:
    caron = _CARON in marks
    acute = _ACUTE in marks
    apos = "'" in marks
    if base == "c":
        return ("č" if caron else "ť" if acute else "c"), False
    if base == "s":
        return ("š" if caron else "ś" if acute else "s"), False
    if base == "z":
        return ("ž" if caron else "z"), False
    if base == "ʒ":
        return ("dž" if caron else "dz"), False
    if base == "n":
        return ("ň" if (caron or acute or apos) else "n"), False
    if base in "td":
        return ({"t": "ť", "d": "ď"}[base] if (caron or apos) else base), False
    # l, r
    if caron or apos:
        return {"l": "ľ", "r": "ř"}[base], False
    # An accent other than the acute on a liquid stresses its syllable.
    return base, accent and not acute


def _neighbour(units: list, idx: int, step: int) -> Unit | None:
    j = idx + step
    if 0 <= j < len(units) and isinstance(units[j], Unit):
        return units[j]
    return None


def _digraphs(units: list) -> list[Unit]:
    """ch → x, dz / dž, and older lj nj rj tj dj → palatal. A hyphen blocks."""
    out: list[Unit] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u == "-":
            i += 1
            continue
        nxt = units[i + 1] if i + 1 < len(units) else None
        if isinstance(nxt, Unit):
            pair = u.letter + nxt.letter
            if pair == "cx":  # "ch" (h already read as x)
                out.append(Unit("x", u.marked or nxt.marked))
                i += 2
                continue
            if pair in ("dz", "dž"):
                out.append(Unit(pair, u.marked or nxt.marked))
                i += 2
                continue
            if u.letter in "lnrtd" and nxt.letter == "j":
                pal = {"l": "ľ", "n": "ň", "r": "ř", "t": "ť", "d": "ď"}[u.letter]
                out.append(Unit(pal, u.marked or nxt.marked))
                i += 2
                continue
        out.append(u)
        i += 1
    return out


def _mark_syllabic(units: list[Unit]) -> None:
    """A plain r or l standing between consonants is a syllable peak."""
    n = len(units)
    for i, u in enumerate(units):
        if u.letter not in ("r", "l"):
            continue
        prev_c = i == 0 or units[i - 1].consonant
        next_c = i == n - 1 or units[i + 1].consonant
        if prev_c and next_c and n > 1:
            u.letter += "̥"


def _syllabify(units: list[Unit]) -> list[list[int]]:
    nuclei = [i for i, u in enumerate(units) if u.nucleus]
    if not nuclei:
        return [list(range(len(units)))]
    syllables: list[list[int]] = []
    start = 0
    for k, ni in enumerate(nuclei):
        if k + 1 < len(nuclei):
            between = list(range(ni + 1, nuclei[k + 1]))
            coda = 1 if len(between) >= 2 and units[between[0]].letter in ("r", "l", "j") else 0
            end = ni + 1 + coda
        else:
            end = len(units)
        syllables.append(list(range(start, end)))
        start = end
    return syllables


def _stress(units: list[Unit], syllables: list[list[int]]) -> int | None:
    if not any(u.nucleus for u in units):
        return None
    for s, idxs in enumerate(syllables):
        if any(units[i].marked for i in idxs):
            return s
    key = "".join(u.letter for u in units)
    if key in CLITICS:
        return None
    return 0
