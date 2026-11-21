"""Shared reading of Koine Greek spelling: tokens, letter units, and stress.

Both `ipa.to_ipa` and the direct respellers start here. This module only
reads the spelling (polytonic, monotonic, or Latin transliteration); it
knows nothing of IPA or of any modern target.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

ROUGH = "̔"
SMOOTH = "̓"
ACUTE = "́"
GRAVE = "̀"
CIRCUMFLEX = "͂"
IOTA_SUB = "ͅ"
DIAERESIS = "̈"
MACRON = "̄"

ACCENTS = frozenset({ACUTE, GRAVE, CIRCUMFLEX})
BREATHINGS = frozenset({ROUGH, SMOOTH})
# Marks that only polytonic spelling writes.
POLYTONIC_MARKS = frozenset({ROUGH, SMOOTH, GRAVE, CIRCUMFLEX, IOTA_SUB})

GREEK_VOWELS = frozenset("αεηιουω")
GREEK_CONSONANTS = frozenset("βγδζθκλμνξπρστφχψ")
# Two vowel letters read as one nucleus. αυ ευ ηυ are the labial ones.
DIPHTHONGS = frozenset({"αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ"})
LABIAL_DIPHTHONGS = frozenset({"αυ", "ευ", "ηυ"})

_VARIANTS = {"ς": "σ", "ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϕ": "φ", "ϰ": "κ", "ϱ": "ρ"}

APOSTROPHES = frozenset({"'", "’", "ʼ", "᾽", "᾿", "ʹ"})
HYPHENS = frozenset({"-", "‐", "‑"})

# Monosyllables that stay unstressed even when an accent is written on them.
# Forms that change meaning with an accent (ὅ, ἥ, οἵ, αἵ relatives, ἕν and
# εἷς "one", εἶ "you are", οὗ "where", accented orthotonic pronouns) are
# not here: written with an accent they are stressed words.
ACCENTED_CLITICS = frozenset(w.replace("ς", "σ") for w in {
    "το", "του", "της", "τω", "τη", "τον", "την", "τα", "των", "τοις", "ταις",
    "τους", "τας", "προς", "συν", "και", "δε", "γαρ", "τε", "μεν", "ουν", "αν",
    "μη",
})

# Every unstressed monosyllable of `phonology.txt` (plus the common
# monosyllabic enclitics). Used for text written without polytonic marks,
# where a bare monosyllable does not tell a clitic from a stressed word.
CLITICS = ACCENTED_CLITICS | frozenset(w.replace("ς", "σ") for w in {
    "ο", "η", "οι", "αι", "εν", "εις", "εκ", "εξ", "ει", "ως", "ου", "ουκ",
    "ουχ", "μου", "σου", "μοι", "σοι", "με", "σε", "τις", "τι", "γε", "πως",
    "που",
})

# Pause marks of the source. `;` is decided per text (see `tokenize`).
_PAUSES = {
    ",": ",", ".": ".", "!": "!", "?": "?", ";": "?",
    ":": ".", "·": ".", "·": ".", "…": ".",
}


@dataclass
class Unit:
    """One vowel nucleus, one consonant letter, or a forced syllable break."""

    kind: str  # "v" | "c" | "b"
    text: str  # base Greek letters: "ει", "γ", ...
    accent: bool = False
    rough: bool = False


@dataclass
class Word:
    units: list[Unit]
    stress: int | None  # index into `units` of the stressed vowel
    elided: bool = False


# --- tokens ---------------------------------------------------------------


def tokenize(text: str) -> list[Word | str]:
    """Split `text` into `Word`s and pause strings (`, . ! ?`)."""
    s = unicodedata.normalize("NFD", text)
    greek = any("Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in s)
    polytonic = any(ch in POLYTONIC_MARKS for ch in s)
    raw: list[tuple[str, bool] | str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isalpha():
            j = i
            while j < n and (
                s[j].isalpha()
                or unicodedata.combining(s[j])
                or (s[j] in HYPHENS and j + 1 < n and s[j + 1].isalpha())
            ):
                j += 1
            elided = j < n and s[j] in APOSTROPHES
            raw.append((s[i:j], elided))
            i = j + 1 if elided else j
            continue
        if ch == ";":
            raw.append("?" if greek else ".")
        elif ch in _PAUSES:
            raw.append(_PAUSES[ch])
        i += 1

    out: list[Word | str] = []
    for item in raw:
        if isinstance(item, str):
            if out and not isinstance(out[-1], str):
                out.append(item)
            continue
        chunk, elided = item
        word = _read_word(chunk, elided, polytonic)
        if word is not None:
            out.append(word)
    return out


def join(pieces: list[tuple[str, str]]) -> str:
    """Join `("w", word)` and `("p", pause)` pieces with spaces."""
    out: list[str] = []
    for kind, value in pieces:
        if kind == "p":
            if out:
                out[-1] += value
        elif value:
            out.append(value)
    return " ".join(out)


# --- one word -------------------------------------------------------------


def _clusters(chunk: str) -> list[tuple[str, set[str]]]:
    """Base characters with their combining marks."""
    out: list[tuple[str, set[str]]] = []
    for ch in chunk:
        if unicodedata.combining(ch):
            if out:
                out[-1][1].add(ch)
            continue
        out.append((ch, set()))
    return out


def _read_word(chunk: str, elided: bool, polytonic: bool) -> Word | None:
    letters = _clusters(chunk)
    latin = any("a" <= b.lower() <= "z" for b, _ in letters)
    if latin:
        letters = _latin_to_greek(letters)
        polytonic = False
    units = _units(letters)
    if not units:
        return None
    key = "".join(u.text for u in units if u.kind != "b")
    # ἐκ- before λ or ρ is a prefix: ἐκ-λέγομαι, ἐκ-ρέω.
    if (
        len(units) > 2
        and units[0].text == "ε"
        and units[1].text == "κ"
        and units[2].text in ("λ", "ρ")
    ):
        units.insert(2, Unit("b", ""))
    return Word(units, _stress(units, key, elided, polytonic), elided)


def _units(letters: list[tuple[str, set[str]]]) -> list[Unit]:
    seq: list[tuple[str, set[str]]] = []
    for base, marks in letters:
        if base in HYPHENS:
            seq.append(("-", set()))
            continue
        b = base.lower()
        b = _VARIANTS.get(b, b)
        if b in GREEK_VOWELS or b in GREEK_CONSONANTS:
            seq.append((b, marks))
    units: list[Unit] = []
    i = 0
    while i < len(seq):
        b, marks = seq[i]
        if b == "-":
            if units and units[-1].kind != "b":
                units.append(Unit("b", ""))
            i += 1
            continue
        if b in GREEK_CONSONANTS:
            units.append(Unit("c", b))
            i += 1
            continue
        if i + 1 < len(seq):
            b2, marks2 = seq[i + 1]
            if (
                b + b2 in DIPHTHONGS
                and DIAERESIS not in marks2
                and not (marks & (ACCENTS | BREATHINGS | {IOTA_SUB}))
            ):
                both = marks | marks2
                units.append(Unit("v", b + b2, bool(both & ACCENTS), ROUGH in both))
                i += 2
                continue
        units.append(Unit("v", b, bool(marks & ACCENTS), ROUGH in marks))
        i += 1
    while units and units[-1].kind == "b":
        units.pop()
    return units


def _stress(units: list[Unit], key: str, elided: bool, polytonic: bool) -> int | None:
    if elided:
        return None
    vowels = [i for i, u in enumerate(units) if u.kind == "v"]
    if not vowels:
        return None
    # The first written accent is the word's own; a second one comes from
    # a following enclitic (ὄνομά σου).
    accented = next((i for i in vowels if units[i].accent), None)
    if len(vowels) == 1:
        if accented is not None:
            return None if key in ACCENTED_CLITICS else accented
        if polytonic:
            return None
        return None if key in CLITICS else vowels[0]
    return accented


# --- Latin transliteration ------------------------------------------------

_LATIN_DIGRAPHS = {
    "nch": "γχ", "nkh": "γχ",
    "ph": "φ", "th": "θ", "ch": "χ", "kh": "χ", "rh": "ρ", "ps": "ψ",
    "ng": "γγ", "nk": "γκ", "nx": "γξ",
}
_LATIN_SINGLE = {
    "a": "α", "b": "β", "c": "κ", "d": "δ", "e": "ε", "f": "φ", "g": "γ",
    "i": "ι", "j": "ι", "k": "κ", "l": "λ", "m": "μ", "n": "ν", "o": "ο",
    "p": "π", "q": "κ", "r": "ρ", "s": "σ", "t": "τ", "u": "υ", "v": "β",
    "w": "ω", "x": "ξ", "y": "υ", "z": "ζ",
}


def _latin_to_greek(letters: list[tuple[str, set[str]]]) -> list[tuple[str, set[str]]]:
    """ph th ch → φ θ χ, y → υ, ē → η, ō → ω, initial h → rough breathing."""
    bases: list[str] = []
    marks: list[set[str]] = []
    for b, m in letters:
        b = b.lower()
        m = set(m)
        if MACRON in m:
            m.discard(MACRON)
            b = {"e": "η", "o": "ω"}.get(b, b)
        bases.append(b)
        marks.append(m)
    out: list[tuple[str, set[str]]] = []
    rough = False
    i, n = 0, len(bases)
    if n > 1 and bases[0] == "h" and bases[1] in "aeiouyηω":
        rough = True
        i = 1
    while i < n:
        b = bases[i]
        if b in "ηω":
            out.append((b, marks[i]))
            i += 1
            continue
        for size in (3, 2):
            chunk = "".join(bases[i:i + size])
            if chunk in _LATIN_DIGRAPHS:
                greek = _LATIN_DIGRAPHS[chunk]
                for k, g in enumerate(greek):
                    out.append((g, set() if k + 1 < len(greek) else set(marks[i + size - 1])))
                i += size
                break
        else:
            if b in HYPHENS:
                out.append((b, set()))
            elif b in _LATIN_SINGLE:
                out.append((_LATIN_SINGLE[b], marks[i]))
            i += 1
    if rough:
        k = next((j for j, (b, _) in enumerate(out) if b in GREEK_VOWELS), None)
        if k is not None:
            if (
                k + 1 < len(out)
                and out[k][0] + out[k + 1][0] in DIPHTHONGS
                and DIAERESIS not in out[k + 1][1]
            ):
                k += 1
            out[k][1].add(ROUGH)
    return out
