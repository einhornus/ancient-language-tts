"""Classical Latin spelling → phonemic syllables, by rule.

Implements the analysis half of `data/lat/phonology.txt`: graphemes,
consonantal i and u, diphthongs, nasal vowels, assimilation, syllables and
stress. The result is a list of items, one per word or pause mark:

    Word   a list of `Syl(stressed, phones)`; phones are IPA symbols
    str    a pause mark: "," "!" "?" or "."

`ipa.to_ipa` writes this structure out as an IPA line; `direct_it` spells it
straight into Italian, so the direct respelling never builds or parses IPA
text. Pure standard library.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- data -----------------------------------------------------------------

_VOWELS = frozenset("aeiouy")

# Monosyllabic prepositions and conjunctions: no stress mark. Keys use i for
# j and u for v (see `_key`).
_UNSTRESSED = frozenset("""
    a ab abs ad cis cum de e ex in ob per post prae pro sub trans
    ac an ast at aut dum et nam ne nec neu quam sed seu si sin ut uel
""".split())

# Fixed words that only look like -que forms: normal stress.
_FAKE_ENCLITIC = frozenset({"itaque", "utique", "undique", "denique"})

# Words that lost or contracted their last syllable: final stress.
_FINAL_STRESS = frozenset({
    "illic", "illuc", "illac", "istic", "istuc", "istac",
    "adhuc", "adduc", "nostras", "uestras", "cuias",
})

# Genitive / vocative in -ī of nouns in -ius: stress of the full form, which
# is the syllable before the -ī.
_IUS_GENITIVE = frozenset({
    "uergili", "ualeri", "mercuri", "ingeni", "consili", "imperi", "auxili",
    "ouidi", "aemili", "caecili", "lucili", "basili", "sergi", "uirgili",
})

# Classical short final vowel even when an edition prints a macron.
_SHORT_FINAL = frozenset({
    "ego", "mihi", "tibi", "sibi", "ubi", "ibi", "nisi", "quasi", "modo",
    "bene", "male", "cito", "duo",
})

# ei / eu / ui diphthongs.
_EI_DIPHTHONG = frozenset({"ei", "hei", "dein", "deinde", "deinceps"})
_EU_DIPHTHONG = frozenset({
    "heu", "eheu", "seu", "neu", "ceu", "orpheus", "theseus", "perseus",
    "prometheus", "nereus", "peleus", "atreus", "tydeus", "zeus", "typhoeus",
})
_UI_DIPHTHONG = frozenset({"cui", "huic", "hui"})

# Word-initial i before a vowel that is still a vowel (Greek words, forms of
# eō "go").
_I_VOCALIC_WORDS = frozenset({
    "iulus", "iuli", "iulo", "iulum", "iule", "io", "ii", "iit", "iere",
    "ierunt", "iens", "iisse", "iissem",
})
_I_VOCALIC_PREFIXES = ("iamb", "iason", "ionic", "ionia")

# i between vowels that stays a vowel (Gāius and the like).
_I_HIATUS_PREFIXES = ("gai", "laius", "achai")

# Prefixes after which i + a stem of iungō, iūs, iaciō … is [j].
_J_PREFIXES = frozenset({
    "ab", "ad", "circum", "con", "de", "dis", "e", "in", "inter", "ob", "per",
    "prae", "pro", "re", "se", "sub", "super", "tra", "trans",
})
_J_STEMS = ("iung", "iunx", "iunct", "iug", "iur", "iud", "iect", "iac", "ieci",
            "iuu", "iut")

# Compounds of iaciō written with one i: [jɪ] (after a vowel [j.jɪ]).
_IC_PREFIXES = frozenset({
    "ab", "ad", "con", "de", "dis", "e", "in", "ob", "pro", "re", "sub", "tra",
})

# u is [w] in suādeō, suāvis, suēscō and their derivatives.
_SU_STEMS = ("suad", "suau", "suesc", "suet", "sueu")

# Prefixes whose boundary overrides stop + liquid (abrumpō, obruō).
_BOUNDARY_PREFIXES = ("ab", "ob", "sub", "ad")

# Hidden and unmarked quantities for words printed with no macron at all.
# The keys are the plain spelling; values carry the macrons. Grammar cannot
# be parsed here, so an ambiguous form gets its commonest reading (rōma is
# the nominative).
_LEXICON_WORDS = """
    rōma rōmae rōmam rōmānus rōmānī rōmānōrum rōmānīs rōmānum rōmāna
    nōn rēx rēgis rēgī rēgem rēgēs rēgum rēgnum rēgna rēgnī rēgnō
    mīlle mīlia mīlium nūllus nūlla nūllum nūllī nūllō nūllae nūllōs
    āctus āctum ācta lēctus lēctum lēcta scrīptus scrīptum scrīpta
    sānctus sāncta sānctum sānctī quīnque quīntus quīnta quīntum
    dīxit dīxī dīxērunt vīxit vīxī lūx lūcem lūcis pāx pācem pācis
    lēx lēgem lēgēs lēgis vōx vōcem vōcis sōl sōlis sōlem
    nōs vōs tū mē tē sē sēsē dē prō sī nē quī quō quā
    trēs ūnus ūna ūnum ūnīus fīlius fīlia fīlium fīliī fīliōrum
    amīcus amīcī amīcum amīca amīcōs amīcīs poēta poētae poētam
    māter mātrem mātris frāter frātrem frātris vīta vītam vītae
    nātūra nātūram nātūrae fēmina fēminae cīvis cīvēs cīvium
    lībertās lībertātem lībertātis virtūs virtūtem virtūtis
    fortūna fortūnam fortūnae glōria glōriam glōriae senātus
    iūlius iūlia iūliī iūlium iūstus iūsta iūstum iūstitia iūs
    mārcus mārcī mārcum cicerō cicerōnis dīcō dīcit dīcere dīcunt
    vērus vēra vērum vēritās fīnis fīnem fīnēs plūs plūrēs
    """
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏ̀-ͯ'’]+|[^\s]")
_PAUSES = {",": ",", "!": "!", "?": "?", ".": ".", ";": ".", ":": "."}

_STOPS = frozenset({"p", "t", "k", "b", "d", "g", "pʰ", "tʰ", "kʰ"})
_LIQUIDS = frozenset({"l", "ɫ", "r"})
_SHORT = {"a": "a", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "y": "y"}
_LONG = {"a": "aː", "e": "eː", "i": "iː", "o": "oː", "u": "uː", "y": "yː"}
_NASAL = {"a": "ãː", "e": "ẽː", "i": "ĩː", "o": "õː", "u": "ũː", "y": "ỹː"}
_I_LIKE = frozenset({"ɪ", "i", "iː", "ĩː", "y", "yː", "ỹː"})


@dataclass
class Syl:
    """One syllable: its phones (IPA symbols) and whether it is stressed."""

    stressed: bool
    phones: list[str] = field(default_factory=list)


Word = list[Syl]
Item = "Word | str"


@dataclass
class _Seg:
    ph: str
    vowel: bool = False
    boundary: bool = False  # a prefix ends after this consonant
    double_l: bool = False


@dataclass
class _Letter:
    ch: str
    long: bool = False
    split: bool = False  # diaeresis: never part of a diphthong


# --- public ---------------------------------------------------------------

def analyse(text: str) -> list:
    """Split `text` into words (lists of `Syl`) and pause marks."""
    items: list = []
    for raw in _TOKEN_RE.findall(text.strip()):
        if raw in _PAUSES:
            mark = _PAUSES[raw]
            if items and not isinstance(items[-1], str):
                items.append(mark)
            elif items and items[-1] != mark:
                items.append(mark)
            continue
        letters = _letters(raw)
        if not letters:
            continue
        word = _word(letters)
        if word:
            items.append(word)
    return items


def is_vowel(ph: str) -> bool:
    return bool(ph) and ph[0] in "aeiouyɛɪɔʊãẽĩõũỹ"


# --- letters --------------------------------------------------------------

def _letters(raw: str) -> list[_Letter]:
    out: list[_Letter] = []
    for ch in unicodedata.normalize("NFD", raw):
        if unicodedata.combining(ch):
            if out:
                if ch == "̄":
                    out[-1].long = True
                elif ch == "̈":
                    out[-1].split = True
                elif ch == "̆":
                    out[-1].long = False
            continue
        low = ch.lower()
        if low == "æ":
            out += [_Letter("a"), _Letter("e")]
        elif low == "œ":
            out += [_Letter("o"), _Letter("e")]
        elif "a" <= low <= "z":
            out.append(_Letter(low))
    return out


def _key(s: str) -> str:
    return s.replace("j", "i").replace("v", "u")


def _build_lexicon() -> dict[str, str]:
    lex: dict[str, str] = {}
    for entry in _LEXICON_WORDS.split():
        letters = _letters(entry)
        lex[_key("".join(l.ch for l in letters))] = entry
    return lex


_LEXICON = _build_lexicon()


def _restore_quantity(letters: list[_Letter]) -> None:
    """Macrons from the lexicon for a word printed with none."""
    if any(l.long for l in letters):
        return
    key = _key("".join(l.ch for l in letters))
    entry = _LEXICON.get(key)
    if entry is None and key.endswith("que") and len(key) > 5:
        entry = _LEXICON.get(key[:-3])  # rōmamque; not -ne/-ve (sine, nāve)
    if entry is None:
        return
    for letter, source in zip(letters, _letters(entry)):
        letter.long = source.long


# --- one word -------------------------------------------------------------

def _word(letters: list[_Letter]) -> Word:
    _restore_quantity(letters)
    s = "".join(l.ch for l in letters)
    key = _key(s)
    if key in _SHORT_FINAL:
        letters[-1].long = False
    roles = _roles(letters, s, key)
    segs = _segments(letters, s, key, roles)
    if not segs:
        return []
    syllables = _syllabify(segs)
    stress = _stress(syllables, letters, key, roles)
    return [Syl(i == stress, [g.ph for g in syl]) for i, syl in enumerate(syllables)]


def _roles(letters: list[_Letter], s: str, key: str) -> list[str]:
    """Role of each letter.

    "V" vowel, "C" consonant, "q" the u of qu, "g" the u of ngu + vowel,
    "j" / "jj" consonantal i, "jI" / "jjI" the single i of an iaciō compound.
    """
    n = len(s)
    roles = ["C" if c not in _VOWELS else "V" for c in s]
    has_v = "v" in s

    # u first: qu, ngu, suādeō, and u printed for v.
    for k, c in enumerate(s):
        if c != "u":
            continue
        nxt = s[k + 1] if k + 1 < n else ""
        if k and s[k - 1] == "q":
            roles[k] = "q"
        elif k >= 2 and s[k - 2:k] == "ng" and nxt in _VOWELS:
            roles[k] = "g"
        elif k and s[k - 1] == "s" and any(key.startswith(p, k - 1) for p in _SU_STEMS):
            roles[k] = "C" if nxt in _VOWELS else "V"
        elif not has_v and not letters[k].long and nxt in _VOWELS:
            # The edition prints u for v: decide from the word.
            prev = s[k - 1] if k else ""
            prev_vowel = prev in "aeoy" or (prev == "u" and roles[k - 1] == "V") or (
                prev == "i" and k - 1 > 0
            )
            if k == 0 or prev_vowel:
                roles[k] = "C"  # uīuō, nouus, auis
            elif nxt == "u" and prev in "rl" and (k + 2 >= n or s[k + 2] not in _VOWELS):
                roles[k] = "C"  # seruus, paruum
    # i and j.
    for k, c in enumerate(s):
        if c not in "ij":
            continue
        nxt = s[k + 1] if k + 1 < n else ""
        next_vowel = nxt in _VOWELS and roles[k + 1] == "V"
        prefix = key[:k]
        if c == "i" and nxt == "c" and k + 2 < n and s[k + 2] in "ie" and prefix in _IC_PREFIXES:
            roles[k] = "jjI" if prefix[-1] in _VOWELS else "jI"
            continue
        if c == "i" and not next_vowel:
            roles[k] = "V"
            continue
        if c == "j" and not next_vowel:
            roles[k] = "j"
            continue
        if k == 0:
            vocalic = key in _I_VOCALIC_WORDS or key.startswith(_I_VOCALIC_PREFIXES)
            roles[k] = "V" if c == "i" and vocalic else "j"
        elif prefix in _J_PREFIXES and any(key.startswith(st, k) for st in _J_STEMS):
            roles[k] = "j"
        elif roles[k - 1] == "V" and not key.startswith(_I_HIATUS_PREFIXES):
            roles[k] = "jj"
        else:
            roles[k] = "j" if c == "j" else "V"
    return roles


def _diphthong(s: str, key: str, k: int, letters: list[_Letter], roles: list[str]) -> str:
    if k + 1 >= len(s) or roles[k + 1] != "V":
        return ""
    if letters[k].long or letters[k + 1].long or letters[k + 1].split:
        return ""
    pair = s[k:k + 2]
    if pair in ("ae", "oe", "au"):
        return pair + "̯"
    if pair == "ei" and key in _EI_DIPHTHONG:
        return "ei̯"
    if pair == "eu" and (key in _EU_DIPHTHONG or key.startswith(("neut", "eur"))):
        return "eu̯"
    if pair == "ui" and key in _UI_DIPHTHONG:
        return "ui̯"
    return ""


def _segments(letters: list[_Letter], s: str, key: str, roles: list[str]) -> list[_Seg]:
    n = len(s)
    segs: list[_Seg] = []
    boundary_at = -1
    for p in _BOUNDARY_PREFIXES:
        if key.startswith(p) and len(key) > len(p) + 1 and key[len(p)] in "lr":
            boundary_at = len(p) - 1
    k = 0
    while k < n:
        c = s[k]
        nxt = s[k + 1] if k + 1 < n else ""
        role = roles[k]
        if role in ("jI", "jjI"):
            segs.append(_Seg("j"))
            if role == "jjI":
                segs.append(_Seg("j"))
            segs.append(_Seg("ɪ", vowel=True))
            k += 1
            continue
        if role in ("j", "jj"):
            segs.append(_Seg("j"))
            if role == "jj":
                segs.append(_Seg("j"))
            k += 1
            continue
        if role == "V":
            diph = _diphthong(s, key, k, letters, roles)
            if diph:
                segs.append(_Seg(diph, vowel=True))
                k += 2
                continue
            if nxt == "n" and k + 2 < n and s[k + 2] in "sf":
                segs.append(_Seg(_NASAL[c], vowel=True))
                k += 2
                continue
            if nxt == "m" and k + 2 == n:
                segs.append(_Seg(_NASAL[c], vowel=True))
                k += 2
                continue
            long = letters[k].long and not (k + 1 < n and roles[k + 1] == "jj")
            segs.append(_Seg((_LONG if long else _SHORT)[c], vowel=True))
            k += 1
            continue
        # consonants
        ph: list[str]
        step = 1
        if c in "uvw":
            ph = ["w"]
        elif c == "q":
            ph = ["kʷ"]
            step = 2 if nxt == "u" else 1
        elif c == "c" and nxt == "h":
            ph, step = ["kʰ"], 2
        elif c == "p" and nxt == "h":
            ph, step = ["pʰ"], 2
        elif c == "t" and nxt == "h":
            ph, step = ["tʰ"], 2
        elif c == "r" and nxt == "h":
            ph, step = ["r"], 2
        elif c in "ck":
            ph = ["k"]
        elif c == "g":
            if k + 1 < n and roles[k + 1] == "g":
                ph, step = ["gʷ"], 2
            elif nxt == "n" and k > 0:
                ph = ["ŋ"]
            else:
                ph = ["g"]
        elif c == "x":
            ph = ["k", "s"]
        elif c == "z":
            between = k > 0 and roles[k - 1] == "V" and nxt in _VOWELS
            ph = ["z", "z"] if between else ["z"]
        elif c == "b":
            ph = ["p"] if nxt in ("s", "t") else ["b"]
        elif c in "nm":
            if nxt and nxt in "cgqkx":
                ph = ["ŋ"]
            else:
                ph = [c]
        elif c == "l":
            segs.append(_Seg("l", double_l=nxt == "l" or (k > 0 and s[k - 1] == "l")))
            if k == boundary_at:
                segs[-1].boundary = True
            k += 1
            continue
        elif c in "pdtfrsh":
            ph = [c]
        else:
            k += 1
            continue
        for p in ph:
            segs.append(_Seg(p))
        if k == boundary_at:
            segs[-1].boundary = True
        k += step

    # Allophones that depend on the next segment.
    for idx, seg in enumerate(segs):
        nxt_seg = segs[idx + 1] if idx + 1 < len(segs) else None
        if seg.ph == "ɪ" and nxt_seg is not None and nxt_seg.vowel:
            seg.ph = "i"
        elif seg.ph == "l" and not seg.double_l:
            if nxt_seg is None or nxt_seg.ph not in _I_LIKE:
                seg.ph = "ɫ"
    return segs


def _syllabify(segs: list[_Seg]) -> list[list[_Seg]]:
    nuclei = [i for i, g in enumerate(segs) if g.vowel]
    if not nuclei:
        return [segs]
    cuts: list[int] = []
    for a, b in zip(nuclei, nuclei[1:]):
        between = segs[a + 1:b]
        m = len(between)
        forced = [i for i, g in enumerate(between) if g.boundary]
        if m == 0:
            cut = b
        elif forced:
            cut = a + 1 + forced[0] + 1
        elif m >= 2 and between[-2].ph in _STOPS and between[-1].ph in _LIQUIDS:
            cut = b - 2
        else:
            cut = b - 1
        cuts.append(cut)
    out: list[list[_Seg]] = []
    start = 0
    for cut in cuts:
        out.append(segs[start:cut])
        start = cut
    out.append(segs[start:])
    return out


def _heavy(syl: list[_Seg]) -> bool:
    nucleus_at = next(i for i, g in enumerate(syl) if g.vowel)
    nucleus = syl[nucleus_at].ph
    if "ː" in nucleus or "̯" in nucleus:
        return True
    return any(g.ph != "h" for g in syl[nucleus_at + 1:])


def _stress(syllables: list[list[_Seg]], letters: list[_Letter], key: str,
            roles: list[str]) -> int | None:
    n = len(syllables)
    if not any(g.vowel for g in syllables[0]):
        return None
    if n == 1:
        return None if key in _UNSTRESSED else 0
    if key in _FINAL_STRESS or (key == "audit" and letters[3].long):
        return n - 1
    if n >= 3 and key in _IUS_GENITIVE:
        return n - 2
    if n >= 3 and _enclitic(key, roles):
        return n - 2
    if n == 2:
        return 0
    return n - 2 if _heavy(syllables[-2]) else n - 3


def _enclitic(key: str, roles: list[str]) -> bool:
    if key.endswith("que"):
        return key not in _FAKE_ENCLITIC
    if key.endswith("ue") and roles[-2] == "C":
        return True
    return key.endswith("ne") and key[-3] not in _VOWELS
