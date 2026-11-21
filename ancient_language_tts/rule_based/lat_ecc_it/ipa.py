"""Ecclesiastical Latin (Italian school) text → IPA, by rule.

Implements `data/lat-ecc/phonology.txt`. Macrons decide syllable weight for
stress only; the written vowels are always short open values.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Unstressed monosyllables (prepositions / conjunctions).
_CLITIC_WORDS = frozenset({
    "in", "ad", "ab", "ex", "de", "cum", "et", "sed", "si", "ut", "ne", "quam",
})

# Fixed words that look like enclitic forms but take normal stress.
_FAKE_ENCLITIC = frozenset({"itaque", "utique", "undique", "denique"})

# Words / stems where eu is a diphthong.
_EU_DIPHTHONG = frozenset({
    "heu", "eheu", "seu", "neu", "ceu", "neuter", "neutrum", "neutri",
    "neutro", "neutra", "neutrae", "neutram", "europa", "europae", "orpheus",
})

# ei diphthong only in these (not the pronoun eī, which is two syllables).
_EI_DIPHTHONG = frozenset({"ei", "hei", "dein", "deinde", "deinceps"})

# i between vowels → j in these stems (and their forms).
_J_BETWEEN = (
    "maior", "maioris", "maiorem", "maiores", "maiorum", "maioribus",
    "peior", "peioris", "peiorem", "peiores", "peiorum", "peioribus",
    "eius", "cuius", "huius", "troia", "troiae", "troiam",
)

# After a prefix, these stems begin with consonantal i.
_J_STEMS = (
    "iung", "iunx", "iunct", "iur", "iac", "ieci", "iect", "iuven", "iuv",
)

# Final-stress contracted forms.
_FINAL_STRESS = frozenset({"illic", "adhuc", "istic", "illuc", "illac", "istic"})

# Hebrew names whose ae is two syllables (Michael, Israel, …).
_AE_HIATUS = ("michael", "raphael", "israel", "ismael", "misael", "nathanael")

# A vowel with a diaeresis is a vowel in hiatus (poëta, aër).
_DIAERESIS = {"ä": "a", "ë": "e", "ï": "i", "ö": "o", "ü": "u", "ÿ": "y"}

_FRONT = frozenset({"e", "i", "y", "ae", "oe", "æ", "œ", "ë", "ï", "ÿ"})

_VOWEL_LETTERS = frozenset("aeiouyæœäëïöüÿ")

# Consonants that Italian always makes long between vowels, and the first
# half written before them. ɲ is handled with gn.
_LONG_BETWEEN_VOWELS = {"ʃ": "ʃ", "ts": "t", "dz": "d"}
# Mute + liquid (and f + liquid) stay as an onset.
_ONSET_CLUSTER = frozenset({"p", "t", "k", "b", "d", "g", "f"})
_LIQUIDS = frozenset({"l", "r"})

_PAUSE_KEEP = frozenset(",!?")
_WORD_RE = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūȲȳÆæŒœŸÿÄäËëÏïÖöÜü]+|[^\s]", re.UNICODE)

_LONG_MAP = str.maketrans({
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ȳ": "y",
    "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U", "Ȳ": "Y",
})


@dataclass(frozen=True)
class Seg:
    """One IPA segment. `long` is historical length (vowels only).

    `weightless` marks the first half of an Italian long [t.ts] / [d.dz]:
    it closes the syllable but does not make it heavy for stress.
    """

    ipa: str
    kind: str  # "v" | "c" | "d" (diphthong)
    long: bool = False
    weightless: bool = False


def to_ipa(text: str) -> str:
    """Transcribe Ecclesiastical Latin `text` into IPA."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    out: list[str] = []
    for raw in _WORD_RE.findall(cleaned):
        if len(raw) == 1 and not raw.isalpha() and raw not in "āēīōūȳæœĀĒĪŌŪȲÆŒ":
            mark = raw
            if mark in ";:":
                mark = "."
            if mark in _PAUSE_KEEP or mark == ".":
                if out and out[-1] == " ":
                    out.pop()
                out.append(mark)
                out.append(" ")
            continue
        out.append(_word_ipa(raw))
        out.append(" ")
    return "".join(out).strip()


def _word_ipa(raw: str) -> str:
    surface = raw
    folded = _fold(surface)
    lower = folded.lower()
    bare = _bare(lower)
    long_mask = _long_mask(surface)
    graphemes = _graphemes(lower)
    segs = _lengthen(_segments(graphemes, bare, long_mask))
    if not segs:
        return ""
    syllables = _syllabify(segs)
    stress = _stress_index(syllables, bare)
    return _format(syllables, stress)


def _fold(text: str) -> str:
    # Combining macrons → precomposed where possible, then strip unknowns.
    text = unicodedata.normalize("NFC", text)
    return text


def _long_mask(surface: str) -> list[bool]:
    """Per-character length flags on the macron-stripped lowercase form."""
    nfc = unicodedata.normalize("NFC", surface)
    chars: list[tuple[str, bool]] = []
    for ch in nfc:
        if unicodedata.combining(ch):
            continue
        name = unicodedata.name(ch, "")
        long = "MACRON" in name or ch in "āēīōūȳĀĒĪŌŪȲ"
        base = ch.translate(_LONG_MAP).lower()
        if base in "æœ":
            # æ/œ become two letters ae/oe, both short for weight (the digraph
            # is a diphthong historically, handled as one ɛ with long=True).
            chars.append((base, True))
        else:
            chars.append((base, long))
    # Expand æ/œ to two letters for index alignment with grapheme string.
    expanded: list[bool] = []
    for base, long in chars:
        if base == "æ":
            expanded.extend([long, False])
        elif base == "œ":
            expanded.extend([long, False])
        else:
            expanded.append(long)
    return expanded


def _bare(surface: str) -> str:
    nfc = unicodedata.normalize("NFC", surface)
    out: list[str] = []
    for ch in nfc:
        if unicodedata.combining(ch):
            continue
        base = ch.translate(_LONG_MAP).lower()
        if base == "æ":
            out.append("ae")
        elif base == "œ":
            out.append("oe")
        else:
            out.append(base)
    return "".join(out)


def _graphemes(lower: str) -> list[str]:
    """Split the bare lowercase form into digraph-aware letters."""
    s = _bare(lower) if any(c in lower for c in "āēīōūȳæœĀĒĪŌŪȲÆŒ") else lower.translate(_LONG_MAP).lower()
    # lower may already be folded; ensure æ/œ gone.
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = unicodedata.normalize("NFC", s)
    bare = []
    for ch in s:
        if unicodedata.combining(ch):
            continue
        bare.append(ch.translate(_LONG_MAP).lower())
    s = "".join(bare).replace("æ", "ae").replace("œ", "oe")
    out: list[str] = []
    hiatus = s.startswith(_AE_HIATUS)
    i = 0
    while i < len(s):
        if hiatus and s[i:i + 2] == "ae":
            out.append("a")
            i += 1
            continue
        if s[i:i + 2] in ("ae", "oe", "au", "eu", "ei", "ch", "ph", "th", "rh", "qu", "gn", "sc", "xc", "bs", "bt"):
            # Most digraphs are decided later; keep letters separate except ae/oe.
            if s[i:i + 2] in ("ae", "oe"):
                out.append(s[i:i + 2])
                i += 2
                continue
        out.append(s[i])
        i += 1
    return out


def _letter_string(graphemes: list[str]) -> str:
    return "".join(graphemes)


def _is_front(graphemes: list[str], index: int) -> bool:
    if index >= len(graphemes):
        return False
    g = graphemes[index]
    if g in _FRONT:
        return True
    if g == "a" and index + 1 < len(graphemes) and graphemes[index + 1] == "e":
        return True  # shouldn't happen if ae is one grapheme
    return False


def _next_vowelish(graphemes: list[str], index: int) -> str | None:
    if index >= len(graphemes):
        return None
    return graphemes[index]


def _segments(graphemes: list[str], bare: str, long_mask: list[bool]) -> list[Seg]:
    s = bare
    longs = _align_longs(graphemes, long_mask, bare)
    segs: list[Seg] = []
    i = 0
    while i < len(graphemes):
        g = graphemes[i]
        nxt = graphemes[i + 1] if i + 1 < len(graphemes) else ""
        prev = graphemes[i - 1] if i else ""

        if g == "ae":
            segs.append(Seg("ɛ", "v", True))
            i += 1
            continue
        if g == "oe":
            segs.append(Seg("ɛ", "v", True))
            i += 1
            continue
        if g == "a" and nxt == "u":
            segs.append(Seg("au", "d", True))
            i += 2
            continue
        if g == "e" and nxt == "u" and _eu_diphthong(s):
            segs.append(Seg("ɛu", "d", True))
            i += 2
            continue
        if g == "e" and nxt == "i" and s in _EI_DIPHTHONG:
            # Pronoun eī carries a macron on i and stays two syllables.
            i_long = longs[i + 1] if i + 1 < len(longs) else False
            if not i_long:
                segs.append(Seg("ɛi", "d", True))
                i += 2
                continue

        if g == "c" and nxt == "h":
            segs.append(Seg("k", "c"))
            i += 2
            continue
        if g == "p" and nxt == "h":
            segs.append(Seg("f", "c"))
            i += 2
            continue
        if g == "t" and nxt == "h":
            segs.append(Seg("t", "c"))
            i += 2
            continue
        if g == "r" and nxt == "h":
            segs.append(Seg("r", "c"))
            i += 2
            continue

        if g == "q" and nxt == "u":
            segs.append(Seg("kʷ", "c"))
            i += 2
            continue

        if g == "g" and nxt == "n":
            segs.append(Seg("ɲ", "c"))
            if i != 0:
                segs.append(Seg("ɲ", "c"))
            i += 2
            continue

        if g == "x" and nxt == "c":
            if _is_front(graphemes, i + 2):
                segs.append(Seg("k", "c"))
                segs.append(Seg("ʃ", "c"))
            else:
                segs.append(Seg("k", "c"))
                segs.append(Seg("s", "c"))
                segs.append(Seg("k", "c"))
            i += 2
            continue

        if g == "s" and nxt == "c" and _is_front(graphemes, i + 2):
            segs.append(Seg("ʃ", "c"))
            i += 2
            continue

        # Word-initial ex- before vowel or h → [ɛgz]; exs + vowel → [ɛgz] too.
        if g == "e" and i == 0 and nxt == "x":
            after = graphemes[i + 2] if i + 2 < len(graphemes) else ""
            after2 = graphemes[i + 3] if i + 3 < len(graphemes) else ""
            skip = 0
            if after and (after[0] in _VOWEL_LETTERS or after in ("ae", "oe") or after == "h"):
                skip = 2
            elif after == "s" and after2 and (after2[0] in _VOWEL_LETTERS or after2 in ("ae", "oe")):
                skip = 3
            if skip:
                segs.append(Seg("ɛ", "v", longs[i] if i < len(longs) else False))
                segs.append(Seg("g", "c"))
                segs.append(Seg("z", "c"))
                i += skip
                continue

        if g == "x":
            segs.append(Seg("k", "c"))
            segs.append(Seg("s", "c"))
            i += 1
            continue

        if g == "t" and nxt == "i":
            after_i = graphemes[i + 2] if i + 2 < len(graphemes) else ""
            preceded = prev in ("s", "t", "x")
            if graphemes[i:] == ["t", "i", "e", "r"]:
                preceded = True  # old passive infinitive -tier: [ti]
            if after_i and (after_i[0] in _VOWEL_LETTERS or after_i in ("ae", "oe")) and not preceded:
                segs.append(Seg("ts", "c"))
                segs.append(Seg("i", "v", longs[i + 1] if i + 1 < len(longs) else False))
                i += 2
                continue

        if g == "n" and nxt == "g":
            after = graphemes[i + 2] if i + 2 < len(graphemes) else ""
            if after == "u":
                after_u = graphemes[i + 3] if i + 3 < len(graphemes) else ""
                if after_u and (after_u[0] in _VOWEL_LETTERS or after_u in ("ae", "oe")):
                    segs.append(Seg("ŋ", "c"))
                    segs.append(Seg("gʷ", "c"))
                    i += 3
                    continue

        if g == "h":
            if _h_as_k(s):
                segs.append(Seg("k", "c"))
            i += 1
            continue

        # Doubled c/g: soft geminate before a front vowel → [t.tʃ] / [d.dʒ].
        if g == "c" and nxt == "c":
            if _is_front(graphemes, i + 2):
                segs.append(Seg("t", "c"))
                segs.append(Seg("tʃ", "c"))
            else:
                segs.append(Seg("k", "c"))
                segs.append(Seg("k", "c"))
            i += 2
            continue
        if g == "g" and nxt == "g":
            if _is_front(graphemes, i + 2):
                segs.append(Seg("d", "c"))
                segs.append(Seg("dʒ", "c"))
            else:
                segs.append(Seg("g", "c"))
                segs.append(Seg("g", "c"))
            i += 2
            continue

        if g == "c":
            segs.append(Seg("tʃ" if _is_front(graphemes, i + 1) else "k", "c"))
            i += 1
            continue
        if g == "g":
            segs.append(Seg("dʒ" if _is_front(graphemes, i + 1) else "g", "c"))
            i += 1
            continue

        # n → ŋ before hard c/ch/g/qu/x (not before soft c/g); n → ɱ before
        # f/ph/v. m never assimilates (Canepari).
        if g == "n":
            if _nasal_velar(graphemes, i):
                segs.append(Seg("ŋ", "c"))
            elif _nasal_labiodental(graphemes, i):
                segs.append(Seg("ɱ", "c"))
            else:
                segs.append(Seg("n", "c"))
            i += 1
            continue

        if g == "z":
            segs.append(Seg("dz", "c"))
            i += 1
            continue

        if g == "j":
            segs.append(Seg("j", "c"))
            i += 1
            continue
        if g == "i" and _i_is_j(graphemes, i, s):
            segs.append(Seg("j", "c"))
            i += 1
            continue

        if g == "y":
            segs.append(Seg("i", "v", longs[i] if i < len(longs) else False))
            i += 1
            continue

        if g == "v":
            segs.append(Seg("v", "c"))
            i += 1
            continue
        if g == "u" and _u_is_consonant(graphemes, i, s):
            segs.append(Seg("v", "c"))
            i += 1
            continue

        if g == "k":
            segs.append(Seg("k", "c"))
            i += 1
            continue

        if g in _DIAERESIS:
            base = _DIAERESIS[g]
            segs.append(Seg({"e": "ɛ", "o": "ɔ", "y": "i"}.get(base, base), "v", longs[i] if i < len(longs) else False))
            i += 1
            continue

        if g in ("a", "e", "i", "o", "u") or g in ("ae", "oe"):
            ipa = {"a": "a", "e": "ɛ", "i": "i", "o": "ɔ", "u": "u", "ae": "ɛ", "oe": "ɛ"}[g]
            segs.append(Seg(ipa, "v", longs[i] if i < len(longs) else (g in ("ae", "oe"))))
            i += 1
            continue

        if g in "pbtdfsmnlr":
            segs.append(Seg(g, "c"))
            i += 1
            continue

        i += 1
    return segs


def _lengthen(segs: list[Seg]) -> list[Seg]:
    """Italian long [ʃ], [ts], [dz] between vowels: write them as geminates."""
    out: list[Seg] = []
    for k, seg in enumerate(segs):
        first = _LONG_BETWEEN_VOWELS.get(seg.ipa)
        if (
            first
            and k > 0
            and segs[k - 1].kind in ("v", "d")
            and k + 1 < len(segs)
            and segs[k + 1].kind in ("v", "d")
        ):
            out.append(Seg(first, "c", weightless=seg.ipa != "ʃ"))
        out.append(seg)
    return out


def _nasal_labiodental(graphemes: list[str], i: int) -> bool:
    """True when n at `i` is [ɱ]: before f, ph, or v."""
    nxt = graphemes[i + 1] if i + 1 < len(graphemes) else ""
    if nxt in ("f", "v"):
        return True
    return nxt == "p" and i + 2 < len(graphemes) and graphemes[i + 2] == "h"


def _nasal_velar(graphemes: list[str], i: int) -> bool:
    """True when n at `i` assimilates to [ŋ]. Soft c/g do not trigger it."""
    nxt = graphemes[i + 1] if i + 1 < len(graphemes) else ""
    if nxt == "x":
        return True
    if nxt == "q" and i + 2 < len(graphemes) and graphemes[i + 2] == "u":
        return True
    if nxt == "c":
        after = i + 2
        if after < len(graphemes) and graphemes[after] == "h":
            return True
        return not _is_front(graphemes, after)
    if nxt == "g":
        return not _is_front(graphemes, i + 2)
    return False


def _align_longs(graphemes: list[str], long_mask: list[bool], bare: str) -> list[bool]:
    longs: list[bool] = []
    pos = 0
    for g in graphemes:
        flags = long_mask[pos:pos + len(g)] if pos < len(long_mask) else []
        longs.append(any(flags) if flags else False)
        pos += len(g)
    return longs


def _eu_diphthong(word: str) -> bool:
    if word in _EU_DIPHTHONG:
        return True
    return any(word.startswith(stem) for stem in ("europ", "orphe", "neut"))


def _h_as_k(word: str) -> bool:
    # mihi, nihil, and forms/compounds built on them.
    return (
        word.startswith("mihi")
        or word.startswith("nihil")
        or "mihi" in word
        or word.startswith("nihili")
    )


def _i_is_j(graphemes: list[str], i: int, word: str) -> bool:
    nxt = graphemes[i + 1] if i + 1 < len(graphemes) else ""
    if not nxt or not (nxt[0] in _VOWEL_LETTERS or nxt in ("ae", "oe")):
        return False
    # Word-initial i before vowel (iam, Iesus, iucundus).
    if i == 0:
        return True
    # Known stems with intervocalic i → j (maior, eius, …).
    if any(word.startswith(stem) for stem in _J_BETWEEN):
        prev = graphemes[i - 1]
        if prev[0] in _VOWEL_LETTERS or prev in ("ae", "oe"):
            return True
    # Prefix + iungō / iūs / iaciō stems: coniungo, iniuria, conicio.
    rest = word[i:]
    if any(rest.startswith(stem) for stem in _J_STEMS):
        return True
    # abiit, potuit, filii: vocalic i in hiatus.
    return False


def _u_is_consonant(graphemes: list[str], i: int, word: str) -> bool:
    # suadeo, suavis, suesco and derivatives: u after s before a vowel → v
    if i >= 1 and graphemes[i - 1] == "s":
        nxt = graphemes[i + 1] if i + 1 < len(graphemes) else ""
        if nxt and (nxt[0] in _VOWEL_LETTERS or nxt in ("ae", "oe")):
            stem = word[i - 1:]
            if stem.startswith(("sua", "sue", "sui", "suo")) and not word in (
                "suus", "sua", "suum", "sui", "suo", "suis", "suorum", "suarum",
            ):
                # suus keeps vocalic u; suavis etc. have consonantal.
                if word.startswith(("suav", "suad", "suesc", "suet")):
                    return True
    return False


def _syllabify(segs: list[Seg]) -> list[list[Seg]]:
    # Find nuclei.
    nuclei = [i for i, s in enumerate(segs) if s.kind in ("v", "d")]
    if not nuclei:
        return [segs] if segs else []
    # Assign each segment to a syllable.
    bounds: list[tuple[int, int]] = []  # inclusive start, exclusive end per syllable
    for si, ni in enumerate(nuclei):
        if si == 0:
            start = 0
        else:
            start = bounds[si - 1][1]
        if si + 1 < len(nuclei):
            nxt = nuclei[si + 1]
            between = segs[ni + 1:nxt]
            split = _coda_onset(between)
            end = ni + 1 + split
        else:
            end = len(segs)
        bounds.append((start, end))
    return [segs[a:b] for a, b in bounds]


def _coda_onset(between: list[Seg]) -> int:
    """How many of `between` close the previous syllable."""
    n = len(between)
    if n == 0:
        return 0
    if n == 1:
        return 0  # single consonant with following vowel
    # mute/f + liquid stay together as onset
    if n >= 2 and between[-2].ipa in _ONSET_CLUSTER and between[-1].ipa in _LIQUIDS:
        return n - 2
    # last goes with following vowel
    return n - 1


def _has_enclitic(bare: str) -> bool:
    """True when -que/-ve/-ne is the interrogative/conjunctive enclitic.

    `-ne` only after a consonant (estne), so ablatives like homine / flumine
    keep ordinary stress.
    """
    if len(bare) <= 2:
        return False
    if bare.endswith("que") or bare.endswith("ve"):
        return True
    if bare.endswith("ne") and bare[-3] not in "aeiouy":
        return True
    return False


def _is_heavy(syllable: list[Seg], is_penult: bool) -> bool:
    nucleus = next(s for s in syllable if s.kind in ("v", "d"))
    if nucleus.kind == "d" or nucleus.long:
        return True
    # Closed by a consonant?
    after = False
    for s in syllable:
        if s.kind in ("v", "d"):
            after = True
            continue
        if after and not s.weightless:
            # coda consonant — but stop+liquid in same syllable don't close
            # for weight; they're onset of next in Classical, already split.
            return True
    return False


def _stress_index(syllables: list[list[Seg]], bare: str) -> int | None:
    n = len(syllables)
    if n == 0:
        return None
    if bare in _CLITIC_WORDS:
        return None
    if bare in _FINAL_STRESS:
        return n - 1
    if bare not in _FAKE_ENCLITIC and _has_enclitic(bare):
        return max(0, n - 2)
    if n == 1:
        return 0
    if n == 2:
        return 0
    if _is_heavy(syllables[-2], True):
        return n - 2
    return n - 3


def _format(syllables: list[list[Seg]], stress: int | None) -> str:
    """Join syllables with `.`; `ˈ` replaces the dot before the stressed one."""
    if not syllables:
        return ""
    parts = ["".join(s.ipa for s in syl) for syl in syllables]
    if stress is None:
        return ".".join(parts)
    out = parts[0] if stress != 0 else "ˈ" + parts[0]
    for i, body in enumerate(parts[1:], start=1):
        if i == stress:
            out += "ˈ" + body
        else:
            out += "." + body
    return out


# --- self-check -----------------------------------------------------------

def _load_examples() -> list[tuple[str, str]]:
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "lat-ecc" / "examples.txt"
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "→" not in line:
            continue
        left, right = line.split("→", 1)
        rows.append((left.strip(), right.strip()))
    return rows


if __name__ == "__main__":
    bad = 0
    for src, want in _load_examples():
        got = to_ipa(src)
        if got != want:
            bad += 1
            print("FAIL")
            print(" src:", src)
            print("want:", want)
            print(" got:", got)
            print()
    total = len(_load_examples())
    print(f"{total - bad}/{total} matched")
