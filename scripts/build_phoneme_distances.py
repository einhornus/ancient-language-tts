"""Write ancient_language_tts/data/phoneme_distances.json.

    python scripts/build_phoneme_distances.py

The table holds a distance, from 0 to 1, between every phoneme a language
can expect (`data/<code>/phonemes.json`) and every glyph a recognized frame
can carry (ZIPA's own characters, and what each language's `zipa.json`
turns them into). check/mismatch.py reads it; a pair it lacks costs 1.

Distances follow a rule of thumb:

* 0.15  very similar: e/ɛ, o/ɔ, i/ɪ, a/æ, j/i, w/u, a sound with and
        without one diacritic (aː/a, kʰ/k, ĩ/i, r̩/r)
* 0.3   neighbors: a/ɛ, e/i, u/o, p/b, s/ʃ, s/θ, f/θ, r/ɾ/ɹ
* 0.4   same class, two features apart: t/s, l/r, e/a, i/u
* 0.5   same class, unrelated: k/f, i/a, m/x
* 0.8   different classes: a/k

Edit the feature tables below and rerun; the JSON is generated, not hand-kept.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ancient_language_tts import ipa_tokens  # noqa: E402
from ancient_language_tts.check import conditioning  # noqa: E402

DATA = ROOT / "ancient_language_tts" / "data"
OUT = DATA / "phoneme_distances.json"

# ------------------------------------------------------------------ vowels
# (height, backness, rounded). Height: i 0 … a 3, with the mid pairs
# (e/ɛ, o/ɔ) and the lax vowels (ɪ, ʊ) close to their neighbors.
_H = {"close": 0.0, "near-close": 0.5, "close-mid": 1.3, "mid": 1.6,
      "open-mid": 1.8, "near-open": 2.5, "open": 3.0}
_B = {"front": 0.0, "central": 0.6, "back": 1.2}
VOWELS = {
    "i": ("close", "front", 0), "y": ("close", "front", 1),
    "ɨ": ("close", "central", 0), "ʉ": ("close", "central", 1),
    "ɯ": ("close", "back", 0), "u": ("close", "back", 1),
    "ɪ": ("near-close", "front", 0), "ʏ": ("near-close", "front", 1),
    "ʊ": ("near-close", "back", 1),
    "e": ("close-mid", "front", 0), "ø": ("close-mid", "front", 1),
    "ɘ": ("close-mid", "central", 0), "ɵ": ("close-mid", "central", 1),
    "ɤ": ("close-mid", "back", 0), "o": ("close-mid", "back", 1),
    "ə": ("mid", "central", 0),
    "ɛ": ("open-mid", "front", 0), "œ": ("open-mid", "front", 1),
    "ɜ": ("open-mid", "central", 0), "ɞ": ("open-mid", "central", 1),
    "ʌ": ("open-mid", "back", 0), "ɔ": ("open-mid", "back", 1),
    "æ": ("near-open", "front", 0), "ɐ": ("near-open", "central", 0),
    "a": ("open", "front", 0), "ɶ": ("open", "front", 1),
    "ɑ": ("open", "back", 0), "ɒ": ("open", "back", 1),
}
ROUND_STEP = 0.9


def vowel_distance(a: str, b: str) -> float:
    ha, ba, ra = VOWELS[a]
    hb, bb, rb = VOWELS[b]
    steps = abs(_H[ha] - _H[hb]) + abs(_B[ba] - _B[bb]) + ROUND_STEP * abs(ra - rb)
    if steps <= 0.7:
        return 0.15
    if steps <= 1.3:
        return 0.3
    if steps <= 2.2:
        return 0.4
    return 0.5


# -------------------------------------------------------------- consonants
# (place, manner, voiced). Places are ordered so that neighbors are 1 apart.
PLACE = {"bilabial": 0, "labiodental": 1, "dental": 2, "alveolar": 3,
         "postalveolar": 4, "retroflex": 4.5, "alveolopalatal": 4.5, "palatal": 5,
         "velar": 6, "uvular": 7, "pharyngeal": 8, "epiglottal": 8.5, "glottal": 9,
         "labiovelar": 6, "labiopalatal": 5}
CONSONANTS = {
    "p": ("bilabial", "stop", 0), "b": ("bilabial", "stop", 1),
    "t": ("alveolar", "stop", 0), "d": ("alveolar", "stop", 1),
    "ʈ": ("retroflex", "stop", 0), "ɖ": ("retroflex", "stop", 1),
    "c": ("palatal", "stop", 0), "ɟ": ("palatal", "stop", 1),
    "k": ("velar", "stop", 0), "g": ("velar", "stop", 1),
    "q": ("uvular", "stop", 0), "ɢ": ("uvular", "stop", 1),
    "ʡ": ("epiglottal", "stop", 0), "ʔ": ("glottal", "stop", 0),
    "ɓ": ("bilabial", "implosive", 1), "ɗ": ("alveolar", "implosive", 1),
    "ᶑ": ("retroflex", "implosive", 1), "ʄ": ("palatal", "implosive", 1),
    "ɠ": ("velar", "implosive", 1), "ʛ": ("uvular", "implosive", 1),
    "m": ("bilabial", "nasal", 1), "ɱ": ("labiodental", "nasal", 1),
    "n": ("alveolar", "nasal", 1), "ɳ": ("retroflex", "nasal", 1),
    "ɲ": ("palatal", "nasal", 1), "ŋ": ("velar", "nasal", 1), "ɴ": ("uvular", "nasal", 1),
    "ʙ": ("bilabial", "trill", 1), "r": ("alveolar", "trill", 1), "ʀ": ("uvular", "trill", 1),
    "ⱱ": ("labiodental", "tap", 1), "ɾ": ("alveolar", "tap", 1), "ɽ": ("retroflex", "tap", 1),
    "ɺ": ("alveolar", "lateral-tap", 1),
    "ɸ": ("bilabial", "fricative", 0), "β": ("bilabial", "fricative", 1),
    "f": ("labiodental", "fricative", 0), "v": ("labiodental", "fricative", 1),
    "θ": ("dental", "fricative", 0), "ð": ("dental", "fricative", 1),
    "s": ("alveolar", "fricative", 0), "z": ("alveolar", "fricative", 1),
    "ʃ": ("postalveolar", "fricative", 0), "ʒ": ("postalveolar", "fricative", 1),
    "ʂ": ("retroflex", "fricative", 0), "ʐ": ("retroflex", "fricative", 1),
    "ɕ": ("alveolopalatal", "fricative", 0), "ʑ": ("alveolopalatal", "fricative", 1),
    "ç": ("palatal", "fricative", 0), "ʝ": ("palatal", "fricative", 1),
    "x": ("velar", "fricative", 0), "ɣ": ("velar", "fricative", 1),
    "χ": ("uvular", "fricative", 0), "ʁ": ("uvular", "fricative", 1),
    "ħ": ("pharyngeal", "fricative", 0), "ʕ": ("pharyngeal", "fricative", 1),
    "ʜ": ("epiglottal", "fricative", 0), "ʢ": ("epiglottal", "fricative", 1),
    "h": ("glottal", "fricative", 0), "ɦ": ("glottal", "fricative", 1),
    "ɧ": ("velar", "fricative", 0), "ʍ": ("labiovelar", "fricative", 0),
    "ɬ": ("alveolar", "lateral-fricative", 0), "ɮ": ("alveolar", "lateral-fricative", 1),
    "ʋ": ("labiodental", "approximant", 1), "ɹ": ("alveolar", "approximant", 1),
    "ɻ": ("retroflex", "approximant", 1), "j": ("palatal", "approximant", 1),
    "ɰ": ("velar", "approximant", 1), "w": ("labiovelar", "approximant", 1),
    "ɥ": ("labiopalatal", "approximant", 1),
    "l": ("alveolar", "lateral", 1), "ɭ": ("retroflex", "lateral", 1),
    "ʎ": ("palatal", "lateral", 1), "ʟ": ("velar", "lateral", 1),
    "ʘ": ("bilabial", "click", 0), "ǀ": ("dental", "click", 0),
    "ǃ": ("alveolar", "click", 0), "ǂ": ("palatal", "click", 0),
    "ǁ": ("alveolar", "click", 0),
}
# Manners that sound alike: one manner step.
_MANNER_NEAR = [
    {"trill", "tap", "approximant"},     # the r sounds: r ɾ ɹ ʀ ʁ
    {"lateral", "lateral-tap"},
    {"stop", "implosive"},
    {"fricative", "lateral-fricative"},
]
# Pairs closer than features say.
SPECIAL_PAIRS = {
    ("r", "ɾ"): 0.15, ("r", "ɹ"): 0.15, ("ɾ", "ɹ"): 0.15, ("r", "ʀ"): 0.3, ("r", "ʁ"): 0.3,
    ("ʀ", "ʁ"): 0.15, ("ɹ", "ɻ"): 0.15, ("r", "ɻ"): 0.3, ("ɾ", "ɽ"): 0.15,
    ("l", "r"): 0.4, ("l", "ɾ"): 0.4, ("l", "ɹ"): 0.4, ("l", "ɫ"): 0.15,
    ("h", "ɦ"): 0.15, ("h", "x"): 0.3, ("h", "χ"): 0.4, ("h", "ħ"): 0.3,
    ("w", "ʍ"): 0.3, ("w", "β"): 0.3, ("w", "ʋ"): 0.3, ("w", "v"): 0.4, ("ʋ", "v"): 0.15,
    ("v", "β"): 0.15, ("f", "ɸ"): 0.15, ("ʍ", "x"): 0.4, ("ʍ", "h"): 0.4,
    ("j", "ʝ"): 0.15, ("j", "ʎ"): 0.3, ("j", "ɥ"): 0.3, ("ʝ", "ʒ"): 0.4,
    ("ʃ", "ɕ"): 0.15, ("ʃ", "ʂ"): 0.15, ("ʒ", "ʑ"): 0.15, ("ʒ", "ʐ"): 0.15,
    ("ç", "ɕ"): 0.3, ("ç", "ʃ"): 0.3, ("x", "χ"): 0.15, ("ɣ", "ʁ"): 0.3,
    ("k", "c"): 0.3, ("g", "ɟ"): 0.3, ("k", "q"): 0.15, ("g", "ɢ"): 0.15,
    ("ŋ", "ɴ"): 0.15, ("n", "ɲ"): 0.3, ("n", "ŋ"): 0.3, ("m", "ɱ"): 0.15,
    ("t", "ʈ"): 0.15, ("d", "ɖ"): 0.15, ("n", "ɳ"): 0.15, ("l", "ɭ"): 0.15,
    ("θ", "s"): 0.3, ("ð", "z"): 0.3, ("θ", "f"): 0.3, ("ð", "v"): 0.3,
    ("θ", "t"): 0.3, ("ð", "d"): 0.3, ("ɣ", "g"): 0.3, ("β", "b"): 0.3,
    ("ʔ", "h"): 0.4,
}
# Glides and the vowels they are made of.
GLIDE_VOWEL = {
    ("j", "i"): 0.15, ("j", "ɪ"): 0.15, ("j", "e"): 0.4, ("j", "y"): 0.3, ("j", "ɨ"): 0.3,
    ("w", "u"): 0.15, ("w", "ʊ"): 0.15, ("w", "o"): 0.4, ("w", "ʉ"): 0.3, ("w", "y"): 0.4,
    ("ɥ", "y"): 0.15, ("ɥ", "i"): 0.3, ("ɰ", "ɯ"): 0.15, ("ʋ", "u"): 0.4,
    ("ʝ", "i"): 0.4, ("ɹ", "ə"): 0.5, ("ɻ", "ə"): 0.5, ("h", "ə"): 0.7,
}
CONSONANT_VOWEL = 0.8
GLIDE_OTHER_VOWEL = 0.6


def consonant_distance(a: str, b: str) -> float:
    pair = SPECIAL_PAIRS.get((a, b), SPECIAL_PAIRS.get((b, a)))
    if pair is not None:
        return pair
    pa, ma, va = CONSONANTS[a]
    pb, mb, vb = CONSONANTS[b]
    place = abs(PLACE[pa] - PLACE[pb])
    if ma == mb:
        manner = 0
    elif any(ma in group and mb in group for group in _MANNER_NEAR):
        manner = 1
    else:
        manner = 2
    voice = int(va != vb)
    if manner == 0 and place == 0:
        return 0.3 if voice else 0.15  # p/b; same features otherwise (ǃ/ǁ)
    if manner == 0 and place <= 1:
        return 0.3 + 0.1 * voice                  # s/ʃ 0.3, s/ʒ 0.4
    if manner == 1 and place <= 1:
        return 0.3 + 0.1 * voice
    if manner == 2 and place <= 1:
        return 0.4 + 0.1 * voice                  # t/s 0.4, t/z 0.5
    if manner == 0 and place <= 3:
        return 0.4 + 0.1 * voice                  # t/k, n/m 0.4
    if place <= 2.5 and manner <= 1:
        return 0.4
    return min(0.6, 0.5 + 0.1 * voice * (manner == 2 and place > 2))  # k/f 0.5, k/v 0.6


# --------------------------------------------------------------- diacritics
# What a diacritic one side has and the other lacks adds.
DIACRITICS = {
    "ː": 0.15, "ˑ": 0.1, "̃": 0.15, "̯": 0.15, "̩": 0.15, "ʰ": 0.15, "ʱ": 0.15,
    "ʲ": 0.15, "ʷ": 0.15, "̥": 0.15, "̬": 0.1, "ˠ": 0.1, "ˤ": 0.1, "ʼ": 0.2,
    "˞": 0.15, "̪": 0.05, "̺": 0.05, "̚": 0.05, "̴": 0.1,
}
# A frame whose glyph is a lone diacritic, against a phoneme that has it,
# or against the sound the diacritic is a shadow of.
LONE_DIACRITIC_HIT = 0.3
LONE_DIACRITIC_SOUND = {"ʷ": {"w", "u"}, "ʲ": {"j", "i"}, "ʰ": {"h"}, "ʱ": {"h", "ɦ"},
                        "˞": {"r", "ɹ", "ɻ"}, "̃": {"n", "m", "ŋ"}}
LONE_DIACRITIC_MISS = 0.8


def split(token: str) -> tuple[str, Counter]:
    text = unicodedata.normalize("NFC", token.replace("ɡ", "g"))
    first, rest = text[:1], text[1:]
    if first not in VOWELS and first not in CONSONANTS:
        # A precomposed letter with its diacritic (ĩ → i + ̃); ç stays ç.
        parts = unicodedata.normalize("NFD", first)
        first, rest = parts[:1], parts[1:] + rest
    return first, Counter(unicodedata.normalize("NFD", rest))


def base_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in VOWELS and b in VOWELS:
        return vowel_distance(a, b)
    if a in CONSONANTS and b in CONSONANTS:
        return consonant_distance(a, b)
    if a in CONSONANTS and b in VOWELS:
        a, b = b, a
    if a in VOWELS and b in CONSONANTS:
        glide = GLIDE_VOWEL.get((b, a))
        if glide is not None:
            return glide
        _, manner, _ = CONSONANTS[b]
        return GLIDE_OTHER_VOWEL if manner == "approximant" else CONSONANT_VOWEL
    return 1.0


def is_lone_diacritic(token: str) -> bool:
    return conditioning.is_modifier(token)


def single_distance(a: str, b: str) -> float:
    """Distance between two single phones (a base letter plus diacritics)."""
    if is_lone_diacritic(a) or is_lone_diacritic(b):
        if is_lone_diacritic(b):
            a, b = b, a
        if is_lone_diacritic(b):
            return 0.0 if a == b else LONE_DIACRITIC_MISS
        mark = unicodedata.normalize("NFD", a)
        base_b, mods_b = split(b)
        if all(mods_b[m] for m in mark):
            return LONE_DIACRITIC_HIT
        if base_b in LONE_DIACRITIC_SOUND.get(a, ()):
            return LONE_DIACRITIC_HIT
        return LONE_DIACRITIC_MISS
    base_a, mods_a = split(a)
    base_b, mods_b = split(b)
    d = base_distance(base_a, base_b)
    unmatched = (mods_a - mods_b) + (mods_b - mods_a)
    d += sum(DIACRITICS.get(m, 0.15) * n for m, n in unmatched.items())
    return min(1.0, d)


def phones(token: str) -> list[str]:
    """A glyph cut into phones: "lj" → ["l", "j"], "kʷ" → ["kʷ"]."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", token):
        if out and (conditioning.is_modifier(ch) or unicodedata.combining(ch)):
            out[-1] += ch
        else:
            out.append(ch)
    return out


SEQUENCE_STEP = 0.15  # a glyph of two phones against one phoneme: per extra phone


def distance(a: str, b: str) -> float:
    pa, pb = phones(a), phones(b)
    if len(pa) == 1 and len(pb) == 1:
        return single_distance(pa[0], pb[0])
    # Two-phone glyph ("lj", "iu") against a phoneme: its closest phone, plus a step.
    best = min(single_distance(x, y) for x in pa for y in pb)
    return min(1.0, best + SEQUENCE_STEP * (abs(len(pa) - len(pb)) or 1))


# ----------------------------------------------------------------- universe


def universe() -> tuple[list[str], list[str]]:
    """(expected phonemes of every language, every glyph a frame may carry)."""
    from ancient_language_tts.check.phonemes import SPECIAL_TOKENS, ZIPA_TOKENS

    model = [t for t in ZIPA_TOKENS if t not in SPECIAL_TOKENS]
    expected: set[str] = set()
    glyphs: set[str] = set(model)
    for folder in sorted(DATA.iterdir()):
        if not (folder / "phonemes.json").is_file():
            continue
        expected |= set(ipa_tokens.phonemes(folder.name))
        lang = conditioning.language(folder.name)
        glyphs |= {lang.glyph(t) for t in model}
    glyphs -= {conditioning.BLANK, conditioning.DROP, ""}
    norm = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731
    return sorted(map(norm, expected)), sorted(map(norm, glyphs))


def check_known(symbols: list[str]) -> None:
    unknown = set()
    for s in symbols:
        for phone in phones(s):
            if is_lone_diacritic(phone):
                continue
            base, mods = split(phone)
            if base not in VOWELS and base not in CONSONANTS:
                unknown.add(base)
            unknown |= {m for m in mods if m not in DIACRITICS}
    if unknown:
        raise SystemExit(f"no features for: {' '.join(sorted(unknown))}")


def main() -> None:
    expected, glyphs = universe()
    columns = sorted(set(expected) | set(glyphs))
    check_known(columns)
    table: dict[str, dict[str, float]] = {}
    for e in expected:
        table[e] = {g: round(distance(e, g), 3) for g in columns if g != e}
    about = [line for line in (__doc__ or "").strip().splitlines()]
    OUT.write_text(_dump({"about": about, "distances": table}), encoding="utf-8")
    pairs = sum(len(row) for row in table.values())
    print(f"{OUT.relative_to(ROOT)}: {len(expected)} expected phonemes x {len(columns)} glyphs, "
          f"{pairs} pairs")


def _dump(data: dict) -> str:
    """JSON with one line per expected phoneme."""
    lines = ["{", '  "about": ' + json.dumps(data["about"], ensure_ascii=False, indent=4)
             .replace("\n", "\n  ") + ",", '  "distances": {']
    rows = list(data["distances"].items())
    for i, (e, row) in enumerate(rows):
        comma = "," if i < len(rows) - 1 else ""
        lines.append(f"    {json.dumps(e, ensure_ascii=False)}: "
                     f"{json.dumps(row, ensure_ascii=False)}{comma}")
    lines += ["  }", "}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
