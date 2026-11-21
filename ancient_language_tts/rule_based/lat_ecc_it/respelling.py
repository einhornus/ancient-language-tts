"""Ecclesiastical Latin IPA → Modern Italian respelling, by rule.

Implements `data/lat-ecc/respelling/it.txt`.
"""

from __future__ import annotations

import re
from pathlib import Path

# Longest-first IPA tokens.
_TOKENS = (
    "tʃ", "dʒ", "kʷ", "gʷ", "ɛu", "ɛi", "au", "ts", "dz", "ʃ", "ɲ", "ŋ", "ɱ",
    "ˈ", ".", ",", "!", "?",
    "a", "ɛ", "i", "ɔ", "u",
    "p", "b", "t", "d", "k", "g", "f", "s", "z", "m", "n", "r", "l", "j", "v",
)

_TOKEN_RE = re.compile("|".join(re.escape(t) for t in sorted(_TOKENS, key=len, reverse=True)))

_VOWELS = frozenset({"a", "ɛ", "i", "ɔ", "u", "au", "ɛu", "ɛi"})
_STRESSED = {
    "a": "à", "ɛ": "è", "i": "ì", "ɔ": "ò", "u": "ù",
    "au": "àu", "ɛu": "èu", "ɛi": "èi",
}
_PLAIN = {
    "a": "a", "ɛ": "e", "i": "i", "ɔ": "o", "u": "u",
    "au": "au", "ɛu": "eu", "ɛi": "ei",
}


def respell(ipa: str) -> str:
    """Respell an Ecclesiastical Latin IPA line into Italian orthography."""
    cleaned = ipa.strip()
    if not cleaned:
        return ""
    out: list[str] = []
    sentence_start = True
    raw_words = cleaned.split()
    for index, raw in enumerate(raw_words):
        word, pause = _strip_pause(raw)
        if word:
            spelled = _word(word)
            if not pause and index + 1 < len(raw_words):
                spelled += _boundary_double(word, raw_words[index + 1])
            if sentence_start and spelled:
                spelled = spelled[0].upper() + spelled[1:]
                sentence_start = False
            out.append(spelled)
        if pause:
            out.append(pause)
            if pause == ".":
                sentence_start = True
        if index + 1 < len(raw_words):
            out.append(" ")
    text = "".join(out)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ([,!?\.])", r"\1", text)
    return text.strip()


def _strip_pause(token: str) -> tuple[str, str]:
    """Split a trailing pause mark from an IPA token. Syllable dots stay."""
    if token and token[-1] in ",!?.":
        # A lone "." is a pause. A final "." after phones is a pause too.
        # Syllable dots never end a token after split() — they sit mid-word.
        return token[:-1], token[-1]
    return token, ""


# A word-final consonant before the same consonant opening the next word is
# written double ("quamm mìnimum"): with one letter on each side the voice
# merges the two into a single short consonant.
_BOUNDARY_DOUBLE = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "c", "g": "g", "f": "f",
    "m": "m", "n": "n", "r": "r", "l": "l", "s": "s",
}


def _boundary_double(ipa_word: str, next_raw: str) -> str:
    """The extra letter that doubles a final consonant shared with the next word."""
    this = [t for t in _tokenize(ipa_word) if t not in ("ˈ", ".")]
    following = [t for t in _tokenize(next_raw) if t not in ("ˈ", ".")]
    if len(this) < 2 or not following:
        return ""
    last, before = this[-1], this[-2]
    if last != following[0] or last not in _BOUNDARY_DOUBLE or before not in _VOWELS:
        return ""
    return _BOUNDARY_DOUBLE[last]


def _tokenize(ipa_word: str) -> list[str]:
    return _TOKEN_RE.findall(ipa_word)


def _word(ipa_word: str) -> str:
    tokens = _tokenize(ipa_word)
    syllables: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    current_stress = False
    for tok in tokens:
        if tok == "ˈ":
            if current:
                syllables.append((current_stress, current))
                current = []
            current_stress = True
            continue
        if tok == ".":
            syllables.append((current_stress, current))
            current = []
            current_stress = False
            continue
        current.append(tok)
    if current or current_stress:
        syllables.append((current_stress, current))

    marked: list[tuple[str, bool]] = []
    for is_stressed, segs in syllables:
        for seg in segs:
            marked.append((seg, bool(is_stressed and seg in _VOWELS)))
    return _spell_segments(marked)


def _spell_segments(marked: list[tuple[str, bool]]) -> str:
    out: list[str] = []
    i = 0
    n = len(marked)
    after_stress = False
    while i < n:
        tok, stress = marked[i]
        nxt = marked[i + 1][0] if i + 1 < n else ""
        nxt2 = marked[i + 2][0] if i + 2 < n else ""
        prev = marked[i - 1][0] if i else ""

        # Vowels
        if tok in _VOWELS:
            if stress:
                out.append(_STRESSED[tok])
                after_stress = True
            elif tok == "ɛ" and after_stress and nxt in _VOWELS:
                # Unstressed /ɛ/ in hiatus after the stress: è, or the voice
                # glides it to [j] (àlea → [ˈalja]).
                out.append("è")
            else:
                out.append(_PLAIN[tok])
            i += 1
            continue

        # /ɲ/ /ɲ.ɲ/ → gn (one)
        if tok == "ɲ":
            if nxt == "ɲ":
                i += 2
            else:
                i += 1
            out.append("gn")
            continue

        # /ŋ/ before k/g, /ɱ/ before f/v → n
        if tok in ("ŋ", "ɱ"):
            out.append("n")
            i += 1
            continue

        # /k/ before /ʃ/ → k
        if tok == "k" and nxt == "ʃ":
            out.append("k")
            i += 1
            continue

        # /ks/ or /k/+/s/ → x
        if tok == "k" and nxt == "s":
            out.append("x")
            i += 2
            continue

        # /s/+/k/ before e/i → sch
        if tok == "s" and nxt == "k" and _front_vowel(nxt2):
            out.append("sch")
            i += 2
            continue

        # /tʃ/
        if tok == "tʃ":
            if _front_vowel(nxt):
                out.append("c")
            else:
                out.append("ci")
            i += 1
            continue

        # /dʒ/
        if tok == "dʒ":
            if _front_vowel(nxt):
                out.append("g")
            else:
                out.append("gi")
            i += 1
            continue

        # Italian long /ʃ.ʃ/ /t.ts/ /d.dz/ → one sc / z: the voice makes
        # them long between vowels by itself.
        if (tok, nxt) in (("ʃ", "ʃ"), ("t", "ts"), ("d", "dz")):
            i += 1
            continue

        # /ʃ/
        if tok == "ʃ":
            if _front_vowel(nxt):
                out.append("sc")
            else:
                out.append("sci")
            i += 1
            continue

        # /k/
        if tok == "k":
            if _front_vowel(nxt):
                out.append("ch")
            else:
                out.append("c")
            i += 1
            continue

        # /g/ + /l/ + /i/ → ghli
        if tok == "g" and nxt == "l" and nxt2 == "i":
            out.append("ghli")
            i += 3
            continue

        # /g/
        if tok == "g":
            if _front_vowel(nxt):
                out.append("gh")
            else:
                out.append("g")
            i += 1
            continue

        # /ts/ /dz/ → z
        if tok in ("ts", "dz"):
            out.append("z")
            i += 1
            continue

        # /z/ → s (voiced s of ex-)
        if tok == "z":
            out.append("s")
            i += 1
            continue

        # /kʷ/ /gʷ/
        if tok == "kʷ":
            out.append("qu")
            i += 1
            continue
        if tok == "gʷ":
            out.append("gu")
            i += 1
            continue

        # /j/ → i
        if tok == "j":
            out.append("i")
            i += 1
            continue

        # /s/ between vowels → ss
        if tok == "s":
            if _is_vowel(prev) and _is_vowel(nxt):
                out.append("ss")
            else:
                out.append("s")
            i += 1
            continue

        # Geminate: same consonant twice → doubled letter (with soft digraphs)
        if tok == nxt and tok not in _VOWELS and tok not in ("ɲ", "ŋ", "j"):
            # t+tʃ already handled as separate; k+k → cc / cch
            if tok == "k":
                if _front_vowel(nxt2):
                    out.append("cch")
                else:
                    out.append("cc")
                i += 2
                continue
            if tok == "t" and False:
                pass
            doubled = {
                "p": "pp", "b": "bb", "t": "tt", "d": "dd", "f": "ff",
                "m": "mm", "n": "nn", "r": "rr", "l": "ll", "v": "vv",
                "s": "ss", "g": "gg",
            }
            if tok in doubled:
                out.append(doubled[tok])
                i += 2
                continue

        # t before tʃ (ecce): /t/+/tʃ/ → cc before e/i
        if tok == "t" and nxt == "tʃ":
            if _front_vowel(nxt2):
                out.append("cc")
            else:
                out.append("cci")
            i += 2
            continue

        # d before dʒ
        if tok == "d" and nxt == "dʒ":
            if _front_vowel(nxt2):
                out.append("gg")
            else:
                out.append("ggi")
            i += 2
            continue

        # Simple letters
        if tok in "pbtdfmnlrv":
            out.append(tok)
            i += 1
            continue

        # Fallback: skip unknown
        i += 1

    return "".join(out)


def _front_vowel(tok: str) -> bool:
    return tok in ("ɛ", "i", "ɛu", "ɛi") or (tok.startswith("ɛ") or tok == "i")


def _is_vowel(tok: str) -> bool:
    return tok in _VOWELS


def _load_examples() -> list[tuple[str, str, str]]:
    path = Path(__file__).resolve().parents[2] / "data" / "lat-ecc" / "respelling" / "it.examples.txt"
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.count("→") < 2:
            continue
        left, mid, right = [p.strip() for p in line.split("→", 2)]
        rows.append((left, mid, right))
    return rows


if __name__ == "__main__":
    bad = 0
    rows = _load_examples()
    for _src, ipa, want in rows:
        got = respell(ipa)
        # Examples are lowercase teaching lines; output capitalizes sentences.
        if got.lower() != want.lower():
            bad += 1
            print("FAIL")
            print(" ipa:", ipa)
            print("want:", want)
            print(" got:", got)
            print()
    print(f"{len(rows) - bad}/{len(rows)} matched")
