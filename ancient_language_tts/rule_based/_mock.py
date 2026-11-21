"""Building blocks for the placeholder rule engines.

A mock engine is two lookup tables. `graphemes` maps source spelling to
IPA; `sounds` maps IPA to one modern spelling. Both are applied longest
key first, and anything not in a table passes through unchanged. Accents
and breathings are dropped before the lookup, so a mock line carries no
stress. Replace a mock module with a real engine (see `lat_ecc_it`) when
the language's rules are written.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping

from .base import Convert, RuleEngine

# Combining marks that only mark accent, tone, or breathing. Macron, caron,
# ogonek, ring below, and the like stay: they change the letter.
_ACCENTS = frozenset({
    "̀",  # grave
    "́",  # acute
    "̂",  # circumflex
    "̏",  # double grave
    "̑",  # inverted breve
    "̓",  # Greek psili (smooth breathing)
    "̔",  # Greek dasia (rough breathing)
    "͂",  # Greek perispomeni
    "ͅ",  # Greek ypogegrammeni (iota subscript)
})

# IPA marks a mock respelling drops when its table does not name them.
_IPA_DROP = str.maketrans("", "", "ˈˌːˑ̯̩̥")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if ch not in _ACCENTS)
    return unicodedata.normalize("NFC", kept)


def _table(mapping: Mapping[str, str]) -> Callable[[str], str]:
    keys = sorted(mapping, key=len, reverse=True)
    if not keys:
        return lambda text: text
    pattern = re.compile("|".join(re.escape(key) for key in keys))
    return lambda text: pattern.sub(lambda m: mapping[m.group(0)], text)


def _syllable_dots(ipa: str) -> str:
    """Drop the syllable dots inside a word, keep a sentence period."""
    return re.sub(r"\.(?=\S)", "", ipa)


def to_ipa(graphemes: Mapping[str, str], *, strip: str = "*") -> Convert:
    """Source spelling → IPA by table lookup."""
    lookup = _table(graphemes)

    def convert(text: str) -> str:
        cleaned = _strip_accents(text.strip().lower())
        cleaned = cleaned.translate(str.maketrans("", "", strip))
        return lookup(cleaned)

    return convert


def respell_ipa(
    sounds: Mapping[str, str],
    *,
    finish: Callable[[str], str] | None = None,
) -> Convert:
    """IPA → modern spelling by table lookup."""
    lookup = _table(sounds)

    def convert(ipa: str) -> str:
        line = lookup(_syllable_dots(ipa.strip())).translate(_IPA_DROP)
        line = unicodedata.normalize("NFC", line)
        if finish is not None:
            line = finish(line)
        return line[:1].upper() + line[1:]

    return convert


def engine(
    source: str,
    graphemes: Mapping[str, str],
    targets: Mapping[str, Mapping[str, str]],
    *,
    strip: str = "*",
    finish: Mapping[str, Callable[[str], str]] | None = None,
) -> RuleEngine:
    """A mock engine for `source`, respelling into each of `targets`.

    The text → respelling conversion is the two tables chained: the mock
    has no direct spelling rules of its own.
    """
    ipa = to_ipa(graphemes, strip=strip)
    from_ipa = {
        code: respell_ipa(table, finish=(finish or {}).get(code))
        for code, table in targets.items()
    }
    direct = {code: _chain(ipa, step) for code, step in from_ipa.items()}
    return RuleEngine(source, ipa, direct, from_ipa, mock=True)


def _chain(first: Convert, second: Convert) -> Convert:
    return lambda text: second(first(text))
