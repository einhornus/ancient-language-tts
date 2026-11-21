"""IPA as a list of tokens.

Everywhere in the package, IPA is a `list[str]`. Each item is one allowed
token of the source language:

- a phoneme from `data/<code>/phonemes.json`: one phone, with its
  diacritics and length ("pʰ", "aː", "i̯"). Diphthongs and affricates are
  two tokens ("a" "u̯", "t" "ʃ"), as the phone recognizer in `check` hears them,
- or a special token: the stress marks ˈ and ˌ, the syllable dot, a space
  between words, a line break, or one of `PUNCTUATION`.

The syllable dot and the sentence period are the same character, so both
are the token ".". A "." followed by a space, a line break, another mark,
or the end of the line is a period; one between phonemes is a syllable
break.

`parse_with_penalty` turns a written line into tokens and keeps each
character that is not an allowed token as a token of its own, reading on
after it, so allowed and unknown tokens may alternate. It also returns the
penalty: the share of symbol tokens (all but the special ones) that are
unknown. `parse` is the strict form: it raises `IpaParseError` when any
part of the line is not an allowed token. `render` joins tokens back into
the written line (`"".join`). `ensure` checks a token list; `unknown` and
`penalty` measure one; `drop_unknown` removes the unknown tokens.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from . import languages
from .errors import TTSError

_DATA = Path(__file__).resolve().parent / "data"

STRESS = "ˈ"
SECONDARY_STRESS = "ˌ"
SYLLABLE = "."
SPACE = " "
LINE_BREAK = "\n"

# Pause and quotation marks a line may keep. "." is listed once: it is
# both the syllable dot and the period.
PUNCTUATION: tuple[str, ...] = (
    ".", ",", "?", "!", ";", ":", "…",
    "-", "–", "—",
    "(", ")", '"', "«", "»", "“", "”", "„", "‘", "’",
    "¿", "¡", "|", "‖",
)

SPECIAL: tuple[str, ...] = (STRESS, SECONDARY_STRESS, SPACE, LINE_BREAK) + PUNCTUATION

# A symbol a model may write in place of the one an inventory uses. It is
# read as the other one only when the inventory never uses it.
_VARIANTS = {
    "ɡ": "g",  # U+0261 script g
    "g": "ɡ",
    "'": STRESS,
    "ʼ": STRESS,
}
# Tie bars (t͡ʃ, t͜ʃ) and the linking undertie. Dropped unless the inventory spells one.
_TIE_BARS = "͜͡‿"


class IpaParseError(TTSError):
    """A line that is not made only of allowed IPA tokens."""


def phonemes(source: str) -> tuple[str, ...]:
    """The phoneme inventory of `source`, from data/<code>/phonemes.json."""
    return _phonemes(languages.ancient(source).code)


@lru_cache(maxsize=None)
def _phonemes(code: str) -> tuple[str, ...]:
    path = _DATA / code / "phonemes.json"
    if not path.is_file():
        raise ValueError(f"missing phoneme inventory data/{code}/phonemes.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) and item for item in data):
        raise ValueError(f"data/{code}/phonemes.json must be a list of non-empty strings")
    items = tuple(dict.fromkeys(unicodedata.normalize("NFC", item) for item in data))
    clash = [item for item in items if item in SPECIAL]
    if clash:
        raise ValueError(f"data/{code}/phonemes.json lists special tokens: {clash}")
    return items


def allowed(source: str) -> frozenset[str]:
    """Every token a line of `source` may hold."""
    return _allowed(languages.ancient(source).code)


@lru_cache(maxsize=None)
def _allowed(code: str) -> frozenset[str]:
    return frozenset(_phonemes(code)) | frozenset(SPECIAL)


def render(tokens: Sequence[str]) -> str:
    """The written line: the tokens joined with nothing between them."""
    return "".join(tokens)


def parse(text: str, source: str) -> list[str]:
    """Cut a written IPA line into allowed tokens of `source`, strictly.

    Reads the line as `parse_with_penalty` does, then raises
    `IpaParseError` when any character is not an allowed token.
    """
    code = languages.ancient(source).code
    tokens, _ = parse_with_penalty(text, code)
    allowed_set = _allowed(code)
    line = render(tokens)
    position = 0
    for token in tokens:
        if token not in allowed_set:
            raise IpaParseError(_unknown(line, position, code))
        position += len(token)
    return tokens


def parse_with_penalty(text: str, source: str) -> tuple[list[str], float]:
    """Cut a written IPA line into tokens of `source`, keeping unknown symbols.

    Strips one pair of /…/ or […] brackets and leading asterisks, folds runs
    of whitespace into one space (or one line break when the run holds one),
    and reads the rest by longest match against the allowed tokens. A
    character that starts no allowed token becomes a token of its own, and
    reading goes on after it, so allowed and unknown tokens may alternate.

    Returns the tokens and the penalty (see `penalty`), from 0.0 to 1.0.
    Raises `IpaParseError` when the line is empty or holds no allowed
    phoneme at all.
    """
    code = languages.ancient(source).code
    line = _normalize(text)
    if not line:
        raise IpaParseError("the IPA line is empty")
    tokens = _longest_match(line, code)
    allowed_set = _allowed(code)
    if not any(token in allowed_set and token not in SPECIAL for token in tokens):
        raise IpaParseError(f"the IPA line has no phonemes: {line!r}")
    return tokens, penalty(tokens, code)


def unknown(tokens: Sequence[str], source: str) -> list[str]:
    """The tokens that are not allowed tokens of `source`, in order."""
    allowed_set = _allowed(languages.ancient(source).code)
    return [token for token in tokens if token not in allowed_set]


def penalty(tokens: Sequence[str], source: str) -> float:
    """The share of symbol tokens (all but the special ones) that are not allowed."""
    symbols = [token for token in tokens if token not in SPECIAL]
    if not symbols:
        return 0.0
    return len(unknown(symbols, source)) / len(symbols)


def drop_unknown(tokens: Sequence[str], source: str) -> list[str]:
    """`tokens` without the ones that are not allowed tokens of `source`."""
    allowed_set = _allowed(languages.ancient(source).code)
    return [token for token in tokens if token in allowed_set]


def ensure(ipa: Sequence[str], source: str, *, allow_unknown: bool = False) -> list[str]:
    """Check that `ipa` is a token list of `source` and return it as a list.

    A plain string is refused, since a string is a sequence of characters,
    not of tokens. Use `parse` for a written line. With `allow_unknown`,
    tokens that are not allowed (as `parse_with_penalty` keeps them) pass.
    """
    if isinstance(ipa, str):
        raise TypeError("IPA is a list of tokens, not a string. Use ipa_tokens.parse() for a written line.")
    tokens = list(ipa)
    if not tokens:
        raise ValueError("ipa is empty")
    code = languages.ancient(source).code
    if allow_unknown:
        return tokens
    allowed_set = _allowed(code)
    bad = [token for token in tokens if token not in allowed_set]
    if bad:
        raise IpaParseError(f"not allowed {code} IPA tokens: {bad!r}")
    return tokens


def _normalize(text: str) -> str:
    line = unicodedata.normalize("NFC", text).strip()
    if len(line) >= 2 and line[0] in "/[" and line[-1] in "/]":
        inner = line[1:-1].strip()
        if inner:
            line = inner
    line = line.lstrip("*").strip()
    out: list[str] = []
    run = ""
    for ch in line:
        if ch.isspace():
            run += ch
            continue
        if run:
            out.append(LINE_BREAK if ("\n" in run or "\r" in run) else SPACE)
            run = ""
        out.append(ch)
    return "".join(out)


def _longest_match(line: str, code: str) -> list[str]:
    allowed_set = _allowed(code)
    symbols = {ch for item in allowed_set for ch in item}
    line = "".join(
        _VARIANTS[ch] if ch not in symbols and _VARIANTS.get(ch) in symbols else ch
        for ch in line
        if ch in symbols or ch not in _TIE_BARS
    )
    longest = max(len(item) for item in allowed_set)
    tokens: list[str] = []
    i = 0
    while i < len(line):
        for size in range(min(longest, len(line) - i), 0, -1):
            piece = line[i : i + size]
            if piece in allowed_set:
                tokens.append(piece)
                i += size
                break
        else:
            tokens.append(line[i])  # not an allowed token: kept on its own
            i += 1
    return tokens


def _unknown(line: str, i: int, code: str) -> str:
    ch = line[i]
    name = unicodedata.name(ch, f"U+{ord(ch):04X}")
    context = line[max(0, i - 12) : i + 12]
    return f"{ch!r} ({name}) at position {i} is not an allowed {code} IPA token (…{context}…)"
