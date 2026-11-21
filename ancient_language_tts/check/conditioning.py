"""Condition the phone recognizer on one ancient language.

ZIPA writes IPA one character at a time from its own vocabulary
(`data/zipa_tokens.json`). A language's phonology uses a smaller set of
phones (`data/<code>/phonemes.json`), and `data/<code>/zipa.json` says how
the model's tokens map onto it:

    {"e": ["ɛ"], "θ": ["tʰ"], "ʎ": ["l", "j"], "h": [], ...}

Each record turns one model token into a sequence of that language's
tokens (or of modifiers such as "ː" that join the phone before them). An
empty list drops the token: a dropped letter counts as silence, a dropped
modifier is just left off. A token with no record stands for itself.

With that, recognition is conditioned on the language:

* every model token is renamed to what the rules make of it, and tokens
  that end up the same sound pool their probability,
* a token that still is no sound of the language is never chosen; the
  frame goes to the best token that is,
* each heard phone (a letter plus the modifiers after it) is cut into the
  language's tokens by longest match. A letter whose language only has it
  with a modifier is completed (Proto-Germanic "ɛ" → "ɛː"), and a modifier
  that fits no token of the language is dropped.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .. import ipa_tokens, languages

_DATA = Path(__file__).resolve().parent.parent / "data"

BLANK = "•"
DROP = "∅"  # a model token that is no sound of the language


def model_tokens() -> list[dict]:
    """ZIPA's vocabulary, from data/zipa_tokens.json: id, token, type, name."""
    return list(_model_tokens())


@lru_cache(maxsize=1)
def _model_tokens() -> tuple[dict, ...]:
    rows = json.loads((_DATA / "zipa_tokens.json").read_text(encoding="utf-8"))
    return tuple(sorted(rows, key=lambda row: row["id"]))


def is_modifier(text: str) -> bool:
    """True when `text` is only diacritics/modifier letters (ʰ, ː, ˞, ̃ …)."""
    return bool(text) and all(
        unicodedata.category(ch) in ("Mn", "Lm", "Sk") for ch in unicodedata.normalize("NFD", text)
    )


@dataclass(frozen=True)
class Language:
    code: str
    inventory: frozenset[str]
    rules: dict[str, tuple[str, ...]]

    @property
    def chars(self) -> frozenset[str]:
        return frozenset(ch for item in self.inventory for ch in unicodedata.normalize("NFD", item))

    def convert(self, token: str) -> str:
        """The written form the rules give one model token ("" when dropped)."""
        return "".join(self.rules.get(token, (token,)))

    def glyph(self, token: str) -> str:
        """What one model token is in this language: a sound, `BLANK` or `DROP`."""
        out = self.convert(token)
        if not out:
            return BLANK
        if is_modifier(out):
            return out if set(unicodedata.normalize("NFD", out)) <= self.chars else BLANK
        tokens = self.tokenize(out)
        return out if tokens and all(t in self.inventory for t in tokens) else DROP

    def tokenize(self, phone: str) -> list[str]:
        """Cut a heard phone into this language's tokens (see the module docstring)."""
        text = unicodedata.normalize("NFC", phone)
        longest = max(len(item) for item in self.inventory)
        out: list[str] = []
        i = 0
        while i < len(text):
            for size in range(min(longest, len(text) - i), 0, -1):
                piece = text[i : i + size]
                if piece in self.inventory:
                    out.append(piece)
                    i += size
                    break
            else:
                ch = text[i]
                i += 1
                if is_modifier(ch):
                    continue  # fits no token here: left off
                completions = sorted(
                    (item for item in self.inventory if item.startswith(ch)), key=len
                )
                out.append(completions[0] if completions else ch)
        return out


def language(source: str) -> Language:
    """The conditioning data of the ancient language `source`."""
    return _language(languages.ancient(source).code)


@lru_cache(maxsize=None)
def _language(code: str) -> Language:
    path = _DATA / code / "zipa.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    rules = {
        unicodedata.normalize("NFC", key): tuple(unicodedata.normalize("NFC", t) for t in value)
        for key, value in raw.items()
    }
    return Language(code, frozenset(ipa_tokens.phonemes(code)), rules)
