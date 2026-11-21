"""Shared pieces of the Middle English engine: word structure, IPA parsing, rendering.

`ipa.analyze` turns spelling into a list of `Word` / `Pause` items. `parse_ipa`
turns an IPA line into the same items, so a respeller can work from either
without going through the other.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

VOWELS = frozenset({
    "a", "aː", "ɛ", "ɛː", "eː", "ɪ", "iː", "ɔ", "ɔː", "oː", "ʊ", "uː", "ə",
    "ai", "au", "ɔu", "iu", "ɛu", "ɔi",
})

_PHONES = tuple(VOWELS) + (
    "tʃ", "dʒ", "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ",
    "ç", "x", "h", "m", "n", "ŋ", "l", "r", "j", "w", "ʍ",
)
_PHONE_RE = re.compile("|".join(re.escape(p) for p in sorted(_PHONES, key=len, reverse=True)))

PAUSES = frozenset(",!?.")


@dataclass(frozen=True)
class Syllable:
    phones: tuple[str, ...]
    stressed: bool = False


@dataclass(frozen=True)
class Word:
    syllables: tuple[Syllable, ...]

    def stress_index(self) -> int | None:
        for i, syl in enumerate(self.syllables):
            if syl.stressed:
                return i
        return None


@dataclass(frozen=True)
class Pause:
    mark: str


Item = Union[Word, Pause]


def parse_ipa(ipa: str) -> list[Item]:
    """Split an IPA line into words and pause marks."""
    items: list[Item] = []
    for raw in ipa.split():
        mark = ""
        if raw[-1] in PAUSES:
            raw, mark = raw[:-1], raw[-1]
        if raw:
            word = parse_word(raw)
            if word.syllables:
                items.append(word)
        if mark:
            items.append(Pause(mark))
    return items


def parse_word(text: str) -> Word:
    syllables: list[Syllable] = []
    current: list[str] = []
    stressed = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "ˈ.":
            if current:
                syllables.append(Syllable(tuple(current), stressed))
            current = []
            stressed = ch == "ˈ"
            i += 1
            continue
        match = _PHONE_RE.match(text, i)
        if match:
            current.append(match.group())
            i = match.end()
        else:
            i += 1
    if current:
        syllables.append(Syllable(tuple(current), stressed))
    return Word(tuple(syllables))


def format_word(word: Word) -> str:
    out: list[str] = []
    for i, syl in enumerate(word.syllables):
        if syl.stressed:
            out.append("ˈ")
        elif i:
            out.append(".")
        out.append("".join(syl.phones))
    return "".join(out)


def format_ipa(items: list[Item]) -> str:
    return render(items, format_word, capitalize=False)


def render(items: list[Item], spell: Callable[[Word], str], capitalize: bool = True) -> str:
    """Join spelled words and pause marks; capitalize each sentence start."""
    out: list[str] = []
    sentence = True
    for item in items:
        if isinstance(item, Pause):
            if out and out[-1] == " ":
                out.pop()
            if not out:
                continue
            out += [item.mark, " "]
            if item.mark in ".!?":
                sentence = True
            continue
        text = spell(item)
        if not text:
            continue
        if capitalize and sentence:
            text = text[0].upper() + text[1:]
        sentence = False
        out += [text, " "]
    return "".join(out).strip()


def flat(word: Word) -> list[tuple[str, int]]:
    """Every phone of a word with the index of its syllable."""
    return [(ph, si) for si, syl in enumerate(word.syllables) for ph in syl.phones]
