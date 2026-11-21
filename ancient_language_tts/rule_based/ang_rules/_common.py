"""Shared plumbing for the Old English respellers.

A line is a list of items: ("w", syllables, stress) for a word, where
syllables is a list of phone lists and stress the index of the ˈ syllable
(or None), and ("p", mark) for a pause. `parse_ipa` builds that from an IPA
string; `ipa.analyse` builds it straight from the Old English spelling.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_PHONE = re.compile(
    "ˈ|\\.|æːɑ̯|eːo̯|æɑ̯|eo̯|tʃ|dʒ|[lrn]̥|[ɑæeiouy]ː"
    "|[^\\s.ˈ][̀-ͯː]*"
)
_PAUSES = ",!?.;:"


def is_vowel(phone: str) -> bool:
    return bool(phone) and phone[0] in "ɑæeiouy"


def parse_ipa(ipa: str) -> list[tuple]:
    items: list[tuple] = []
    for token in ipa.split():
        marks: list[str] = []
        while token and token[-1] in _PAUSES:
            marks.insert(0, "." if token[-1] in ";:" else token[-1])
            token = token[:-1]
        if token:
            sylls: list[list[str]] = [[]]
            stress: int | None = None
            for ph in _PHONE.findall(token):
                if ph in ("ˈ", "."):
                    if sylls[-1]:
                        sylls.append([])
                    if ph == "ˈ":
                        stress = len(sylls) - 1
                else:
                    sylls[-1].append(ph)
            if not sylls[-1]:
                sylls.pop()
            if sylls:
                if stress is not None and stress >= len(sylls):
                    stress = None
                items.append(("w", sylls, stress))
        items.extend(("p", m) for m in marks)
    return items


def render(items: list[tuple], spell: Callable[[list[list[str]], "int | None"], str]) -> str:
    """Spell each word, keep pauses, capitalize each sentence start."""
    out: list[str] = []
    start = True
    for item in items:
        if item[0] == "p":
            if out and out[-1] == " ":
                out.pop()
            out += [item[1], " "]
            if item[1] in ".!?":
                start = True
            continue
        word = spell(item[1], item[2])
        if not word:
            continue
        if start:
            word = word[0].upper() + word[1:]
            start = False
        out += [word, " "]
    return "".join(out).strip()
