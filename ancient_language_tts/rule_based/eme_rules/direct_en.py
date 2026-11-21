"""Early Modern English spelling → modern English respelling, in one pass.

No IPA string is built or parsed: the spelling analysis of `ipa.analyze`
(lexical sets, syllables, verse) is spelled straight into English letters
with the table of `respelling/en.txt`.
"""

from __future__ import annotations

from .ipa import Word, analyze
from .respell_en import join, spell_syllables


def to_english(text: str) -> str:
    """Respell Early Modern English `text` for a modern English voice."""
    pieces: list[str] = []
    for item in analyze(text):
        if isinstance(item, Word):
            pieces.append(spell_syllables([syms for _stress, syms in item.syllables()]))
        else:
            pieces.append(item)
    return join(pieces)
