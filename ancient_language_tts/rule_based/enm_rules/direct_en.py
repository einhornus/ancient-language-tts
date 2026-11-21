"""Middle English spelling → Modern English respelling, with no IPA string.

The spelling is analysed into syllables of sounds (`ipa.analyze`, the same
letter rules `to_ipa` uses), and those are spelled directly by the rules of
`respell_en`. No IPA text is produced or parsed, so the result is always the
same as `respell_en.respell(to_ipa(text))`.
"""

from __future__ import annotations

from .ipa import analyze
from .respell_en import spell_items


def to_english(text: str) -> str:
    """Respell Middle English `text` for an English reader."""
    return spell_items(analyze(text))
