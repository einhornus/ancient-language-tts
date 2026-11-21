"""Middle English spelling → Modern Dutch respelling, with no IPA string.

The spelling is analysed into syllables of sounds (`ipa.analyze`, the same
letter rules `to_ipa` uses), and those are spelled directly by the rules of
`respell_nl`. No IPA text is produced or parsed, so the result is always the
same as `respell_nl.respell(to_ipa(text))`.
"""

from __future__ import annotations

from .ipa import analyze
from .respell_nl import spell_items


def to_dutch(text: str) -> str:
    """Respell Middle English `text` for a Dutch reader."""
    return spell_items(analyze(text))
