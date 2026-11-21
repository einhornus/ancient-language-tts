"""Classical Latin spelling → Modern Italian respelling, with no IPA step.

The Latin spelling is analysed into syllables and sounds (`spelling`) and
spelled straight into Italian by the same rules as `respell_it`
(`data/lat/respelling/it.txt`). No IPA text is written or parsed, so the
result equals `respell_it(to_ipa(text))` by construction.
"""

from __future__ import annotations

from .respell_it import spell_items
from .spelling import analyse


def direct_it(text: str) -> str:
    """Respell Classical Latin `text` as Italian, straight from the spelling."""
    return spell_items(analyse(text))
