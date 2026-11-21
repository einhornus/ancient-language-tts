"""Old English spelling → Dutch respelling, with no IPA string.

The spelling analysis (`ipa.analyse`: sounds, syllables, stress) feeds the
Dutch word speller of `respell_nl` directly, so the result is the same as
`respell_nl.respell(to_ipa(text))` without building or parsing IPA.
"""

from __future__ import annotations

from .ipa import analyse
from .respell_nl import spell_items


def to_dutch(text: str) -> str:
    """Respell Old English `text` as Dutch."""
    return spell_items(analyse(text))
