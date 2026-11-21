"""Old English spelling → Icelandic respelling, with no IPA string.

The spelling analysis (`ipa.analyse`: sounds, syllables, stress) feeds the
Icelandic word speller of `respell_is` directly, so the result is the same
as `respell_is.respell(to_ipa(text))` without building or parsing IPA.
"""

from __future__ import annotations

from .ipa import analyse
from .respell_is import spell_items


def to_icelandic(text: str) -> str:
    """Respell Old English `text` as Icelandic."""
    return spell_items(analyse(text))
