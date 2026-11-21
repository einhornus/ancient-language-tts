"""Rule-based Old English (Late West Saxon) → IPA, Icelandic and Dutch.

`to_ipa` applies `data/ang/phonology.txt`. `ipa_to_icelandic` / `ipa_to_dutch` apply
`data/ang/respelling/{is,nl}.txt` to IPA. `to_icelandic` / `to_dutch` go from
the Old English spelling straight to the respelling, with no IPA string.
"""

from .direct_is import to_icelandic
from .direct_nl import to_dutch
from .ipa import to_ipa
from .respell_is import respell as ipa_to_icelandic
from .respell_nl import respell as ipa_to_dutch

SOURCE = "ang"

__all__ = ["SOURCE", "ipa_to_dutch", "ipa_to_icelandic", "to_dutch", "to_icelandic", "to_ipa"]
