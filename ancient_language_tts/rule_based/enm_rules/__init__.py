"""Rule-based Middle English → IPA, Modern English and Modern Dutch.

`to_ipa` applies `data/enm/phonology.txt`.
`ipa_to_english` / `ipa_to_dutch` (modules `respell_en` / `respell_nl`) apply
`data/enm/respelling/{en,nl}.txt` to IPA.
`to_english` / `to_dutch` respell the spelling directly, with no IPA string.
"""

from .direct_en import to_english
from .direct_nl import to_dutch
from .ipa import to_ipa
from .respell_en import respell as ipa_to_english
from .respell_nl import respell as ipa_to_dutch

SOURCE = "enm"
TARGETS = ("en", "nl")

__all__ = ["SOURCE", "TARGETS", "ipa_to_dutch", "ipa_to_english", "to_dutch", "to_english", "to_ipa"]
