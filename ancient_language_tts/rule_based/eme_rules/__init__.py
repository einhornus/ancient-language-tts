"""Rule-based Early Modern English (c. 1600, Original Pronunciation).

`to_ipa` applies `data/eme/phonology.txt`.
`respell` applies `data/eme/respelling/en.txt` to that IPA.
`to_english` gives the same English respelling straight from the spelling.
"""

from .direct_en import to_english
from .ipa import to_ipa
from .respell_en import respell

SOURCE = "eme"

__all__ = ["SOURCE", "respell", "to_english", "to_ipa"]
