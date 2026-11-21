"""Rule-based Proto-Indo-European → IPA and → Lithuanian.

`to_ipa` applies `data/ine-pro/phonology.txt`.
`ipa_to_lithuanian` applies `data/ine-pro/respelling/lt.txt` to that IPA.
`to_lithuanian` gives the same Lithuanian straight from the spelling.
"""

from .direct_lt import to_lithuanian
from .ipa import to_ipa
from .respell_lt import respell as ipa_to_lithuanian

SOURCE = "ine-pro"

__all__ = ["SOURCE", "ipa_to_lithuanian", "to_ipa", "to_lithuanian"]
