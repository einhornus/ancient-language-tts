"""Rule-based Proto-Slavic → IPA and → Ukrainian.

`to_ipa` applies `data/sla-pro/phonology.txt`.
`ipa_to_ukrainian` applies `data/sla-pro/respelling/uk.txt` to that IPA.
`to_ukrainian` gives the same Ukrainian straight from the spelling.
"""

from .direct_uk import to_ukrainian
from .ipa import to_ipa
from .respell_uk import respell as ipa_to_ukrainian

SOURCE = "sla-pro"
TARGETS = ("uk",)

__all__ = ["SOURCE", "TARGETS", "ipa_to_ukrainian", "to_ipa", "to_ukrainian"]
