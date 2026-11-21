"""Rule-based Proto-Germanic → IPA, Icelandic, and Dutch.

`to_ipa` applies `data/gem-pro/phonology.txt`. `respell_ipa_is` / `respell_ipa_nl`
apply `data/gem-pro/respelling/{is,nl}.txt` to that IPA. `to_icelandic` /
`to_dutch` give the same respellings straight from the spelling.
"""

from .direct_is import to_icelandic
from .direct_nl import to_dutch
from .ipa import to_ipa
from .respell_is import respell as respell_ipa_is
from .respell_nl import respell as respell_ipa_nl

SOURCE = "gem-pro"
TARGETS = ("is", "nl")

__all__ = ["SOURCE", "TARGETS", "respell_ipa_is", "respell_ipa_nl", "to_dutch", "to_icelandic", "to_ipa"]
