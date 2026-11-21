"""Rule-based Classical Latin (restored pronunciation).

`to_ipa` applies `data/lat/phonology.txt`. For each modern target T,
`respell_T` applies `data/lat/respelling/T.txt` to IPA and `direct_T` gives
the same respelling straight from the Latin spelling.
"""

from .direct_it import direct_it
from .ipa import to_ipa
from .respell_it import respell_it

SOURCE = "lat"

RESPELL = {"it": direct_it}
RESPELL_IPA = {"it": respell_it}

__all__ = ["RESPELL", "RESPELL_IPA", "SOURCE", "direct_it", "respell_it", "to_ipa"]
