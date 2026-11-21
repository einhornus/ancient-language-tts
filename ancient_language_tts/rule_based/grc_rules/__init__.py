"""Rule-based Koine Greek (first century AD).

`to_ipa` applies `data/grc/phonology.txt`.
`respell_el.respell` (exported as `ipa_to_modern_greek`) applies `data/grc/respelling/el.txt` to that IPA.
`direct_el.to_modern_greek` gives the same Modern Greek respelling straight
from the Greek spelling, with no IPA step.
"""

from .direct_el import to_modern_greek
from .ipa import to_ipa
from .respell_el import respell as ipa_to_modern_greek

SOURCE = "grc"
TARGETS = ("el",)


def convert(text: str) -> tuple[str, str]:
    """Return `(ipa, modern_greek_respelling)` for Koine Greek `text`."""
    ipa = to_ipa(text)
    return ipa, ipa_to_modern_greek(ipa)


__all__ = ["SOURCE", "TARGETS", "convert", "ipa_to_modern_greek", "to_ipa", "to_modern_greek"]
