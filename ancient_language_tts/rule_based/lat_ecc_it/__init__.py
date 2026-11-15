"""Rule-based Ecclesiastical Latin → Modern Italian.

`to_ipa` applies `data/lat-ecc/phonology.txt`.
`respell` applies `data/lat-ecc/respelling/it.txt` to that IPA.
"""

from .direct import to_italian
from .ipa import to_ipa
from .respelling import respell

SOURCE = "lat-ecc"
TARGET = "it"


def convert(text: str) -> tuple[str, str]:
    """Return `(ipa, italian_respelling)` for Ecclesiastical Latin `text`."""
    ipa = to_ipa(text)
    return ipa, respell(ipa)


__all__ = ["SOURCE", "TARGET", "convert", "respell", "to_ipa", "to_italian"]
