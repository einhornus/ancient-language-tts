"""Classical Latin. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "qu": "kʷ", "gu": "gʷ", "ph": "pʰ", "th": "tʰ", "ch": "kʰ",
    "ae": "ae̯", "oe": "oe̯", "au": "au̯",
    "ā": "aː", "ē": "eː", "ī": "iː", "ō": "oː", "ū": "uː", "ȳ": "yː",
    "c": "k", "v": "w", "x": "ks", "j": "j",
    "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ",
}

TO_ITALIAN = {
    "kʷ": "qu", "gʷ": "gu", "pʰ": "p", "tʰ": "t", "kʰ": "c",
    "ae̯": "ai", "oe̯": "oi", "au̯": "au",
    "k": "c", "w": "u", "y": "i",
    "ɛ": "e", "ɪ": "i", "ɔ": "o", "ʊ": "u", "ŋ": "n",
}

ENGINE = _mock.engine("lat", GRAPHEMES, {"it": TO_ITALIAN})
